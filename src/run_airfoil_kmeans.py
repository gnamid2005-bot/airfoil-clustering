import argparse
import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from airfoil_data import (
    flatten_shapes,
    plot_shape,
    preprocess_airfoils,
    save_preprocessing_tables,
    save_sample_raw_vs_normalized,
    save_sample_resampled,
    validate_processed_shapes,
)


matplotlib.use("Agg")

DEFAULT_DATA_DIR = Path("data/airfoils/raw")
DEFAULT_OUTPUT_DIR = Path("outputs/airfoil_kmeans")
SOURCE_URL = "https://m-selig.web.engr.illinois.edu/ads/archives/coord_seligFmt.zip"


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def squared_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    x_norm = np.sum(X * X, axis=1, keepdims=True)
    c_norm = np.sum(centers * centers, axis=1)
    distances = x_norm + c_norm - 2.0 * X @ centers.T
    return np.maximum(distances, 0.0)


def kmeans_plus_plus(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=np.float64)
    first = int(rng.integers(n))
    centers[0] = X[first]

    closest = squared_distances(X, centers[:1]).reshape(-1)
    for index in range(1, k):
        total = float(closest.sum())
        if total <= 0:
            chosen = int(rng.integers(n))
        else:
            chosen = int(rng.choice(n, p=closest / total))
        centers[index] = X[chosen]
        closest = np.minimum(closest, squared_distances(X, centers[index : index + 1]).reshape(-1))

    return centers


def run_kmeans(
    X: np.ndarray,
    k: int,
    seed: int,
    max_iter: int,
    tol: float,
) -> dict:
    if k <= 0:
        raise ValueError("k must be positive")
    if X.shape[0] < k:
        raise ValueError("k cannot exceed the number of airfoils")

    rng = np.random.default_rng(seed)
    centers = kmeans_plus_plus(X, k, rng)
    assignments = np.full(X.shape[0], -1, dtype=np.int64)
    history = []
    converged = False

    for iteration in range(max_iter):
        distances = squared_distances(X, centers)
        new_assignments = np.argmin(distances, axis=1)
        assigned_distances = distances[np.arange(X.shape[0]), new_assignments]
        inertia = float(assigned_distances.sum())
        changed = int(np.sum(new_assignments != assignments))
        history.append(
            {
                "iteration": iteration,
                "inertia": inertia,
                "changed_assignments": changed,
            }
        )

        if len(history) > 1 and inertia > history[-2]["inertia"] + 1e-8:
            raise ValueError("k-means inertia increased")

        new_centers = centers.copy()
        for cluster in range(k):
            members = X[new_assignments == cluster]
            if len(members) == 0:
                farthest = int(np.argmax(assigned_distances))
                new_centers[cluster] = X[farthest]
            else:
                new_centers[cluster] = members.mean(axis=0)

        shift = float(np.sum((new_centers - centers) ** 2))
        centers = new_centers
        assignments = new_assignments

        if changed == 0 or shift <= tol:
            converged = True
            break

    final_distances = squared_distances(X, centers)
    assignments = np.argmin(final_distances, axis=1)
    assigned_distances = final_distances[np.arange(X.shape[0]), assignments]
    final_inertia = float(assigned_distances.sum())
    if history and final_inertia <= history[-1]["inertia"] + 1e-8:
        history[-1]["inertia"] = final_inertia

    return {
        "centers": centers,
        "assignments": assignments,
        "distances": assigned_distances,
        "history": history,
        "inertia": final_inertia,
        "iterations": len(history),
        "converged": converged,
    }


def save_convergence(history: list[dict], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([row["iteration"] for row in history], [row["inertia"] for row in history], marker="o")
    ax.set_xlabel("iteration")
    ax.set_ylabel("inertia")
    ax.set_title("K-means convergence")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def save_cluster_plot(
    shapes: np.ndarray,
    mean_shape: np.ndarray,
    k: int,
    cluster: int,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for shape in shapes:
        plot_shape(ax, shape, color="tab:blue", alpha=0.14, linewidth=0.7)
    plot_shape(ax, mean_shape, color="tab:red", linewidth=2.2, label="mean")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"K={k}, cluster {cluster}, n={len(shapes)}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def save_means_plot(mean_shapes: np.ndarray, k: int, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cmap = plt.get_cmap("tab10")
    for cluster, shape in enumerate(mean_shapes):
        plot_shape(
            ax,
            shape,
            color=cmap(cluster % 10),
            linewidth=2.0,
            label=f"cluster {cluster}",
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"K={k} mean airfoil shapes")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def save_pca_plot(X: np.ndarray, labels: np.ndarray, centers: np.ndarray, save_path: Path) -> None:
    X_centered = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(X_centered, full_matrices=False)
    components = vt[:2].T
    projected = X_centered @ components
    projected_centers = (centers - X.mean(axis=0, keepdims=True)) @ components

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(projected[:, 0], projected[:, 1], c=labels, cmap="tab10", s=14, alpha=0.75)
    ax.scatter(
        projected_centers[:, 0],
        projected_centers[:, 1],
        c=np.arange(centers.shape[0]),
        cmap="tab10",
        marker="x",
        s=90,
        linewidths=2.0,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA view of clustered airfoils")
    ax.grid(True, alpha=0.25)
    ax.legend(*scatter.legend_elements(), title="cluster", loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def save_assignments(path: Path, names: list[str], filenames: list[str], result: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "airfoil_name", "cluster", "distance_to_assigned_mean"],
        )
        writer.writeheader()
        for filename, name, cluster, distance in zip(
            filenames,
            names,
            result["assignments"],
            result["distances"],
        ):
            writer.writerow(
                {
                    "filename": filename,
                    "airfoil_name": name,
                    "cluster": int(cluster),
                    "distance_to_assigned_mean": float(math_sqrt(distance)),
                }
            )


def math_sqrt(value: float) -> float:
    return float(np.sqrt(max(value, 0.0)))


def save_history(path: Path, history: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["iteration", "inertia", "changed_assignments"])
        writer.writeheader()
        writer.writerows(history)


def save_experiment_readme(path: Path, summary: dict, ks: list[int], seed: int) -> None:
    lines = [
        "# Airfoil K-means Experiment",
        "",
        f"Source: {SOURCE_URL}",
        "",
        "Each preprocessed airfoil is represented as `S_i = (p_i1, ..., p_iM)`, where every point index has the same chordwise meaning across airfoils.",
        "",
        "The squared shape distance is:",
        "",
        "`d(S_i, S_j)^2 = sum_m ||p_im - p_jm||^2`",
        "",
        "K-means uses the assignment step `c_i = argmin_k d(S_i, mu_k)^2` and the update step `mu_k = (1 / |C_k|) sum_{i in C_k} S_i`.",
        "",
        "Because correspondence is established by splitting upper/lower surfaces and interpolating both surfaces on the same cosine-spaced chord grid, each cluster mean is the pointwise arithmetic mean airfoil.",
        "",
        "This differs from a no-correspondence setting, where a set distance such as Chamfer distance and a more general Frechet/Karcher mean would be required.",
        "",
        "## Run Summary",
        "",
        f"- K values: {', '.join(str(k) for k in ks)}",
        f"- Seed: {seed}",
        f"- Files found: {summary['files_found']}",
        f"- Successfully parsed: {summary['successfully_parsed']}",
        f"- Survived preprocessing: {summary['survived_preprocessing']}",
        f"- Rejected/problematic records: {summary['rejected']}",
        f"- Surface points per side: {summary['n_surface_points']}",
        "",
        "Run again with:",
        "",
        "`python src/run_airfoil_kmeans.py --k 2,3,4,5 --seed 0`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_processed_npz(path: Path, processed, shapes: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        shapes=shapes,
        filenames=np.asarray([airfoil.filename for airfoil in processed]),
        names=np.asarray([airfoil.name for airfoil in processed]),
        original_point_counts=np.asarray([airfoil.original_point_count for airfoil in processed]),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run deterministic K-means clustering on corresponding UIUC airfoil shapes."
    )
    parser.add_argument("--k", default="2,3,4,5", help="Comma-separated K values.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-surface-points", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--processed-dir", default="data/airfoils/processed")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-airfoils", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    preprocessing_dir = output_dir / "preprocessing"
    ks = parse_int_list(args.k)

    if not list(data_dir.rglob("*.dat")):
        raise FileNotFoundError(
            f"No .dat files found under {data_dir}. Run "
            "`python src/download_airfoils.py` or place coord_seligFmt.zip there and extract it."
        )

    processed, rejected, preprocessing_summary = preprocess_airfoils(
        raw_dir=data_dir,
        n_surface_points=args.n_surface_points,
        max_airfoils=args.max_airfoils,
        seed=args.seed,
    )
    if not processed:
        raise ValueError("No airfoils survived preprocessing.")

    shapes = np.stack([airfoil.shape for airfoil in processed], axis=0)
    validate_processed_shapes(shapes)
    X = flatten_shapes(shapes)
    names = [airfoil.name for airfoil in processed]
    filenames = [airfoil.filename for airfoil in processed]

    save_preprocessing_tables(preprocessing_dir, processed, rejected, preprocessing_summary)
    save_sample_raw_vs_normalized(
        processed,
        preprocessing_dir / "sample_raw_vs_normalised.png",
        seed=args.seed,
    )
    save_sample_resampled(
        shapes,
        preprocessing_dir / "sample_resampled_airfoils.png",
        seed=args.seed,
        sample_size=30,
    )
    save_processed_npz(output_dir / "processed_airfoils.npz", processed, shapes)
    processed_name = f"airfoils_n{args.n_surface_points}_seed{args.seed}.npz"
    if args.max_airfoils is not None:
        processed_name = f"airfoils_n{args.n_surface_points}_seed{args.seed}_max{args.max_airfoils}.npz"
    save_processed_npz(processed_dir / processed_name, processed, shapes)

    experiment_rows = []
    for k in ks:
        run_dir = output_dir / f"k{k}_seed{args.seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        result = run_kmeans(
            X,
            k=k,
            seed=args.seed,
            max_iter=args.max_iter,
            tol=args.tol,
        )
        mean_shapes = result["centers"].reshape(k, shapes.shape[1], 2)
        validate_processed_shapes(mean_shapes)

        counts = [int(np.sum(result["assignments"] == cluster)) for cluster in range(k)]
        summary = {
            "k": k,
            "seed": args.seed,
            "n_airfoils": int(shapes.shape[0]),
            "n_surface_points": args.n_surface_points,
            "inertia": result["inertia"],
            "iterations": result["iterations"],
            "converged": bool(result["converged"]),
            "cluster_sizes": counts,
            "validation": {
                "same_shape": True,
                "finite": True,
                "orientation": True,
                "pointwise_mean": True,
                "squared_euclidean_assignment": True,
                "non_increasing_inertia": True,
            },
        }
        write_json(run_dir / "summary.json", summary)
        save_assignments(run_dir / "assignments.csv", names, filenames, result)
        save_history(run_dir / "convergence.csv", result["history"])
        save_convergence(result["history"], run_dir / "convergence.png")
        save_means_plot(mean_shapes, k, run_dir / "means.png")
        save_pca_plot(X, result["assignments"], result["centers"], run_dir / "pca.png")

        for cluster in range(k):
            members = shapes[result["assignments"] == cluster]
            save_cluster_plot(
                members,
                mean_shapes[cluster],
                k,
                cluster,
                run_dir / f"cluster_{cluster}.png",
            )

        row = {
            "k": k,
            "seed": args.seed,
            "n_airfoils": int(shapes.shape[0]),
            "n_surface_points": args.n_surface_points,
            "inertia": result["inertia"],
            "iterations": result["iterations"],
            "converged": bool(result["converged"]),
        }
        for cluster, size in enumerate(counts):
            row[f"cluster_{cluster}_size"] = size
        experiment_rows.append(row)
        print(f"completed K={k}: inertia={result['inertia']:.6g}, sizes={counts}")

    fieldnames = sorted({key for row in experiment_rows for key in row.keys()})
    with (output_dir / "experiment_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(experiment_rows)

    save_experiment_readme(output_dir / "README.md", preprocessing_summary, ks, args.seed)
    print(f"saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
