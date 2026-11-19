import argparse
import csv
import itertools
import json
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from airfoil_data import flatten_shapes, plot_shape, validate_processed_shapes
from run_airfoil_kmeans import (
    run_kmeans,
    save_cluster_plot,
    save_means_plot,
    save_pca_plot,
    squared_distances,
)


matplotlib.use("Agg")

DEFAULT_PROCESSED_PATH = Path("data/airfoils/processed/airfoils_n100_seed0.npz")
DEFAULT_OUTPUT_DIR = Path("outputs/airfoil_k_selection")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_processed_dataset(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find processed airfoil data: {path}")
    data = np.load(path)
    shapes = data["shapes"].astype(np.float64)
    validate_processed_shapes(shapes)
    return {
        "shapes": shapes,
        "X": flatten_shapes(shapes),
        "filenames": data["filenames"].astype(str),
        "names": data["names"].astype(str),
        "original_point_counts": data["original_point_counts"],
    }


def run_directory(output_dir: Path, k: int, seed: int) -> Path:
    return output_dir / "runs" / f"k{k}_seed{seed}"


def completed_run(run_dir: Path) -> bool:
    summary_path = run_dir / "summary.json"
    result_path = run_dir / "result.npz"
    if not summary_path.exists() or not result_path.exists():
        return False
    try:
        return read_json(summary_path).get("status") == "completed"
    except Exception:
        return False


def comb2(values: np.ndarray) -> float:
    values = values.astype(np.float64)
    return float(np.sum(values * (values - 1.0) / 2.0))


def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    if labels_a.shape != labels_b.shape:
        raise ValueError("label arrays must have the same shape")
    n = labels_a.shape[0]
    if n < 2:
        return 1.0

    _, a_inverse = np.unique(labels_a, return_inverse=True)
    _, b_inverse = np.unique(labels_b, return_inverse=True)
    contingency = np.zeros((a_inverse.max() + 1, b_inverse.max() + 1), dtype=np.int64)
    np.add.at(contingency, (a_inverse, b_inverse), 1)

    sum_comb = comb2(contingency.reshape(-1))
    sum_a = comb2(contingency.sum(axis=1))
    sum_b = comb2(contingency.sum(axis=0))
    total = n * (n - 1.0) / 2.0
    expected = sum_a * sum_b / total if total > 0 else 0.0
    maximum = 0.5 * (sum_a + sum_b)
    denominator = maximum - expected
    if abs(denominator) <= 1e-12:
        return 1.0
    return float((sum_comb - expected) / denominator)


def silhouette_scores_squared(
    X: np.ndarray,
    labels: np.ndarray,
    sample_size: int | None,
    seed: int,
) -> dict:
    n = X.shape[0]
    if sample_size is not None and sample_size > 0 and sample_size < n:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(n, size=sample_size, replace=False))
        X_eval = X[indices]
        labels_eval = labels[indices]
        sampled = True
    else:
        indices = np.arange(n)
        X_eval = X
        labels_eval = labels
        sampled = False

    distances = squared_distances(X_eval, X_eval)
    unique_labels = np.unique(labels_eval)
    scores = np.zeros(X_eval.shape[0], dtype=np.float64)

    for index in range(X_eval.shape[0]):
        own = labels_eval[index]
        own_mask = labels_eval == own
        own_count = int(np.sum(own_mask))
        if own_count <= 1:
            scores[index] = 0.0
            continue

        a_value = float(np.sum(distances[index, own_mask]) / (own_count - 1))
        b_value = np.inf
        for other in unique_labels:
            if other == own:
                continue
            other_mask = labels_eval == other
            if np.any(other_mask):
                b_value = min(b_value, float(np.mean(distances[index, other_mask])))

        if not np.isfinite(b_value) or max(a_value, b_value) <= 1e-12:
            scores[index] = 0.0
        else:
            scores[index] = (b_value - a_value) / max(a_value, b_value)

    per_cluster = {}
    for label in unique_labels:
        mask = labels_eval == label
        per_cluster[int(label)] = {
            "mean": float(np.mean(scores[mask])) if np.any(mask) else 0.0,
            "size_in_silhouette_eval": int(np.sum(mask)),
        }

    return {
        "scores": scores,
        "indices": indices,
        "sampled": sampled,
        "sample_size": int(len(indices)),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "negative_proportion": float(np.mean(scores < 0.0)),
        "per_cluster": per_cluster,
    }


def geometric_descriptors(shapes: np.ndarray) -> dict[str, np.ndarray]:
    half = shapes.shape[1] // 2
    x = shapes[0, :half, 0]
    upper_y = shapes[:, :half, 1]
    lower_y = shapes[:, half:, 1]
    thickness = upper_y - lower_y
    camber = 0.5 * (upper_y + lower_y)
    max_thickness_index = np.argmax(thickness, axis=1)
    max_camber_index = np.argmax(np.abs(camber), axis=1)
    return {
        "maximum_thickness": thickness[np.arange(shapes.shape[0]), max_thickness_index],
        "x_maximum_thickness": x[max_thickness_index],
        "maximum_abs_camber": np.abs(camber[np.arange(shapes.shape[0]), max_camber_index]),
        "signed_camber_at_max_abs": camber[np.arange(shapes.shape[0]), max_camber_index],
        "x_maximum_abs_camber": x[max_camber_index],
        "trailing_edge_thickness": thickness[:, -1],
    }


def cluster_descriptor_rows(
    descriptors: dict[str, np.ndarray],
    labels: np.ndarray,
    k: int,
) -> list[dict]:
    rows = []
    for cluster in range(k):
        mask = labels == cluster
        row = {
            "cluster": cluster,
            "cluster_size": int(np.sum(mask)),
        }
        for name, values in descriptors.items():
            row[f"{name}_mean"] = float(np.mean(values[mask])) if np.any(mask) else 0.0
            row[f"{name}_std"] = float(np.std(values[mask])) if np.any(mask) else 0.0
        rows.append(row)
    return rows


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_assignments(path: Path, filenames, names, labels, distances) -> None:
    rows = []
    for filename, name, label, distance in zip(filenames, names, labels, distances):
        rows.append(
            {
                "filename": filename,
                "airfoil_name": name,
                "cluster": int(label),
                "distance_to_assigned_mean": float(np.sqrt(max(distance, 0.0))),
                "squared_distance_to_assigned_mean": float(distance),
            }
        )
    save_csv(path, rows)


def save_silhouette_plot(
    scores: np.ndarray,
    labels: np.ndarray,
    mean_score: float,
    k: int,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    y_lower = 10
    cmap = plt.get_cmap("tab10")
    for cluster in range(k):
        cluster_scores = np.sort(scores[labels == cluster])
        y_upper = y_lower + len(cluster_scores)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_scores,
            facecolor=cmap(cluster % 10),
            alpha=0.7,
        )
        ax.text(-0.05, y_lower + 0.5 * len(cluster_scores), str(cluster))
        y_lower = y_upper + 10
    ax.axvline(mean_score, color="tab:red", linestyle="--", linewidth=1.5, label="mean")
    ax.set_xlabel("silhouette value using squared Euclidean dissimilarity")
    ax.set_ylabel("airfoils grouped by cluster")
    ax.set_title(f"K={k} silhouette distribution")
    ax.set_xlim(-1.0, 1.0)
    ax.legend()
    ax.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def save_cluster_size_plot(counts: list[int], k: int, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(np.arange(k), counts, color="tab:blue", alpha=0.8)
    ax.set_xlabel("cluster")
    ax.set_ylabel("number of airfoils")
    ax.set_title(f"K={k} cluster sizes")
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def save_ari_heatmap(matrix: np.ndarray, seeds: list[int], k: int, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(seeds)), labels=seeds)
    ax.set_yticks(np.arange(len(seeds)), labels=seeds)
    ax.set_xlabel("seed")
    ax.set_ylabel("seed")
    ax.set_title(f"K={k} pairwise ARI")
    fig.colorbar(image, ax=ax, label="ARI")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def save_elbow_plot(summary_rows: list[dict], save_path: Path) -> None:
    rows = sorted(summary_rows, key=lambda row: row["k"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(
        [row["k"] for row in rows],
        [row["mean_inertia"] for row in rows],
        yerr=[row["std_inertia"] for row in rows],
        marker="o",
        capsize=3,
    )
    ax.set_xlabel("K")
    ax.set_ylabel("mean inertia across seeds")
    ax.set_title("Elbow diagnostic")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def save_metric_plot(summary_rows: list[dict], key: str, ylabel: str, title: str, save_path: Path) -> None:
    rows = sorted(summary_rows, key=lambda row: row["k"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([row["k"] for row in rows], [row[key] for row in rows], marker="o")
    ax.set_xlabel("K")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def run_one(
    dataset: dict,
    k: int,
    seed: int,
    args,
) -> dict:
    run_dir = run_directory(Path(args.output_dir), k, seed)
    if completed_run(run_dir) and not args.force:
        return read_json(run_dir / "summary.json")

    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    result = run_kmeans(
        dataset["X"],
        k=k,
        seed=seed,
        max_iter=args.max_iter,
        tol=args.tol,
    )
    elapsed = time.perf_counter() - start
    counts = [int(np.sum(result["assignments"] == cluster)) for cluster in range(k)]
    min_size = min(counts)
    max_size = max(counts)
    silhouette = silhouette_scores_squared(
        dataset["X"],
        result["assignments"],
        sample_size=args.silhouette_sample_size,
        seed=args.silhouette_seed,
    )

    np.savez_compressed(
        run_dir / "result.npz",
        centers=result["centers"],
        assignments=result["assignments"],
        squared_distances_to_assigned_mean=result["distances"],
        silhouette_scores=silhouette["scores"],
        silhouette_indices=silhouette["indices"],
    )
    save_assignments(
        run_dir / "assignments.csv",
        dataset["filenames"],
        dataset["names"],
        result["assignments"],
        result["distances"],
    )
    save_csv(run_dir / "convergence.csv", result["history"])

    summary = {
        "status": "completed",
        "k": k,
        "seed": seed,
        "n_airfoils": int(dataset["X"].shape[0]),
        "n_shape_coordinates": int(dataset["X"].shape[1]),
        "n_surface_points": int(dataset["shapes"].shape[1] // 2),
        "inertia": result["inertia"],
        "iterations": result["iterations"],
        "converged": bool(result["converged"]),
        "cluster_sizes": counts,
        "minimum_cluster_size": min_size,
        "maximum_cluster_size": max_size,
        "smallest_to_largest_cluster_ratio": float(min_size / max_size) if max_size else 0.0,
        "clusters_below_small_threshold": int(sum(size < args.small_cluster_threshold for size in counts)),
        "small_cluster_threshold": args.small_cluster_threshold,
        "silhouette_mean": silhouette["mean"],
        "silhouette_std": silhouette["std"],
        "silhouette_negative_proportion": silhouette["negative_proportion"],
        "silhouette_sample_size": silhouette["sample_size"],
        "silhouette_sampled": silhouette["sampled"],
        "silhouette_per_cluster": silhouette["per_cluster"],
        "runtime_seconds": elapsed,
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def load_run_result(output_dir: Path, k: int, seed: int) -> dict:
    run_dir = run_directory(output_dir, k, seed)
    summary = read_json(run_dir / "summary.json")
    arrays = np.load(run_dir / "result.npz")
    return {
        "summary": summary,
        "assignments": arrays["assignments"],
        "centers": arrays["centers"],
        "distances": arrays["squared_distances_to_assigned_mean"],
        "silhouette_scores": arrays["silhouette_scores"],
        "silhouette_indices": arrays["silhouette_indices"],
    }


def stability_rows(output_dir: Path, ks: list[int], seeds: list[int]) -> tuple[list[dict], dict[int, np.ndarray]]:
    rows = []
    matrices = {}
    for k in ks:
        assignments = {
            seed: load_run_result(output_dir, k, seed)["assignments"]
            for seed in seeds
        }
        matrix = np.eye(len(seeds), dtype=np.float64)
        for left_index, right_index in itertools.combinations(range(len(seeds)), 2):
            left_seed = seeds[left_index]
            right_seed = seeds[right_index]
            ari = adjusted_rand_index(assignments[left_seed], assignments[right_seed])
            matrix[left_index, right_index] = ari
            matrix[right_index, left_index] = ari
            rows.append(
                {
                    "k": k,
                    "seed_a": left_seed,
                    "seed_b": right_seed,
                    "adjusted_rand_index": ari,
                }
            )
        matrices[k] = matrix
    return rows, matrices


def summarise_by_k(
    output_dir: Path,
    ks: list[int],
    seeds: list[int],
    stability: list[dict],
    small_cluster_threshold: int,
) -> list[dict]:
    rows = []
    stability_by_k = {k: [] for k in ks}
    for row in stability:
        stability_by_k[row["k"]].append(row["adjusted_rand_index"])

    mean_inertia_by_k = {}
    for k in ks:
        summaries = [load_run_result(output_dir, k, seed)["summary"] for seed in seeds]
        inertias = np.asarray([row["inertia"] for row in summaries], dtype=np.float64)
        silhouettes = np.asarray([row["silhouette_mean"] for row in summaries], dtype=np.float64)
        min_sizes = [row["minimum_cluster_size"] for row in summaries]
        small_counts = [row["clusters_below_small_threshold"] for row in summaries]
        mean_inertia_by_k[k] = float(np.mean(inertias))
        ari_values = np.asarray(stability_by_k[k], dtype=np.float64)
        rows.append(
            {
                "k": k,
                "mean_inertia": float(np.mean(inertias)),
                "std_inertia": float(np.std(inertias)),
                "mean_silhouette": float(np.mean(silhouettes)),
                "std_silhouette": float(np.std(silhouettes)),
                "mean_pairwise_ari": float(np.mean(ari_values)) if len(ari_values) else 1.0,
                "std_pairwise_ari": float(np.std(ari_values)) if len(ari_values) else 0.0,
                "min_pairwise_ari": float(np.min(ari_values)) if len(ari_values) else 1.0,
                "max_pairwise_ari": float(np.max(ari_values)) if len(ari_values) else 1.0,
                "minimum_cluster_size_across_seeds": int(min(min_sizes)),
                "mean_clusters_below_small_threshold": float(np.mean(small_counts)),
                "small_cluster_threshold": small_cluster_threshold,
            }
        )

    for row in rows:
        k = row["k"]
        previous = mean_inertia_by_k.get(k - 1)
        if previous is None:
            row["inertia_reduction_from_previous"] = ""
            row["percent_reduction_from_previous"] = ""
        else:
            reduction = previous - row["mean_inertia"]
            row["inertia_reduction_from_previous"] = reduction
            row["percent_reduction_from_previous"] = 100.0 * reduction / previous

    reductions = {
        row["k"]: row["inertia_reduction_from_previous"]
        for row in rows
        if row["inertia_reduction_from_previous"] != ""
    }
    for row in rows:
        k = row["k"]
        next_reduction = reductions.get(k + 1)
        current_reduction = reductions.get(k)
        if current_reduction is None or next_reduction is None:
            row["second_difference"] = ""
        else:
            row["second_difference"] = current_reduction - next_reduction

    return rows


def select_candidate_ks(summary_rows: list[dict], max_candidates: int = 3) -> list[int]:
    rows = sorted(summary_rows, key=lambda row: row["k"])
    candidates = []

    second_diff_rows = [row for row in rows if row["second_difference"] != ""]
    if second_diff_rows:
        candidates.append(max(second_diff_rows, key=lambda row: row["second_difference"])["k"])

    candidates.append(max(rows, key=lambda row: row["mean_silhouette"])["k"])
    candidates.append(max(rows, key=lambda row: row["mean_pairwise_ari"])["k"])

    plateau_rows = [
        row
        for row in rows
        if row["percent_reduction_from_previous"] != ""
        and row["percent_reduction_from_previous"] < 10.0
    ]
    if plateau_rows:
        candidates.append(plateau_rows[0]["k"])

    unique = []
    for k in candidates:
        if k not in unique:
            unique.append(k)
    return unique[:max_candidates]


def representative_seed(output_dir: Path, k: int, seeds: list[int]) -> int:
    summaries = [load_run_result(output_dir, k, seed)["summary"] for seed in seeds]
    inertias = np.asarray([summary["inertia"] for summary in summaries])
    median = float(np.median(inertias))
    index = int(np.argmin(np.abs(inertias - median)))
    return seeds[index]


def save_candidate_visuals(
    dataset: dict,
    output_dir: Path,
    candidates: list[int],
    seeds: list[int],
    stability_matrices: dict[int, np.ndarray],
) -> list[dict]:
    rows = []
    descriptors = geometric_descriptors(dataset["shapes"])
    for k in candidates:
        seed = representative_seed(output_dir, k, seeds)
        loaded = load_run_result(output_dir, k, seed)
        labels = loaded["assignments"]
        centers = loaded["centers"]
        mean_shapes = centers.reshape(k, dataset["shapes"].shape[1], 2)
        figure_dir = output_dir / "figures" / f"k{k}_seed{seed}"
        figure_dir.mkdir(parents=True, exist_ok=True)
        save_means_plot(mean_shapes, k, figure_dir / "means.png")
        save_pca_plot(dataset["X"], labels, centers, figure_dir / "pca.png")
        save_cluster_size_plot(
            [int(np.sum(labels == cluster)) for cluster in range(k)],
            k,
            figure_dir / "cluster_sizes.png",
        )
        silhouette_indices = loaded["silhouette_indices"]
        save_silhouette_plot(
            loaded["silhouette_scores"],
            labels[silhouette_indices],
            float(np.mean(loaded["silhouette_scores"])),
            k,
            figure_dir / "silhouette.png",
        )
        save_ari_heatmap(stability_matrices[k], seeds, k, figure_dir / "ari_heatmap.png")
        for cluster in range(k):
            save_cluster_plot(
                dataset["shapes"][labels == cluster],
                mean_shapes[cluster],
                k,
                cluster,
                figure_dir / f"cluster_{cluster}.png",
            )
        descriptor_rows = cluster_descriptor_rows(descriptors, labels, k)
        save_csv(figure_dir / "descriptor_means_by_cluster.csv", descriptor_rows)
        for row in descriptor_rows:
            rows.append({"k": k, "seed": seed, **row})
    return rows


def write_report(
    output_dir: Path,
    config: dict,
    summary_rows: list[dict],
    candidates: list[int],
) -> None:
    rows = sorted(summary_rows, key=lambda row: row["k"])
    best_silhouette = max(rows, key=lambda row: row["mean_silhouette"])
    best_stability = max(rows, key=lambda row: row["mean_pairwise_ari"])
    second_rows = [row for row in rows if row["second_difference"] != ""]
    best_elbow = max(second_rows, key=lambda row: row["second_difference"]) if second_rows else None

    table_lines = [
        "| K | mean inertia | % reduction | silhouette | mean ARI | min size | small clusters |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        reduction = row["percent_reduction_from_previous"]
        reduction_text = "" if reduction == "" else f"{reduction:.2f}"
        table_lines.append(
            "| {k} | {inertia:.4f} | {reduction} | {sil:.4f} | {ari:.4f} | {min_size} | {small:.2f} |".format(
                k=row["k"],
                inertia=row["mean_inertia"],
                reduction=reduction_text,
                sil=row["mean_silhouette"],
                ari=row["mean_pairwise_ari"],
                min_size=row["minimum_cluster_size_across_seeds"],
                small=row["mean_clusters_below_small_threshold"],
            )
        )

    conclusion = (
        "当前运行结果更适合解释为候选 K 的诊断，而不是最终证明。"
        "若 elbow、silhouette、stability 和视觉解释不一致，应报告一个 K 范围。"
    )
    if best_elbow and best_elbow["k"] == best_silhouette["k"] == best_stability["k"]:
        conclusion = f"当前指标共同支持 K={best_silhouette['k']}，但仍建议用后续 registration cost 验证。"

    lines = [
        "# Airfoil K Selection Report",
        "",
        "## 1. Experimental question",
        "",
        "This study asks why one K-means assignment may be preferable, how to choose K, and whether the airfoil shapes look discretely clustered or continuously varying.",
        "",
        "## 2. Existing shape representation and distance",
        "",
        f"The processed dataset is `{config['processed_path']}`. It contains {config['n_airfoils']} airfoils. Each airfoil has {config['n_points_per_shape']} corresponding 2D points and is flattened to {config['n_shape_coordinates']} coordinates.",
        "",
        "K-means uses squared Euclidean distance on the flattened corresponding-point representation, equivalent to summing squared pointwise distances over corresponding airfoil locations.",
        "",
        "## 3. Why inertia alone cannot select K",
        "",
        "Inertia is guaranteed to decrease as K increases, so lower inertia alone rewards more clusters. It must be read with marginal improvement, stability, silhouette, cluster sizes, and visual interpretability.",
        "",
        "## 4. Elbow analysis",
        "",
        "The report computes `Delta J_K = J_{K-1} - J_K` and `Delta^2 J_K = Delta J_K - Delta J_{K+1}` from mean inertia across seeds. Large second difference suggests a possible elbow, but this is only a diagnostic because smooth or hierarchical data may have no sharp elbow.",
        "",
        f"Best K by second-difference elbow: {best_elbow['k'] if best_elbow else 'not available'}.",
        "",
        "## 5. Silhouette analysis",
        "",
        "Silhouette values are computed using the same squared-Euclidean dissimilarity used by clustering. High average silhouette suggests better separation; negative values indicate airfoils closer on average to another cluster than to their assigned cluster.",
        "",
        f"Best K by mean silhouette: {best_silhouette['k']}.",
        "",
        "## 6. Stability analysis",
        "",
        "Stability is measured with pairwise Adjusted Rand Index across random seeds. ARI accounts for label permutations, so clusters can be compared even when seed-specific cluster labels are renamed.",
        "",
        f"Best K by mean pairwise ARI: {best_stability['k']}.",
        "",
        "## 7. Cluster-size analysis",
        "",
        f"Small clusters are counted using threshold {config['small_cluster_threshold']}. Very small clusters may be meaningful rare families or may mostly isolate outliers, so they require visual inspection.",
        "",
        "## 8. Numerical summary",
        "",
        *table_lines,
        "",
        "## 9. Candidate K values",
        "",
        f"Selected for detailed visual inspection: {', '.join(str(k) for k in candidates)}.",
        "",
        "## 10. Limitations",
        "",
        "The correspondence is constructed by upper/lower surface splitting and chordwise interpolation. The analysis does not perform registration, probabilistic modelling, or no-correspondence shape clustering. PCA figures are visual diagnostics only, not the clustering representation.",
        "",
        "## 11. Recommended K or range",
        "",
        conclusion,
        "",
        "For downstream velocity-field registration, the best K may need validation by registration cost rather than unsupervised clustering diagnostics alone.",
        "",
    ]
    (output_dir / "best_k_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate K selection diagnostics for corresponding-point airfoil K-means."
    )
    parser.add_argument("--k", default="2,3,4,5,6,7,8,9,10")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--processed-path", default=str(DEFAULT_PROCESSED_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--small-cluster-threshold", type=int, default=50)
    parser.add_argument(
        "--silhouette-sample-size",
        type=int,
        default=3000,
        help="Use exact silhouette when n <= this value; use this many sampled airfoils otherwise.",
    )
    parser.add_argument("--silhouette-seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-visuals", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runs").mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    ks = parse_int_list(args.k)
    seeds = parse_int_list(args.seeds)
    dataset = load_processed_dataset(Path(args.processed_path))

    config = {
        "processed_path": args.processed_path,
        "output_dir": args.output_dir,
        "k_values": ks,
        "seeds": seeds,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "small_cluster_threshold": args.small_cluster_threshold,
        "silhouette_sample_size": args.silhouette_sample_size,
        "silhouette_seed": args.silhouette_seed,
        "n_airfoils": int(dataset["shapes"].shape[0]),
        "n_points_per_shape": int(dataset["shapes"].shape[1]),
        "n_shape_coordinates": int(dataset["X"].shape[1]),
    }
    write_json(output_dir / "config.json", config)

    run_rows = []
    start = time.perf_counter()
    for k in ks:
        for seed in seeds:
            summary = run_one(dataset, k, seed, args)
            run_rows.append(
                {
                    "k": summary["k"],
                    "seed": summary["seed"],
                    "inertia": summary["inertia"],
                    "iterations": summary["iterations"],
                    "converged": summary["converged"],
                    "minimum_cluster_size": summary["minimum_cluster_size"],
                    "maximum_cluster_size": summary["maximum_cluster_size"],
                    "smallest_to_largest_cluster_ratio": summary["smallest_to_largest_cluster_ratio"],
                    "clusters_below_small_threshold": summary["clusters_below_small_threshold"],
                    "silhouette_mean": summary["silhouette_mean"],
                    "silhouette_std": summary["silhouette_std"],
                    "silhouette_negative_proportion": summary["silhouette_negative_proportion"],
                    "runtime_seconds": summary["runtime_seconds"],
                    **{
                        f"cluster_{index}_size": size
                        for index, size in enumerate(summary["cluster_sizes"])
                    },
                }
            )
            print(
                f"K={k} seed={seed}: inertia={summary['inertia']:.6g}, "
                f"silhouette={summary['silhouette_mean']:.4f}"
            )

    save_csv(output_dir / "k_selection_results.csv", run_rows)
    pairwise_rows, stability_matrices = stability_rows(output_dir, ks, seeds)
    save_csv(output_dir / "pairwise_stability.csv", pairwise_rows)
    summary_rows = summarise_by_k(
        output_dir,
        ks,
        seeds,
        pairwise_rows,
        args.small_cluster_threshold,
    )
    save_csv(output_dir / "k_selection_summary.csv", summary_rows)
    candidates = select_candidate_ks(summary_rows)
    candidate_rows = [row for row in summary_rows if row["k"] in candidates]
    save_csv(output_dir / "candidate_k_summary.csv", candidate_rows)

    save_elbow_plot(summary_rows, output_dir / "figures" / "elbow.png")
    save_metric_plot(
        summary_rows,
        "mean_silhouette",
        "mean silhouette",
        "Silhouette by K",
        output_dir / "figures" / "silhouette_by_k.png",
    )
    save_metric_plot(
        summary_rows,
        "mean_pairwise_ari",
        "mean pairwise ARI",
        "Stability by K",
        output_dir / "figures" / "stability_by_k.png",
    )

    descriptor_rows = []
    if not args.skip_visuals:
        descriptor_rows = save_candidate_visuals(
            dataset,
            output_dir,
            candidates,
            seeds,
            stability_matrices,
        )
        save_csv(output_dir / "candidate_descriptor_means.csv", descriptor_rows)

    write_report(output_dir, config, summary_rows, candidates)
    elapsed = time.perf_counter() - start
    print(f"saved K-selection outputs to {output_dir}")
    print(f"elapsed_seconds={elapsed:.2f}")
    if run_rows:
        print(f"mean_seconds_per_run={elapsed / len(run_rows):.2f}")


if __name__ == "__main__":
    main()
