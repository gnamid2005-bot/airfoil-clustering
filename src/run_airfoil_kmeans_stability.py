"""Standalone direct-coordinate K-means stability experiment for airfoils."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import time
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score


DEFAULT_INPUT = Path("data/airfoils/processed/airfoils_n100_seed0.npz")
DEFAULT_OUTPUT = Path("outputs/airfoil_kmeans_stability")
KS = list(range(2, 21))
SEEDS = list(range(20))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def load_and_validate(path: Path) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as archive:
        if "shapes" not in archive.files:
            raise ValueError("Input archive has no 'shapes' array")
        shapes = np.asarray(archive["shapes"], dtype=np.float64)
        keys = list(archive.files)
    if shapes.ndim != 3 or shapes.shape[2] != 2:
        raise ValueError(f"Expected (N, points, 2), got {shapes.shape}")
    finite = np.isfinite(shapes)
    x = shapes[:, :, 0]
    shared_x_exact = bool(np.array_equal(x, np.broadcast_to(x[0], x.shape)))
    x_variances = np.var(x, axis=0)
    n, points, coordinates = shapes.shape
    report = {
        "input_path": str(path),
        "input_sha256": sha256(path),
        "archive_keys": keys,
        "array_shape": [int(n), int(points), int(coordinates)],
        "n_airfoils": int(n),
        "points_per_airfoil": int(points),
        "upper_surface_points": int(points // 2),
        "lower_surface_points": int(points - points // 2),
        "all_samples_same_dimensionality": True,
        "flattened_dimension": int(points * coordinates),
        "flatten_order": "NumPy C order: [point0_x, point0_y, point1_x, point1_y, ...]",
        "x_coordinates_identical_across_airfoils_exact": shared_x_exact,
        "x_coordinate_dimensions_constant_exact": int(points if shared_x_exact else np.count_nonzero(np.all(x == x[:1], axis=0))),
        "x_coordinate_zero_variance_dimensions_from_np_var": int(np.count_nonzero(x_variances == 0.0)),
        "x_coordinate_max_variance": float(np.max(x_variances)),
        "pointwise_correspondence_established": bool(shared_x_exact and points % 2 == 0),
        "nan_count": int(np.isnan(shapes).sum()),
        "infinite_count": int(np.isinf(shapes).sum()),
        "all_values_finite": bool(finite.all()),
        "dtype_used": str(shapes.dtype),
    }
    if not report["all_values_finite"]:
        raise ValueError("Input contains NaN or infinite values")
    return shapes, report


def knee_by_chord(ks: np.ndarray, inertia: np.ndarray) -> tuple[int, np.ndarray]:
    x = (ks - ks.min()) / (ks.max() - ks.min())
    y = (inertia - inertia.min()) / (inertia.max() - inertia.min())
    x1, y1, x2, y2 = x[0], y[0], x[-1], y[-1]
    distances = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / math.hypot(y2 - y1, x2 - x1)
    return int(ks[int(np.argmax(distances))]), distances


def style_axis(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.22, color="#9CA3AF")
    ax.spines[["top", "right"]].set_visible(False)


def save_figures(output: Path, stability: list[dict], inertia: list[dict], sizes: list[dict]) -> None:
    fig_dir = output / "figures"
    fig_dir.mkdir()
    k = np.asarray([r["k"] for r in stability])
    mean_ari = np.asarray([r["mean_pairwise_ari"] for r in stability])
    std_ari = np.asarray([r["std_pairwise_ari"] for r in stability])
    min_ari = np.asarray([r["min_pairwise_ari"] for r in stability])

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.fill_between(k, np.clip(mean_ari - std_ari, -1, 1), np.clip(mean_ari + std_ari, -1, 1), color="#93C5FD", alpha=0.42, label="mean ± standard deviation")
    ax.plot(k, mean_ari, "o-", color="#1D4ED8", linewidth=2.0, label="mean pairwise ARI")
    ax.plot(k, min_ari, "v--", color="#C2410C", linewidth=1.3, markersize=5, label="minimum pairwise ARI")
    ax.set_xticks(k)
    ax.set_ylim(-0.05, 1.05)
    style_axis(ax, "Number of clusters K", "Adjusted Rand Index", "K-means Stability vs Number of Clusters")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "kmeans_stability_vs_k.png", dpi=220)
    plt.close(fig)

    ik = np.asarray([r["k"] for r in inertia])
    im = np.asarray([r["mean_inertia"] for r in inertia])
    istd = np.asarray([r["std_inertia"] for r in inertia])
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.fill_between(ik, im - istd, im + istd, color="#FDE68A", alpha=0.55, label="mean ± standard deviation")
    ax.plot(ik, im, "o-", color="#B45309", linewidth=2.0, label="mean inertia")
    ax.set_xticks(ik)
    style_axis(ax, "Number of clusters K", "Mean K-means inertia", "K-means Inertia vs Number of Clusters")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "kmeans_inertia_vs_k.png", dpi=220)
    plt.close(fig)

    rk = np.asarray([r["k"] for r in inertia if r["relative_inertia_improvement"] != ""])
    rv = np.asarray([r["relative_inertia_improvement"] for r in inertia if r["relative_inertia_improvement"] != ""], dtype=float)
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.plot(rk, rv, "o-", color="#7C3AED", linewidth=2.0)
    ax.set_xticks(ik)
    ax.yaxis.set_major_formatter(lambda x, pos: f"{100*x:.1f}%")
    style_axis(ax, "Number of clusters K", "Relative inertia improvement", "Relative Inertia Improvement vs K")
    fig.tight_layout()
    fig.savefig(fig_dir / "relative_inertia_improvement_vs_k.png", dpi=220)
    plt.close(fig)

    sk = np.asarray([r["k"] for r in sizes])
    sm = np.asarray([r["mean_min_cluster_size"] for r in sizes])
    smin = np.asarray([r["smallest_cluster_observed"] for r in sizes])
    smed = np.asarray([r["median_min_cluster_size"] for r in sizes])
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.plot(sk, sm, "o-", color="#0F766E", linewidth=2.0, label="mean minimum size")
    ax.plot(sk, smed, "s--", color="#2563EB", linewidth=1.3, label="median minimum size")
    ax.plot(sk, smin, "v--", color="#C2410C", linewidth=1.3, label="smallest observed")
    ax.axhline(10, color="#374151", linestyle=":", label="warning threshold: size < 10")
    ax.set_xticks(sk)
    style_axis(ax, "Number of clusters K", "Smallest cluster size", "Minimum Cluster Size vs K")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "minimum_cluster_size_vs_k.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), sharex=True)
    axes[0, 0].plot(k, mean_ari, "o-", color="#1D4ED8"); axes[0, 0].fill_between(k, np.clip(mean_ari-std_ari,-1,1), np.clip(mean_ari+std_ari,-1,1), color="#93C5FD", alpha=.4)
    style_axis(axes[0, 0], "", "Mean pairwise ARI", "Stability")
    axes[0, 1].plot(ik, im, "o-", color="#B45309"); axes[0, 1].fill_between(ik, im-istd, im+istd, color="#FDE68A", alpha=.5)
    style_axis(axes[0, 1], "", "Mean inertia", "Elbow")
    axes[1, 0].plot(rk, rv, "o-", color="#7C3AED"); axes[1, 0].yaxis.set_major_formatter(lambda x,pos:f"{100*x:.1f}%")
    style_axis(axes[1, 0], "Number of clusters K", "Relative improvement", "Diminishing returns")
    axes[1, 1].plot(sk, sm, "o-", color="#0F766E", label="mean minimum"); axes[1, 1].plot(sk, smin, "v--", color="#C2410C", label="smallest observed"); axes[1, 1].axhline(10,color="#374151",linestyle=":")
    style_axis(axes[1, 1], "Number of clusters K", "Smallest cluster size", "Cluster-size diagnostic"); axes[1, 1].legend(frameon=False)
    for ax in axes.flat: ax.set_xticks(ik)
    fig.suptitle("K-means Stability and Elbow Diagnostics", fontsize=15)
    fig.tight_layout()
    fig.savefig(fig_dir / "combined_diagnostics.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    shapes, input_report = load_and_validate(args.input)
    x_full = np.ascontiguousarray(shapes.reshape(shapes.shape[0], -1))
    x_y = np.ascontiguousarray(shapes[:, :, 1])
    labels_cube = np.empty((len(KS), len(SEEDS), shapes.shape[0]), dtype=np.int32)
    all_runs: list[dict] = []

    for ki, k in enumerate(KS):
        for si, seed in enumerate(SEEDS):
            model = KMeans(n_clusters=k, init="k-means++", n_init=1, max_iter=1000, tol=1e-6, random_state=seed, algorithm="lloyd")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                labels = model.fit_predict(x_full)
            convergence_warnings = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
            counts = np.bincount(labels, minlength=k)
            labels_cube[ki, si] = labels
            all_runs.append({
                "k": k, "seed": seed, "inertia": float(model.inertia_), "n_iterations": int(model.n_iter_),
                "converged": not convergence_warnings and model.n_iter_ < model.max_iter,
                "convergence_warning": " | ".join(convergence_warnings),
                "n_nonempty_clusters": int(np.count_nonzero(counts)), "cluster_sizes": json.dumps(counts.tolist()),
                "minimum_cluster_size": int(counts.min()), "maximum_cluster_size": int(counts.max()),
                "smallest_cluster_fraction": float(counts.min() / shapes.shape[0]),
                "flag_cluster_size_lt_10": bool(np.any(counts < 10)),
                "flag_cluster_fraction_lt_1pct": bool(np.any(counts / shapes.shape[0] < 0.01)),
            })

    pairwise: list[dict] = []
    stability: list[dict] = []
    inertia_summary: list[dict] = []
    size_summary: list[dict] = []
    for ki, k in enumerate(KS):
        values = []
        for a, b in combinations(range(len(SEEDS)), 2):
            ari = float(adjusted_rand_score(labels_cube[ki, a], labels_cube[ki, b]))
            pairwise.append({"k": k, "seed_a": SEEDS[a], "seed_b": SEEDS[b], "ari": ari})
            values.append(ari)
        av = np.asarray(values)
        stability.append({"k": k, "n_pairwise_comparisons": len(values), "mean_pairwise_ari": float(av.mean()), "std_pairwise_ari": float(av.std(ddof=0)), "median_pairwise_ari": float(np.median(av)), "min_pairwise_ari": float(av.min()), "max_pairwise_ari": float(av.max()), "p10_pairwise_ari": float(np.percentile(av, 10)), "p90_pairwise_ari": float(np.percentile(av, 90))})
        runs = [r for r in all_runs if r["k"] == k]
        iv = np.asarray([r["inertia"] for r in runs])
        inertia_summary.append({"k": k, "mean_inertia": float(iv.mean()), "std_inertia": float(iv.std(ddof=0)), "median_inertia": float(np.median(iv)), "min_inertia": float(iv.min()), "max_inertia": float(iv.max()), "relative_inertia_improvement": ""})
        mins = np.asarray([r["minimum_cluster_size"] for r in runs])
        maxs = np.asarray([r["maximum_cluster_size"] for r in runs])
        fractions = np.asarray([r["smallest_cluster_fraction"] for r in runs])
        size_summary.append({"k": k, "mean_min_cluster_size": float(mins.mean()), "smallest_cluster_observed": int(mins.min()), "median_min_cluster_size": float(np.median(mins)), "largest_cluster_size_observed": int(maxs.max()), "mean_smallest_cluster_fraction": float(fractions.mean()), "runs_with_cluster_size_lt_10": int(sum(r["flag_cluster_size_lt_10"] for r in runs)), "runs_with_cluster_fraction_lt_1pct": int(sum(r["flag_cluster_fraction_lt_1pct"] for r in runs)), "diagnostic_warning": bool(any(r["flag_cluster_size_lt_10"] or r["flag_cluster_fraction_lt_1pct"] for r in runs))})
    for i in range(1, len(inertia_summary)):
        prev, current = inertia_summary[i - 1]["mean_inertia"], inertia_summary[i]["mean_inertia"]
        inertia_summary[i]["relative_inertia_improvement"] = float((prev - current) / prev)
    knee_k, knee_distances = knee_by_chord(np.asarray(KS), np.asarray([r["mean_inertia"] for r in inertia_summary]))
    for row, distance in zip(inertia_summary, knee_distances): row["normalized_chord_knee_distance"] = float(distance)

    # Small sanity check: constant x dimensions imply identical Euclidean distances.
    rng = np.random.default_rng(20260818)
    pair_a = rng.integers(0, shapes.shape[0], 10000)
    pair_b = rng.integers(0, shapes.shape[0], 10000)
    full_d2 = np.sum((x_full[pair_a] - x_full[pair_b]) ** 2, axis=1)
    y_d2 = np.sum((x_y[pair_a] - x_y[pair_b]) ** 2, axis=1)
    sanity_model = KMeans(n_clusters=knee_k, init="k-means++", n_init=1, max_iter=1000, tol=1e-6, random_state=0, algorithm="lloyd").fit(x_y)
    sanity = {"sampled_pairs": 10000, "max_absolute_squared_distance_difference": float(np.max(np.abs(full_d2-y_d2))), "distance_arrays_exactly_equal": bool(np.array_equal(full_d2, y_d2)), "k_checked": knee_k, "seed_checked": 0, "full_vs_y_labels_ari": float(adjusted_rand_score(labels_cube[KS.index(knee_k), 0], sanity_model.labels_)), "full_inertia": float(next(r["inertia"] for r in all_runs if r["k"] == knee_k and r["seed"] == 0)), "y_only_inertia": float(sanity_model.inertia_)}

    write_csv(args.output / "kmeans_all_runs.csv", all_runs, list(all_runs[0]))
    write_csv(args.output / "kmeans_pairwise_ari.csv", pairwise, list(pairwise[0]))
    write_csv(args.output / "kmeans_stability_summary.csv", stability, list(stability[0]))
    write_csv(args.output / "kmeans_inertia_summary.csv", inertia_summary, list(inertia_summary[0]))
    write_csv(args.output / "kmeans_cluster_size_summary.csv", size_summary, list(size_summary[0]))
    np.savez_compressed(args.output / "kmeans_labels.npz", labels=labels_cube, ks=np.asarray(KS), seeds=np.asarray(SEEDS))
    save_figures(args.output, stability, inertia_summary, size_summary)

    runtime = time.perf_counter() - started
    config = {"experiment": "standalone direct-coordinate K-means stability", "input": input_report, "representation": {"primary": "full flattened corresponding shape coordinates", "dimension": int(x_full.shape[1]), "pca": False, "velocity_fields": False, "gmm": False, "registration": False}, "k_values": KS, "seeds": SEEDS, "kmeans": {"implementation": "sklearn.cluster.KMeans", "initialization": "k-means++", "n_init": 1, "max_iter": 1000, "tolerance": 1e-6, "algorithm": "lloyd", "random_state": "explicit seed 0..19"}, "stability": {"metric": "Adjusted Rand Index", "pairs_per_k": 190, "label_alignment": False, "std_ddof": 0}, "cluster_size_warnings": {"absolute_size_lt": 10, "fraction_lt": 0.01}, "knee": {"method": "maximum perpendicular distance from normalized mean-inertia curve to endpoint chord", "estimated_k": knee_k, "diagnostic_only": True}, "optional_y_only_sanity_check": sanity, "software": {"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__}, "runtime_seconds_before_report": runtime}
    json_dump(args.output / "experiment_config.json", config)
    json_dump(args.output / "input_validation.json", input_report)

    validation = {"exactly_19_k_values": len(KS) == 19, "exactly_20_runs_each_k": all(sum(r["k"] == k for r in all_runs) == 20 for k in KS), "total_fits_380": len(all_runs) == 380, "exactly_190_ari_each_k": all(sum(r["k"] == k for r in pairwise) == 190 for k in KS), "ari_finite": bool(np.isfinite([r["ari"] for r in pairwise]).all()), "inertia_finite": bool(np.isfinite([r["inertia"] for r in all_runs]).all()), "all_runs_exactly_k_nonempty_clusters": all(r["n_nonempty_clusters"] == r["k"] for r in all_runs), "all_runs_converged": all(r["converged"] for r in all_runs), "input_sha256_after": sha256(args.input), "input_unchanged": sha256(args.input) == input_report["input_sha256"]}
    json_dump(args.output / "validation.json", validation)
    print(json.dumps({"runtime_seconds": runtime, "knee_k": knee_k, "validation": validation, "sanity": sanity}, indent=2))


if __name__ == "__main__":
    main()
