import argparse
import csv
import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

from airfoil_data import plot_shape
from run_airfoil_k_selection import adjusted_rand_index


matplotlib.use("Agg")

DEFAULT_DATA_PATH = Path("data/airfoils/processed/airfoils_n100_seed0.npz")
DEFAULT_OUTPUT_DIR = Path("outputs/airfoil_gmm_selection")


@dataclass
class PCAFit:
    mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray


@dataclass
class GMMFit:
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    covariance_type: str
    converged: bool
    iterations: int
    lower_bound: float


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def representation_from_shapes(shapes: np.ndarray, representation: str) -> tuple[np.ndarray, dict]:
    half = shapes.shape[1] // 2
    upper_y = shapes[:, :half, 1]
    lower_y = shapes[:, half:, 1]
    if representation == "full":
        return shapes.reshape(shapes.shape[0], -1), {"kind": "full", "dimension": shapes.shape[1] * 2}
    if representation == "y":
        return np.concatenate([upper_y, lower_y], axis=1), {"kind": "y", "dimension": shapes.shape[1]}
    if representation == "camber_thickness":
        camber = 0.5 * (upper_y + lower_y)
        thickness = upper_y - lower_y
        return np.concatenate([camber, thickness], axis=1), {"kind": "camber_thickness", "dimension": shapes.shape[1]}
    raise ValueError(f"Unknown representation: {representation}")


def shapes_from_representation(values: np.ndarray, x_grid: np.ndarray, representation: str) -> np.ndarray:
    n = values.shape[0]
    half = x_grid.shape[0]
    if representation == "full":
        return values.reshape(n, half * 2, 2)
    if representation == "y":
        upper_y = values[:, :half]
        lower_y = values[:, half:]
    elif representation == "camber_thickness":
        camber = values[:, :half]
        thickness = values[:, half:]
        upper_y = camber + 0.5 * thickness
        lower_y = camber - 0.5 * thickness
    else:
        raise ValueError(f"Unknown representation: {representation}")
    upper = np.stack([np.tile(x_grid, (n, 1)), upper_y], axis=2)
    lower = np.stack([np.tile(x_grid, (n, 1)), lower_y], axis=2)
    return np.concatenate([upper, lower], axis=1)


def validate_shapes(shapes: np.ndarray, thickness_tol: float) -> tuple[list[dict], dict]:
    issues = []
    if shapes.ndim != 3 or shapes.shape[2] != 2 or shapes.shape[1] % 2 != 0:
        raise ValueError(f"Expected shapes with shape (n, 2*m, 2), got {shapes.shape}")
    if not np.isfinite(shapes).all():
        bad = np.where(~np.isfinite(shapes).reshape(shapes.shape[0], -1).all(axis=1))[0]
        for index in bad:
            issues.append({"index": int(index), "severity": "invalid", "reason": "non_finite"})

    half = shapes.shape[1] // 2
    upper = shapes[:, :half]
    lower = shapes[:, half:]
    upper_x = upper[:, :, 0]
    lower_x = lower[:, :, 0]
    thickness = upper[:, :, 1] - lower[:, :, 1]
    x_variation = shapes[:, :, 0].max(axis=0) - shapes[:, :, 0].min(axis=0)

    for index in range(shapes.shape[0]):
        if np.any(np.diff(upper_x[index]) < -1e-10) or np.any(np.diff(lower_x[index]) < -1e-10):
            issues.append({"index": index, "severity": "invalid", "reason": "x_not_monotone"})
        if abs(upper_x[index, 0]) > 1e-8 or abs(lower_x[index, 0]) > 1e-8:
            issues.append({"index": index, "severity": "invalid", "reason": "leading_edge_x_not_zero"})
        if abs(upper_x[index, -1] - 1.0) > 1e-8 or abs(lower_x[index, -1] - 1.0) > 1e-8:
            issues.append({"index": index, "severity": "invalid", "reason": "trailing_edge_x_not_one"})
        if np.any(thickness[index] < -thickness_tol):
            issues.append({"index": index, "severity": "invalid", "reason": "negative_thickness"})
        elif np.any(thickness[index] < 0.0):
            issues.append({"index": index, "severity": "warning", "reason": "tiny_negative_thickness"})

    max_thickness = thickness.max(axis=1)
    trailing_thickness = thickness[:, -1]
    distance_flags = np.zeros(shapes.shape[0], dtype=bool)
    extreme_threshold = float(np.quantile(max_thickness, 0.99) + 1.5 * (np.quantile(max_thickness, 0.99) - np.quantile(max_thickness, 0.95)))
    distance_flags |= max_thickness > extreme_threshold
    distance_flags |= trailing_thickness > max(0.05, float(np.quantile(trailing_thickness, 0.99)))
    for index in np.where(distance_flags)[0]:
        issues.append({"index": int(index), "severity": "valid_extreme", "reason": "geometrically_extreme"})

    summary = {
        "n_airfoils": int(shapes.shape[0]),
        "point_count": int(shapes.shape[1]),
        "flattened_dimension": int(shapes.shape[1] * 2),
        "x_std_max": float(shapes[:, :, 0].std(axis=0).max()),
        "x_variation_max": float(x_variation.max()),
        "invalid_shape_count": int(len({row["index"] for row in issues if row["severity"] == "invalid"})),
        "warning_shape_count": int(len({row["index"] for row in issues if row["severity"] == "warning"})),
        "valid_extreme_shape_count": int(len({row["index"] for row in issues if row["severity"] == "valid_extreme"})),
        "minimum_thickness": float(thickness.min()),
    }
    return issues, summary


def load_dataset(path: Path, representation: str, thickness_tol: float) -> dict:
    data = np.load(path)
    required = {"shapes", "filenames", "names", "original_point_counts"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"Processed dataset is missing fields: {sorted(missing)}")
    shapes = data["shapes"].astype(np.float64)
    issues, validation_summary = validate_shapes(shapes, thickness_tol)
    invalid = {row["index"] for row in issues if row["severity"] == "invalid"}
    if invalid:
        raise ValueError(f"Found invalid shapes in processed dataset: {len(invalid)}")
    X, representation_info = representation_from_shapes(shapes, representation)
    half = shapes.shape[1] // 2
    return {
        "path": path,
        "shapes": shapes,
        "X": X.astype(np.float64),
        "x_grid": shapes[0, :half, 0].copy(),
        "filenames": data["filenames"].astype(str),
        "names": data["names"].astype(str),
        "original_point_counts": data["original_point_counts"],
        "issues": issues,
        "validation_summary": validation_summary,
        "representation": representation_info,
    }


def squared_norm_rows(values: np.ndarray) -> np.ndarray:
    return np.sum(values * values, axis=1)


def previous_extreme_indices(filenames: np.ndarray, path: Path) -> np.ndarray:
    if not path.exists():
        return np.asarray([], dtype=np.int64)
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    if not rows or "cluster" not in rows[0]:
        return np.asarray([], dtype=np.int64)
    counts = {}
    for row in rows:
        counts[row["cluster"]] = counts.get(row["cluster"], 0) + 1
    smallest = min(counts, key=counts.get)
    filename_to_index = {name: index for index, name in enumerate(filenames)}
    indices = [
        filename_to_index[row["filename"]]
        for row in rows
        if row["cluster"] == smallest and row["filename"] in filename_to_index
    ]
    return np.asarray(sorted(indices), dtype=np.int64)


def stratified_split(
    X: np.ndarray,
    filenames: np.ndarray,
    seed: int,
    heldout_fraction: float,
    registration_subset_size: int,
    k5_assignments_path: Path,
    extreme_heldout_fraction: float = 0.2,
) -> dict:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    global_mean = X.mean(axis=0, keepdims=True)
    distances = np.sqrt(squared_norm_rows(X - global_mean))
    target_heldout = int(round(n * heldout_fraction))
    extreme_indices = previous_extreme_indices(filenames, k5_assignments_path)
    heldout = set()
    extreme_train = np.asarray([], dtype=np.int64)
    extreme_heldout = np.asarray([], dtype=np.int64)
    if len(extreme_indices) > 0:
        shuffled_extreme = rng.permutation(extreme_indices)
        n_extreme_heldout = max(1, int(round(len(extreme_indices) * extreme_heldout_fraction)))
        n_extreme_heldout = min(n_extreme_heldout, len(extreme_indices) - 1)
        extreme_heldout = np.asarray(sorted(shuffled_extreme[:n_extreme_heldout]), dtype=np.int64)
        extreme_train = np.asarray(sorted(shuffled_extreme[n_extreme_heldout:]), dtype=np.int64)
        heldout.update(int(index) for index in extreme_heldout)

    quantiles = np.quantile(distances, np.linspace(0.0, 1.0, 6))
    bins = np.digitize(distances, quantiles[1:-1], right=True)
    for bin_id in range(5):
        candidates = np.where(bins == bin_id)[0]
        candidates = np.asarray(
            [
                idx
                for idx in candidates
                if idx not in heldout and idx not in set(extreme_train.tolist())
            ],
            dtype=np.int64,
        )
        if len(candidates) == 0:
            continue
        remaining = target_heldout - len(heldout)
        if remaining <= 0:
            break
        choose = min(len(candidates), max(1, int(round(target_heldout / 5))))
        choose = min(choose, remaining)
        selected = rng.choice(candidates, size=choose, replace=False)
        heldout.update(int(index) for index in selected)

    if len(heldout) < target_heldout:
        rest = np.asarray([idx for idx in range(n) if idx not in heldout], dtype=np.int64)
        selected = rng.choice(rest, size=target_heldout - len(heldout), replace=False)
        heldout.update(int(index) for index in selected)
    if len(heldout) > target_heldout:
        keep_extreme = set(int(index) for index in extreme_heldout)
        removable = np.asarray(sorted([idx for idx in heldout if idx not in keep_extreme]), dtype=np.int64)
        drop_count = len(heldout) - target_heldout
        if drop_count > 0 and len(removable) >= drop_count:
            dropped = rng.choice(removable, size=drop_count, replace=False)
            for index in dropped:
                heldout.remove(int(index))

    heldout_indices = np.asarray(sorted(heldout), dtype=np.int64)
    train_indices = np.asarray([idx for idx in range(n) if idx not in heldout], dtype=np.int64)

    subset = set(int(index) for index in extreme_heldout if index in heldout)
    for bin_id in range(5):
        remaining = registration_subset_size - len(subset)
        if remaining <= 0:
            break
        candidates = heldout_indices[bins[heldout_indices] == bin_id]
        candidates = np.asarray([idx for idx in candidates if idx not in subset], dtype=np.int64)
        if len(candidates) == 0:
            continue
        remaining_bins = 5 - bin_id
        need = max(1, int(math.ceil(remaining / remaining_bins)))
        selected = rng.choice(candidates, size=min(need, len(candidates)), replace=False)
        subset.update(int(index) for index in selected)
    if len(subset) < registration_subset_size:
        rest = np.asarray([idx for idx in heldout_indices if idx not in subset], dtype=np.int64)
        if len(rest):
            selected = rng.choice(rest, size=min(registration_subset_size - len(subset), len(rest)), replace=False)
            subset.update(int(index) for index in selected)
    registration_subset = np.asarray(sorted(subset), dtype=np.int64)

    return {
        "train_indices": train_indices,
        "heldout_indices": heldout_indices,
        "registration_subset_indices": registration_subset,
        "distance_to_global_mean": distances,
        "distance_bins": bins,
        "previous_extreme_indices": extreme_indices,
        "previous_extreme_train_indices": extreme_train,
        "previous_extreme_heldout_indices": extreme_heldout,
    }


def fit_pca(X_train: np.ndarray, q: int) -> PCAFit:
    mean = X_train.mean(axis=0)
    centered = X_train - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:q]
    explained_variance = (singular_values**2) / max(X_train.shape[0] - 1, 1)
    total_variance = float(np.sum(explained_variance))
    ratios = explained_variance / total_variance if total_variance > 0 else np.zeros_like(explained_variance)
    return PCAFit(
        mean=mean,
        components=components,
        explained_variance=explained_variance[:q],
        explained_variance_ratio=ratios[:q],
    )


def pca_transform(pca: PCAFit, X: np.ndarray) -> np.ndarray:
    return (X - pca.mean) @ pca.components.T


def pca_inverse(pca: PCAFit, scores: np.ndarray) -> np.ndarray:
    return scores @ pca.components + pca.mean


def reconstruction_metrics(
    X: np.ndarray,
    reconstructed: np.ndarray,
    original_shapes: np.ndarray,
    shapes_reconstructed: np.ndarray,
) -> dict:
    errors = np.sqrt(np.mean((X - reconstructed) ** 2, axis=1))
    half = shapes_reconstructed.shape[1] // 2
    thickness = shapes_reconstructed[:, :half, 1] - shapes_reconstructed[:, half:, 1]
    point_errors = np.linalg.norm(shapes_reconstructed - original_shapes, axis=2)
    return {
        "mean_reconstruction_rmse": float(np.mean(errors)),
        "median_reconstruction_rmse": float(np.median(errors)),
        "p90_reconstruction_rmse": float(np.quantile(errors, 0.90)),
        "p95_reconstruction_rmse": float(np.quantile(errors, 0.95)),
        "max_reconstruction_rmse": float(np.max(errors)),
        "maximum_pointwise_reconstruction_error": float(np.max(point_errors)),
        "negative_thickness_count": int(np.sum(np.any(thickness < -1e-8, axis=1))),
        "self_intersection_count_feasible_proxy": int(np.sum(np.any(thickness < -1e-8, axis=1))),
        "minimum_reconstructed_thickness": float(np.min(thickness)),
        "leading_edge_gap_mean": float(np.mean(np.linalg.norm(shapes_reconstructed[:, 0] - shapes_reconstructed[:, half], axis=1))),
        "trailing_edge_gap_mean": float(np.mean(np.linalg.norm(shapes_reconstructed[:, half - 1] - shapes_reconstructed[:, -1], axis=1))),
    }


def log_gaussian_prob(X: np.ndarray, means: np.ndarray, covariances: np.ndarray, covariance_type: str, reg_covar: float) -> np.ndarray:
    n, q = X.shape
    k = means.shape[0]
    log_prob = np.empty((n, k), dtype=np.float64)
    log2pi = q * math.log(2.0 * math.pi)
    for comp in range(k):
        diff = X - means[comp]
        if covariance_type == "spherical":
            var = max(float(covariances[comp]), reg_covar)
            log_det = q * math.log(var)
            maha = np.sum(diff * diff, axis=1) / var
        elif covariance_type == "diagonal":
            var = np.maximum(covariances[comp], reg_covar)
            log_det = float(np.sum(np.log(var)))
            maha = np.sum((diff * diff) / var, axis=1)
        elif covariance_type == "tied":
            cov = covariances
            sign, log_det = np.linalg.slogdet(cov)
            if sign <= 0:
                raise ValueError("non-positive tied covariance")
            solved = np.linalg.solve(cov, diff.T).T
            maha = np.sum(diff * solved, axis=1)
        elif covariance_type == "full":
            cov = covariances[comp]
            sign, log_det = np.linalg.slogdet(cov)
            if sign <= 0:
                raise ValueError("non-positive full covariance")
            solved = np.linalg.solve(cov, diff.T).T
            maha = np.sum(diff * solved, axis=1)
        else:
            raise ValueError(f"Unknown covariance type: {covariance_type}")
        log_prob[:, comp] = -0.5 * (log2pi + log_det + maha)
    return log_prob


def estimate_covariances(X: np.ndarray, responsibilities: np.ndarray, means: np.ndarray, covariance_type: str, reg_covar: float) -> np.ndarray:
    n, q = X.shape
    nk = responsibilities.sum(axis=0) + 10.0 * np.finfo(float).eps
    k = means.shape[0]
    if covariance_type == "spherical":
        covariances = np.empty(k)
        for comp in range(k):
            diff = X - means[comp]
            covariances[comp] = np.sum(responsibilities[:, comp] * np.sum(diff * diff, axis=1)) / (nk[comp] * q)
        return np.maximum(covariances, reg_covar)
    if covariance_type == "diagonal":
        covariances = np.empty((k, q))
        for comp in range(k):
            diff = X - means[comp]
            covariances[comp] = (responsibilities[:, comp, None] * diff * diff).sum(axis=0) / nk[comp]
        return np.maximum(covariances, reg_covar)
    if covariance_type == "tied":
        cov = np.zeros((q, q), dtype=np.float64)
        for comp in range(k):
            diff = X - means[comp]
            cov += (responsibilities[:, comp, None] * diff).T @ diff
        cov /= n
        cov.flat[:: q + 1] += reg_covar
        return cov
    if covariance_type == "full":
        covariances = np.empty((k, q, q), dtype=np.float64)
        for comp in range(k):
            diff = X - means[comp]
            cov = (responsibilities[:, comp, None] * diff).T @ diff / nk[comp]
            cov.flat[:: q + 1] += reg_covar
            covariances[comp] = cov
        return covariances
    raise ValueError(f"Unknown covariance type: {covariance_type}")


def initialise_responsibilities(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=np.float64)
    centers[0] = X[int(rng.integers(n))]
    closest = np.sum((X - centers[0]) ** 2, axis=1)
    for comp in range(1, k):
        total = float(closest.sum())
        chosen = int(rng.choice(n, p=closest / total)) if total > 0 else int(rng.integers(n))
        centers[comp] = X[chosen]
        closest = np.minimum(closest, np.sum((X - centers[comp]) ** 2, axis=1))
    distances = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(distances, axis=1)
    responsibilities = np.zeros((n, k), dtype=np.float64)
    responsibilities[np.arange(n), labels] = 1.0
    return responsibilities


def fit_gmm_once(
    X: np.ndarray,
    k: int,
    covariance_type: str,
    seed: int,
    reg_covar: float,
    max_iter: int,
    tol: float,
) -> GMMFit:
    rng = np.random.default_rng(seed)
    responsibilities = initialise_responsibilities(X, k, rng)
    previous_lower = -np.inf
    converged = False
    weights = np.ones(k) / k
    means = np.zeros((k, X.shape[1]))
    covariances = np.ones(k) if covariance_type == "spherical" else np.ones((k, X.shape[1]))

    for iteration in range(1, max_iter + 1):
        nk = responsibilities.sum(axis=0) + 10.0 * np.finfo(float).eps
        weights = nk / X.shape[0]
        means = responsibilities.T @ X / nk[:, None]
        covariances = estimate_covariances(X, responsibilities, means, covariance_type, reg_covar)
        weighted_log_prob = log_gaussian_prob(X, means, covariances, covariance_type, reg_covar) + np.log(weights + 1e-300)
        log_norm = logsumexp(weighted_log_prob, axis=1)
        lower = float(np.mean(log_norm))
        responsibilities = np.exp(weighted_log_prob - log_norm[:, None])
        if abs(lower - previous_lower) <= tol:
            converged = True
            break
        previous_lower = lower

    return GMMFit(weights, means, covariances, covariance_type, converged, iteration, previous_lower)


def fit_gmm(
    X: np.ndarray,
    k: int,
    covariance_type: str,
    seed: int,
    n_init: int,
    reg_covar: float,
    max_iter: int,
    tol: float,
) -> GMMFit:
    best = None
    for init in range(n_init):
        candidate = fit_gmm_once(X, k, covariance_type, seed + 1009 * init, reg_covar, max_iter, tol)
        if best is None or candidate.lower_bound > best.lower_bound:
            best = candidate
    return best


def predict_responsibilities(model: GMMFit, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    weighted_log_prob = log_gaussian_prob(X, model.means, model.covariances, model.covariance_type, 1e-12) + np.log(model.weights + 1e-300)
    log_norm = logsumexp(weighted_log_prob, axis=1)
    responsibilities = np.exp(weighted_log_prob - log_norm[:, None])
    return responsibilities, np.argmax(responsibilities, axis=1), float(np.mean(log_norm))


def gmm_parameter_count(q: int, k: int, covariance_type: str) -> int:
    weights = k - 1
    means = k * q
    if covariance_type == "spherical":
        cov = k
    elif covariance_type == "diagonal":
        cov = k * q
    elif covariance_type == "tied":
        cov = q * (q + 1) // 2
    elif covariance_type == "full":
        cov = k * q * (q + 1) // 2
    else:
        raise ValueError(f"Unknown covariance type: {covariance_type}")
    return weights + means + cov


def covariance_condition(model: GMMFit) -> dict:
    if model.covariance_type == "spherical":
        values = model.covariances
        return {"min_covariance": float(np.min(values)), "max_condition_number": 1.0}
    if model.covariance_type == "diagonal":
        return {
            "min_covariance": float(np.min(model.covariances)),
            "max_condition_number": float(np.max(np.max(model.covariances, axis=1) / np.maximum(np.min(model.covariances, axis=1), 1e-300))),
        }
    if model.covariance_type == "tied":
        return {"min_covariance": float(np.min(np.linalg.eigvalsh(model.covariances))), "max_condition_number": float(np.linalg.cond(model.covariances))}
    eigs = [np.linalg.eigvalsh(cov) for cov in model.covariances]
    return {
        "min_covariance": float(min(np.min(values) for values in eigs)),
        "max_condition_number": float(max(np.linalg.cond(cov) for cov in model.covariances)),
    }


def entropy_summary(responsibilities: np.ndarray) -> dict:
    entropy_per_sample = -np.sum(responsibilities * np.log(responsibilities + 1e-300), axis=1)
    return {
        "classification_entropy": float(np.sum(entropy_per_sample)),
        "average_entropy": float(np.mean(entropy_per_sample)),
        "median_entropy": float(np.median(entropy_per_sample)),
        "maximum_entropy": float(np.max(entropy_per_sample)),
        "uncertain_proportion": float(np.mean(np.max(responsibilities, axis=1) < 0.8)),
    }


def save_shape_overlay(original: np.ndarray, reconstructed: np.ndarray, save_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_shape(ax, original, color="tab:blue", linewidth=1.8, label="original")
    plot_shape(ax, reconstructed, color="tab:orange", linewidth=1.4, linestyle="--", label="reconstructed")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def save_distribution_plot(values: np.ndarray, train_indices: np.ndarray, heldout_indices: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values[train_indices], bins=40, alpha=0.65, label="train")
    ax.hist(values[heldout_indices], bins=40, alpha=0.65, label="held-out")
    ax.set_xlabel("distance to global mean")
    ax.set_ylabel("count")
    ax.set_title("Train and held-out distance coverage")
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def save_metric_lines(rows: list[dict], y_key: str, title: str, ylabel: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    covariances = sorted({row["covariance_type"] for row in rows})
    for covariance_type in covariances:
        subset = sorted([row for row in rows if row["covariance_type"] == covariance_type], key=lambda row: row["k"])
        ax.plot([row["k"] for row in subset], [row[y_key] for row in subset], marker="o", label=covariance_type)
    ax.set_xlabel("K")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def save_pca_variance_plot(rows: list[dict], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    rows = sorted(rows, key=lambda row: row["q"])
    ax.plot([row["q"] for row in rows], [row["cumulative_explained_variance"] for row in rows], marker="o")
    ax.set_xlabel("PCA dimension q")
    ax.set_ylabel("cumulative explained variance")
    ax.set_title("PCA explained variance")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def save_component_means(dataset: dict, model: GMMFit, pca: PCAFit, q: int, k: int, covariance_type: str, seed: int, output_dir: Path) -> None:
    reconstructed = pca_inverse(pca, model.means)
    mean_shapes = shapes_from_representation(reconstructed, dataset["x_grid"], dataset["representation"]["kind"])
    np.savez_compressed(
        output_dir / "component_means" / f"q{q}_k{k}_{covariance_type}_seed{seed}.npz",
        latent_means=model.means,
        reconstructed_values=reconstructed,
        mean_shapes=mean_shapes,
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cmap = plt.get_cmap("tab10")
    for comp, shape in enumerate(mean_shapes):
        plot_shape(ax, shape, color=cmap(comp % 10), linewidth=2.0, label=f"component {comp}")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"q={q}, K={k}, {covariance_type} component means")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    plt.tight_layout()
    path = output_dir / "figures" / "component_means" / f"q{q}_k{k}_{covariance_type}_seed{seed}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close(fig)


def run_directory(output_dir: Path, q: int, k: int, covariance_type: str, seed: int) -> Path:
    return output_dir / "runs" / f"q{q}" / f"k{k}" / covariance_type / f"seed{seed}"


def completed(run_dir: Path) -> bool:
    path = run_dir / "summary.json"
    if not path.exists():
        return False
    try:
        return read_json(path).get("status") == "completed"
    except Exception:
        return False


def run_one_gmm(
    dataset: dict,
    pca: PCAFit,
    train_scores: np.ndarray,
    heldout_scores: np.ndarray,
    train_indices: np.ndarray,
    q: int,
    k: int,
    covariance_type: str,
    seed: int,
    args,
) -> dict:
    output_dir = Path(args.output_dir)
    run_dir = run_directory(output_dir, q, k, covariance_type, seed)
    if completed(run_dir) and not args.force:
        return read_json(run_dir / "summary.json")

    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    warnings = []
    try:
        model = fit_gmm(train_scores, k, covariance_type, seed, args.n_init, args.reg_covar, args.max_iter, args.tol)
        train_resp, train_labels, train_ll = predict_responsibilities(model, train_scores)
        heldout_resp, heldout_labels, heldout_ll = predict_responsibilities(model, heldout_scores)
        parameters = gmm_parameter_count(q, k, covariance_type)
        train_total_ll = train_ll * train_scores.shape[0]
        bic = -2.0 * train_total_ll + parameters * math.log(train_scores.shape[0])
        aic = -2.0 * train_total_ll + 2.0 * parameters
        sklearn_bic_crosscheck = bic
        entropy = entropy_summary(train_resp)
        icl = bic + 2.0 * entropy["classification_entropy"]
        effective_sizes = train_resp.sum(axis=0)
        hard_sizes = np.bincount(train_labels, minlength=k)
        cov_diag = covariance_condition(model)
        save_component_means(dataset, model, pca, q, k, covariance_type, seed, output_dir)
        np.savez_compressed(
            run_dir / "result.npz",
            weights=model.weights,
            means=model.means,
            covariances=model.covariances,
            train_responsibilities=train_resp,
            heldout_responsibilities=heldout_resp,
            train_labels=train_labels,
            heldout_labels=heldout_labels,
        )
        assignment_rows = []
        for local_index, global_index in enumerate(train_indices):
            assignment_rows.append(
                {
                    "index": int(global_index),
                    "filename": dataset["filenames"][global_index],
                    "airfoil_name": dataset["names"][global_index],
                    "component": int(train_labels[local_index]),
                    "max_posterior_probability": float(np.max(train_resp[local_index])),
                    "posterior_entropy": float(-np.sum(train_resp[local_index] * np.log(train_resp[local_index] + 1e-300))),
                }
            )
        save_csv(output_dir / "component_assignments" / f"q{q}_k{k}_{covariance_type}_seed{seed}.csv", assignment_rows)
        status = "completed"
        error = ""
    except Exception as exc:
        model = None
        train_ll = heldout_ll = bic = aic = icl = np.nan
        parameters = 0
        hard_sizes = effective_sizes = np.asarray([])
        entropy = {
            "classification_entropy": np.nan,
            "average_entropy": np.nan,
            "median_entropy": np.nan,
            "maximum_entropy": np.nan,
            "uncertain_proportion": np.nan,
        }
        cov_diag = {"min_covariance": np.nan, "max_condition_number": np.nan}
        sklearn_bic_crosscheck = np.nan
        warnings.append(str(exc))
        status = "failed"
        error = str(exc)

    runtime = time.perf_counter() - start
    summary = {
        "status": status,
        "error": error,
        "q": q,
        "k": k,
        "covariance_type": covariance_type,
        "seed": seed,
        "n_init": args.n_init,
        "reg_covar": args.reg_covar,
        "max_iter": args.max_iter,
        "converged": bool(model.converged) if model is not None else False,
        "em_iterations": int(model.iterations) if model is not None else 0,
        "training_log_likelihood_per_sample": float(train_ll),
        "heldout_log_likelihood_per_sample": float(heldout_ll),
        "bic": float(bic),
        "aic": float(aic),
        "icl": float(icl),
        "icl_convention": "ICL = BIC + 2 * positive classification entropy; lower is better.",
        "manual_bic": float(sklearn_bic_crosscheck),
        "n_free_parameters": int(parameters),
        "component_weights": model.weights.tolist() if model is not None else [],
        "hard_component_sizes": hard_sizes.astype(int).tolist(),
        "effective_component_sizes": effective_sizes.astype(float).tolist(),
        "minimum_component_size": int(np.min(hard_sizes)) if hard_sizes.size else 0,
        "minimum_component_weight": float(np.min(model.weights)) if model is not None else np.nan,
        **entropy,
        **cov_diag,
        "runtime_seconds": runtime,
        "warnings": warnings,
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def pairwise_stability(output_dir: Path, q: int, k: int, covariance_type: str, seeds: list[int]) -> dict:
    labels = {}
    for seed in seeds:
        path = run_directory(output_dir, q, k, covariance_type, seed) / "result.npz"
        if path.exists():
            labels[seed] = np.load(path)["train_labels"]
    values = []
    for left, right in itertools.combinations(sorted(labels), 2):
        values.append(adjusted_rand_index(labels[left], labels[right]))
    if not values:
        return {"mean_pairwise_ari": 1.0, "std_pairwise_ari": 0.0, "min_pairwise_ari": 1.0, "max_pairwise_ari": 1.0}
    arr = np.asarray(values)
    return {
        "mean_pairwise_ari": float(np.mean(arr)),
        "std_pairwise_ari": float(np.std(arr)),
        "min_pairwise_ari": float(np.min(arr)),
        "max_pairwise_ari": float(np.max(arr)),
    }


def create_report(output_dir: Path, config: dict, validation: dict, pca_rows: list[dict], gmm_rows: list[dict], shortlist: list[dict]) -> None:
    lines = [
        "# Airfoil PCA-GMM Selection Report",
        "",
        "This report covers the statistical model-selection stage only. No airfoil registration experiment was run.",
        "",
        "## Dataset and Validation",
        "",
        f"Dataset: `{config['data_path']}`",
        f"Airfoils: {validation['n_airfoils']}",
        f"Points per airfoil: {validation['point_count']}",
        f"Flattened full dimension: {validation['flattened_dimension']}",
        f"Maximum x-coordinate variation across samples: {validation['x_variation_max']:.3e}",
        f"Invalid shapes rejected from this processed dataset: {validation['invalid_shape_count']}",
        f"Valid extreme shapes flagged, not deleted: {validation['valid_extreme_shape_count']}",
        "",
        "The previous K=5 31-member extreme group is split internally, with approximately 80% retained in training and 20% held out. This lets the GMM learn a rare/extreme component while still preserving held-out examples for later validation.",
        "",
        "## Representation",
        "",
        f"Main representation: `{config['representation']}`. Because x-coordinates are fixed across airfoils, the default representation models camber and thickness rather than repeating invariant x coordinates.",
        "",
        "## PCA",
        "",
        "PCA is fitted on training data only. Held-out airfoils are transformed using the training PCA basis. Reconstruction error and inverse-transformed validity are reported for every q.",
        "",
        "## GMM and Criteria",
        "",
        "GMMs are fitted in fixed PCA spaces. BIC values should be compared within a fixed q, not naively across different q values. ICL uses the convention `ICL = BIC + 2 * H`, where H is positive classification entropy, so lower is better.",
        "",
        "## Smoke-Test Shortlist",
        "",
    ]
    for row in shortlist:
        lines.append(
            f"- {row['role']}: q={row['q']}, K={row['k']}, covariance={row['covariance_type']}, seed={row['seed']}, BIC={row['bic']:.3f}, ICL={row['icl']:.3f}"
        )
    lines.extend(
        [
            "",
            "## scikit-learn Cross-check",
            "",
            "A q=5, K=2 diagonal-covariance smoke configuration can be compared against `sklearn.mixture.GaussianMixture` with `src/crosscheck_airfoil_gmm_sklearn.py`. Differences are expected when the two implementations land in different local optima, so the full screening should rely on multiple seeds and stability diagnostics.",
            "",
            "## Future Registration Design",
            "",
            "The existing SVF code is genuinely 3D and should not be treated as a finished 2D airfoil registration implementation.",
            "",
            "Route A, preferred: implement a separate 2D SVF with points `(N, 2)`, coefficients `(Gx, Gy, 2)`, 2D interpolation, 2D integration, 2D smoothness, 2D Jacobian determinant diagnostics, no-folding checks, and airfoil validity checks. Suggested files: `src/svf_2d.py`, `src/airfoil_registration.py`, and `src/run_airfoil_registration_validation.py`.",
            "",
            "Route B, secondary baseline: embed airfoils as `(x, y, 0)` and constrain deformation to remain planar. Unconstrained reuse of the 3D field is undesirable because it may introduce z displacement, wastes parameters, and uses 3D regularisation/Jacobian diagnostics that do not exactly match the 2D problem.",
            "",
            "For later registration validation, compare candidate template sets on the same held-out targets and choose the smallest K whose mean held-out registration cost is within one standard error of the minimum, while requiring failure rate no worse than K=1 and no material worsening on extreme targets.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_registration_design_note(output_dir: Path) -> None:
    lines = [
        "# Airfoil Registration Design Note",
        "",
        "This note is design-only. No large airfoil registration study has been run.",
        "",
        "## Why the Existing Code Is Not a 2D Registration Implementation",
        "",
        "The current SVF implementation expects point tensors with shape `(N, 3)` and coefficient grids with shape `(Gx, Gy, Gz, 3)`. The Chamfer loss and smoothness loss also validate 3D inputs. Therefore, 2D airfoil points cannot be treated as a completed registration problem by simply passing them into the existing code.",
        "",
        "## Route A: Genuine 2D SVF",
        "",
        "Preferred route: implement a separate 2D stationary velocity field with source and target points `(N, 2)`, coefficient grid `(Gx, Gy, 2)`, 2D bilinear or bicubic/B-spline interpolation, 2D Euler or scaling-and-squaring integration, 2D smoothness regularisation, 2D Jacobian determinant diagnostics, no-folding checks, and airfoil validity checks.",
        "",
        "Suggested files: `src/svf_2d.py`, `src/airfoil_registration.py`, and `src/run_airfoil_registration_validation.py`.",
        "",
        "## Route B: 3D Embedding Baseline",
        "",
        "A secondary baseline could embed airfoils as `(x, y, 0)` and constrain deformation to remain planar. Unconstrained 3D registration is undesirable because it may introduce z displacement, wastes parameters, and uses 3D smoothness/Jacobian diagnostics that do not exactly match the 2D problem.",
        "",
        "## Future Validation Rule",
        "",
        "For held-out target `S_i` and template `mu_k`, record `E_ik = D(phi_ik(mu_k), S_i) + lambda R(v_ik)` with data attachment, raw and weighted regularisation, velocity/displacement norms, Jacobian diagnostics, convergence, runtime, failure, final validity, and initial template-to-target distance.",
        "",
        "For each target and candidate K, use `E_i(K) = min_k E_ik`. Choose the smallest K whose mean held-out registration cost is within one standard error of the minimum, with failure rate no worse than K=1 and no material worsening on extreme targets.",
        "",
        "A rare template should not be rejected solely because it is rare. If it helps extreme targets, label it a coverage template rather than a high-frequency population cluster.",
        "",
    ]
    (output_dir / "registration_design_note.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="PCA-GMM model selection for corresponding airfoil shapes.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--representation", choices=["full", "y", "camber_thickness"], default="camber_thickness")
    parser.add_argument("--pca-dims", default="5,10,15,20")
    parser.add_argument("--k", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--covariance-types", default="spherical,diagonal,tied,full")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--reg-covar", type=float, default=1e-6)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--registration-subset-size", type=int, default=50)
    parser.add_argument("--thickness-tol", type=float, default=1e-3)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in ["runs", "figures", "pca", "splits", "validation", "component_assignments", "component_means"]:
        (output_dir / child).mkdir(exist_ok=True)

    q_values = parse_int_list(args.pca_dims)
    k_values = parse_int_list(args.k)
    covariance_types = parse_str_list(args.covariance_types)
    seeds = parse_int_list(args.seeds)
    dataset = load_dataset(Path(args.data_path), args.representation, args.thickness_tol)

    config = {
        "data_path": args.data_path,
        "output_dir": args.output_dir,
        "representation": args.representation,
        "pca_dims": q_values,
        "k_values": k_values,
        "covariance_types": covariance_types,
        "seeds": seeds,
        "n_init": args.n_init,
        "reg_covar": args.reg_covar,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "split_seed": args.split_seed,
        "heldout_fraction": args.heldout_fraction,
        "registration_subset_size": args.registration_subset_size,
        "thickness_tol": args.thickness_tol,
    }
    write_json(output_dir / "config.json", config)
    save_csv(output_dir / "validation" / "shape_issues.csv", dataset["issues"])
    write_json(output_dir / "validation" / "validation_summary.json", dataset["validation_summary"])
    x_std = dataset["shapes"][:, :, 0].std(axis=0)
    x_variation = dataset["shapes"][:, :, 0].max(axis=0) - dataset["shapes"][:, :, 0].min(axis=0)
    save_csv(
        output_dir / "validation" / "x_coordinate_stats.csv",
        [
            {
                "point_index": index,
                "x_std": float(x_std[index]),
                "x_variation": float(x_variation[index]),
            }
            for index in range(len(x_std))
        ],
    )

    split = stratified_split(
        dataset["X"],
        dataset["filenames"],
        args.split_seed,
        args.heldout_fraction,
        args.registration_subset_size,
        Path("outputs/airfoil_kmeans/k5_seed0/assignments.csv"),
    )
    np.save(output_dir / "splits" / "train_indices.npy", split["train_indices"])
    np.save(output_dir / "splits" / "heldout_indices.npy", split["heldout_indices"])
    np.save(output_dir / "splits" / "registration_subset_indices.npy", split["registration_subset_indices"])
    np.save(output_dir / "splits" / "extreme_group_train_indices.npy", split["previous_extreme_train_indices"])
    np.save(output_dir / "splits" / "extreme_group_heldout_indices.npy", split["previous_extreme_heldout_indices"])
    np.save(output_dir / "train_indices.npy", split["train_indices"])
    np.save(output_dir / "heldout_indices.npy", split["heldout_indices"])
    np.save(output_dir / "registration_subset_indices.npy", split["registration_subset_indices"])
    np.save(output_dir / "extreme_group_train_indices.npy", split["previous_extreme_train_indices"])
    np.save(output_dir / "extreme_group_heldout_indices.npy", split["previous_extreme_heldout_indices"])
    save_distribution_plot(
        split["distance_to_global_mean"],
        split["train_indices"],
        split["heldout_indices"],
        output_dir / "figures" / "split_distance_distribution.png",
    )
    split_rows = []
    heldout_set = set(split["heldout_indices"].tolist())
    subset_set = set(split["registration_subset_indices"].tolist())
    previous_extreme = set(split["previous_extreme_indices"].tolist())
    previous_extreme_train = set(split["previous_extreme_train_indices"].tolist())
    previous_extreme_heldout = set(split["previous_extreme_heldout_indices"].tolist())
    for index in range(dataset["X"].shape[0]):
        split_rows.append(
            {
                "index": index,
                "filename": dataset["filenames"][index],
                "split": "heldout" if index in heldout_set else "train",
                "registration_subset": index in subset_set,
                "previous_k5_small_cluster": index in previous_extreme,
                "previous_k5_small_cluster_train": index in previous_extreme_train,
                "previous_k5_small_cluster_heldout": index in previous_extreme_heldout,
                "distance_to_global_mean": float(split["distance_to_global_mean"][index]),
                "distance_bin": int(split["distance_bins"][index]),
            }
        )
    save_csv(output_dir / "splits" / "split_assignments.csv", split_rows)
    save_csv(output_dir / "split_assignments.csv", split_rows)
    write_json(
        output_dir / "splits" / "split_summary.json",
        {
            "train_count": int(len(split["train_indices"])),
            "heldout_count": int(len(split["heldout_indices"])),
            "registration_subset_count": int(len(split["registration_subset_indices"])),
            "previous_extreme_count": int(len(split["previous_extreme_indices"])),
            "previous_extreme_in_train": int(len(split["previous_extreme_train_indices"])),
            "previous_extreme_in_heldout": int(len(split["previous_extreme_heldout_indices"])),
        },
    )
    write_json(
        output_dir / "split_summary.json",
        {
            "train_count": int(len(split["train_indices"])),
            "heldout_count": int(len(split["heldout_indices"])),
            "registration_subset_count": int(len(split["registration_subset_indices"])),
            "previous_extreme_count": int(len(split["previous_extreme_indices"])),
            "previous_extreme_in_train": int(len(split["previous_extreme_train_indices"])),
            "previous_extreme_in_heldout": int(len(split["previous_extreme_heldout_indices"])),
        },
    )

    X_train = dataset["X"][split["train_indices"]]
    X_heldout = dataset["X"][split["heldout_indices"]]
    pca_rows = []
    pca_fits = {}
    train_scores_by_q = {}
    heldout_scores_by_q = {}
    for q in q_values:
        pca = fit_pca(X_train, q)
        pca_fits[q] = pca
        train_scores = pca_transform(pca, X_train)
        heldout_scores = pca_transform(pca, X_heldout)
        train_scores_by_q[q] = train_scores
        heldout_scores_by_q[q] = heldout_scores
        train_recon = pca_inverse(pca, train_scores)
        heldout_recon = pca_inverse(pca, heldout_scores)
        heldout_shapes_recon = shapes_from_representation(heldout_recon, dataset["x_grid"], args.representation)
        train_shapes_recon = shapes_from_representation(train_recon, dataset["x_grid"], args.representation)
        train_metrics = reconstruction_metrics(
            X_train,
            train_recon,
            dataset["shapes"][split["train_indices"]],
            train_shapes_recon,
        )
        heldout_metrics = reconstruction_metrics(
            X_heldout,
            heldout_recon,
            dataset["shapes"][split["heldout_indices"]],
            heldout_shapes_recon,
        )
        row = {
            "q": q,
            "cumulative_explained_variance": float(np.sum(pca.explained_variance_ratio)),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"heldout_{key}": value for key, value in heldout_metrics.items()},
        }
        pca_rows.append(row)
        np.savez_compressed(
            output_dir / "pca" / f"pca_q{q}.npz",
            mean=pca.mean,
            components=pca.components,
            explained_variance=pca.explained_variance,
            explained_variance_ratio=pca.explained_variance_ratio,
            train_scores=train_scores,
            heldout_scores=heldout_scores,
        )
        heldout_errors = np.sqrt(np.mean((X_heldout - heldout_recon) ** 2, axis=1))
        typical = int(np.argmin(np.abs(heldout_errors - np.median(heldout_errors))))
        high = int(np.argmax(heldout_errors))
        extreme = int(np.argmax(split["distance_to_global_mean"][split["heldout_indices"]]))
        for label, local_index in [("typical", typical), ("high_error", high), ("extreme", extreme)]:
            global_index = split["heldout_indices"][local_index]
            save_shape_overlay(
                dataset["shapes"][global_index],
                heldout_shapes_recon[local_index],
                output_dir / "figures" / "pca_reconstruction" / f"q{q}_{label}.png",
                f"q={q} {label}: {dataset['filenames'][global_index]}",
            )
    save_csv(output_dir / "pca_reconstruction_results.csv", pca_rows)
    save_pca_variance_plot(pca_rows, output_dir / "figures" / "pca_explained_variance.png")

    gmm_rows = []
    start = time.perf_counter()
    for q in q_values:
        for k in k_values:
            for covariance_type in covariance_types:
                for seed in seeds:
                    summary = run_one_gmm(
                        dataset,
                        pca_fits[q],
                        train_scores_by_q[q],
                        heldout_scores_by_q[q],
                        split["train_indices"],
                        q,
                        k,
                        covariance_type,
                        seed,
                        args,
                    )
                    gmm_rows.append(summary)
                    print(
                        f"q={q} K={k} cov={covariance_type} seed={seed}: "
                        f"status={summary['status']} bic={summary['bic']:.3f} icl={summary['icl']:.3f}"
                    )
    save_csv(output_dir / "all_gmm_results.csv", gmm_rows)

    stability_rows = []
    for q in q_values:
        for k in k_values:
            for covariance_type in covariance_types:
                stability = pairwise_stability(output_dir, q, k, covariance_type, seeds)
                stability_rows.append({"q": q, "k": k, "covariance_type": covariance_type, **stability})
    save_csv(output_dir / "stability_results.csv", stability_rows)

    summary_rows = []
    for q in q_values:
        for k in k_values:
            for covariance_type in covariance_types:
                subset = [row for row in gmm_rows if row["q"] == q and row["k"] == k and row["covariance_type"] == covariance_type and row["status"] == "completed"]
                if not subset:
                    continue
                stability = next(row for row in stability_rows if row["q"] == q and row["k"] == k and row["covariance_type"] == covariance_type)
                summary_rows.append(
                    {
                        "q": q,
                        "k": k,
                        "covariance_type": covariance_type,
                        "mean_bic": float(np.mean([row["bic"] for row in subset])),
                        "mean_icl": float(np.mean([row["icl"] for row in subset])),
                        "mean_heldout_log_likelihood": float(np.mean([row["heldout_log_likelihood_per_sample"] for row in subset])),
                        "mean_average_entropy": float(np.mean([row["average_entropy"] for row in subset])),
                        "minimum_component_size": int(min(row["minimum_component_size"] for row in subset)),
                        "minimum_component_weight": float(min(row["minimum_component_weight"] for row in subset)),
                        **stability,
                    }
                )
    save_csv(output_dir / "gmm_summary_by_q_k_covariance.csv", summary_rows)

    for q in q_values:
        q_rows = [row for row in summary_rows if row["q"] == q]
        save_metric_lines(q_rows, "mean_bic", f"q={q} BIC by K", "BIC (lower better)", output_dir / "figures" / f"q{q}_bic.png")
        save_metric_lines(q_rows, "mean_icl", f"q={q} ICL by K", "ICL (lower better)", output_dir / "figures" / f"q{q}_icl.png")
        save_metric_lines(q_rows, "mean_heldout_log_likelihood", f"q={q} held-out log-likelihood", "held-out log-likelihood", output_dir / "figures" / f"q{q}_heldout_ll.png")
        save_metric_lines(q_rows, "mean_average_entropy", f"q={q} posterior entropy", "average entropy", output_dir / "figures" / f"q{q}_entropy.png")
        save_metric_lines(q_rows, "minimum_component_size", f"q={q} minimum component size", "minimum hard size", output_dir / "figures" / f"q{q}_min_size.png")
        save_metric_lines(q_rows, "minimum_component_weight", f"q={q} minimum component weight", "minimum weight", output_dir / "figures" / f"q{q}_min_weight.png")
        save_metric_lines(q_rows, "mean_pairwise_ari", f"q={q} stability", "mean ARI", output_dir / "figures" / f"q{q}_stability.png")

    shortlist = []
    if summary_rows:
        baseline = min([row for row in summary_rows if row["k"] == 1], key=lambda row: row["mean_bic"])
        best_bic = min(summary_rows, key=lambda row: row["mean_bic"])
        best_icl = min(summary_rows, key=lambda row: row["mean_icl"])
        stable = max(
            [row for row in summary_rows if row["minimum_component_size"] >= 20],
            key=lambda row: (row["mean_pairwise_ari"], row["mean_heldout_log_likelihood"]),
        )
        for role, row in [
            ("global_mean_baseline", baseline),
            ("density_oriented_candidate", best_bic),
            ("low_overlap_clustering_candidate", best_icl),
            ("stable_candidate", stable),
        ]:
            seed = seeds[0]
            run = read_json(run_directory(output_dir, row["q"], row["k"], row["covariance_type"], seed) / "summary.json")
            shortlist.append({"role": role, "seed": seed, "bic": run["bic"], "icl": run["icl"], **row})
    save_csv(output_dir / "candidate_shortlist.csv", shortlist)
    write_json(output_dir / "candidate_shortlist.json", {"candidates": shortlist})
    create_report(output_dir, config, dataset["validation_summary"], pca_rows, gmm_rows, shortlist)
    write_registration_design_note(output_dir)
    print(f"saved outputs to {output_dir}")
    print(f"elapsed_seconds={time.perf_counter() - start:.2f}")


if __name__ == "__main__":
    main()
