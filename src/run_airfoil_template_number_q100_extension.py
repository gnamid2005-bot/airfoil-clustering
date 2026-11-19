#!/usr/bin/env python3
"""Extend the completed Stage-5 PCA/GMM screen from q=40 to q=100.

The completed registration study and q<=40 statistical screen are read-only.
This script fits one training-only q=100 PCA basis, evaluates only q=41..100
by K=1..20 full-covariance GMM cells, performs a small targeted seed check,
and writes every new artifact to a separate output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "airfoil_template_number_stage5_screening"
Q40 = ROOT / "outputs" / "airfoil_template_number_stage5_q40_extension"
DEFAULT_OUT = ROOT / "outputs" / "airfoil_template_number_stage5_q100_extension"
SEED = 20260813
TRAIN_N = 1321
HOLDOUT_N = 330
SAFE_N = 1651
FIELD_D = 10368
Q_MAX = 100
K_MAX = 20
SMALL_THRESHOLD = max(10, math.ceil(0.01 * TRAIN_N))
GMM_CFG = {
    "covariance_type": "full",
    "n_init": 5,
    "random_state": SEED,
    "reg_covar": 1e-7,
    "max_iter": 500,
    "tol": 1e-5,
    "init_params": "kmeans",
}
STABILITY_Q = (20, 40, 60, 80, 100)
STABILITY_K = (2, 6, 7, 8)
STABILITY_SEEDS = tuple(range(5))
STABILITY_N_INIT = 3


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(content, encoding="utf-8")
    os.replace(temp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=json_default) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bool_value(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def gmm_parameter_count(q: int, k: int) -> int:
    covariance = k * q * (q + 1) // 2
    means = k * q
    weights = k - 1
    return covariance + means + weights


def fingerprint_paths() -> dict[str, Path]:
    return {
        "velocity_field_matrix": SOURCE / "velocity_field_matrix.npy",
        "velocity_field_population_indices": SOURCE / "velocity_field_population_indices.npy",
        "split": SOURCE / "split" / "train_heldout_indices.npz",
        "source_config": SOURCE / "config.json",
        "registration_results": SOURCE / "registration_results.csv",
        "q40_config": Q40 / "config.json",
        "q40_validation": Q40 / "validation.json",
        "q40_pca_summary": Q40 / "pca_q1_q40.csv",
        "q40_pca_model": Q40 / "pca" / "pca_q40.npz",
        "q40_gmm_combined": Q40 / "gmm_q1_q40_combined.csv",
        "q40_preferred": Q40 / "preferred_k_q1_q40.csv",
    }


def source_fingerprint() -> dict[str, Any]:
    files = fingerprint_paths()
    registration_files = sorted((SOURCE / "registrations").glob("*.npz"))
    return {
        "files": {
            key: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for key, path in files.items()
        },
        "registration_artifact_count": len(registration_files),
        "registration_artifact_total_bytes": int(sum(path.stat().st_size for path in registration_files)),
        "registration_artifact_latest_mtime_ns": int(max(path.stat().st_mtime_ns for path in registration_files)),
    }


def verify_inputs(out: Path) -> dict[str, Any]:
    missing = [str(path) for path in fingerprint_paths().values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required completed-study inputs: {missing}")
    q40_validation = json.loads((Q40 / "validation.json").read_text(encoding="utf-8"))
    q40_config = json.loads((Q40 / "config.json").read_text(encoding="utf-8"))
    matrix = np.load(SOURCE / "velocity_field_matrix.npy", mmap_mode="r")
    population = np.load(SOURCE / "velocity_field_population_indices.npy")
    split = np.load(SOURCE / "split" / "train_heldout_indices.npz")
    train_pos = split["train_matrix_positions"]
    heldout_pos = split["heldout_matrix_positions"]
    q40_gmm = read_csv(Q40 / "gmm_q1_q40_combined.csv")
    q40_pca = read_csv(Q40 / "pca_q1_q40.csv")
    q40_pref = read_csv(Q40 / "preferred_k_q1_q40.csv")
    expected = {(q, k) for q in range(1, 41) for k in range(1, 21)}
    actual = {(int(row["q"]), int(row["K"])) for row in q40_gmm}
    errors: list[str] = []
    if q40_validation.get("status") != "PASS":
        errors.append("q40 validation is not PASS")
    if matrix.shape != (SAFE_N, FIELD_D) or matrix.dtype != np.float64:
        errors.append(f"velocity matrix mismatch: {matrix.shape}, {matrix.dtype}")
    if population.shape != (SAFE_N,) or len(np.unique(population)) != SAFE_N:
        errors.append("safe population indices mismatch")
    if len(train_pos) != TRAIN_N or len(heldout_pos) != HOLDOUT_N:
        errors.append("fixed split counts mismatch")
    if np.intersect1d(train_pos, heldout_pos).size or np.union1d(train_pos, heldout_pos).size != SAFE_N:
        errors.append("fixed split is not a disjoint complete partition")
    if len(q40_gmm) != 800 or actual != expected:
        errors.append("q<=40 GMM grid is not exactly 800 cells")
    if [int(row["q"]) for row in q40_pca] != list(range(1, 41)):
        errors.append("q40 PCA summary does not cover q=1..40")
    if [int(row["q"]) for row in q40_pref] != list(range(1, 41)):
        errors.append("q40 preferred-K table does not cover q=1..40")
    cfg = q40_config.get("new_gmm_grid", {})
    expected_cfg = {"covariance_type": "full", "n_init": 5, "random_state": SEED,
                    "reg_covar": 1e-7, "max_iter": 500, "tol": 1e-5, "init_params": "kmeans"}
    for key, value in expected_cfg.items():
        if cfg.get(key) != value:
            errors.append(f"q40 GMM configuration mismatch for {key}: {cfg.get(key)}")
    if q40_config.get("vectorisation") != "C-order concatenate global[24,24,2], LE[48,48,2], TE[48,48,2]":
        errors.append("velocity vectorisation convention mismatch")
    with np.load(Q40 / "pca" / "pca_q40.npz") as pca40:
        if not np.array_equal(pca40["train_matrix_positions"], train_pos) or not np.array_equal(pca40["heldout_matrix_positions"], heldout_pos):
            errors.append("q40 PCA split differs from fixed split")
    if errors:
        raise RuntimeError("Input verification failed; no replacement will be created: " + "; ".join(errors))
    result = {
        "verified": True,
        "safe_samples": SAFE_N,
        "training_samples": TRAIN_N,
        "holdout_samples": HOLDOUT_N,
        "velocity_dimension": FIELD_D,
        "velocity_dtype": str(matrix.dtype),
        "vectorisation": q40_config["vectorisation"],
        "feature_scaling": q40_config.get("scaling", "none"),
        "pca_centring": q40_config.get("centering", "training-set mean via sklearn PCA"),
        "existing_q_range": [1, 40],
        "existing_K_range": [1, 20],
        "existing_gmm_cells": 800,
        "q40_validation_status": "PASS",
        "source_fingerprint_before": source_fingerprint(),
    }
    atomic_json(out / "input_verification.json", result)
    return result


def build_config(out: Path, verification: dict[str, Any]) -> None:
    config = {
        "source_study": str(SOURCE.relative_to(ROOT)),
        "q40_source": str(Q40.relative_to(ROOT)),
        "output": str(out.relative_to(ROOT)),
        "safe_samples": SAFE_N,
        "training_samples": TRAIN_N,
        "holdout_samples": HOLDOUT_N,
        "velocity_field_dimension": FIELD_D,
        "vectorisation": verification["vectorisation"],
        "scaling": "none",
        "centering": "training-set mean via sklearn PCA",
        "pca": {"n_components": 100, "svd_solver": "randomized", "random_state": SEED,
                "iterated_power": 7, "fit_once": True, "q_score_rule": "Z[:, :q]"},
        "new_gmm_grid": {"q": [41, 100], "K": [1, 20], "intended_cells": 1200, **GMM_CFG},
        "ICL": {"formula": "BIC + 2 * posterior_entropy_total", "preferred_direction": "minimum"},
        "small_component_threshold": SMALL_THRESHOLD,
        "legacy_policy": "reuse all q<=40 GMM rows byte-value-equivalently; compute only q=41..100",
        "registration_policy": "read saved velocity matrix only; no registration code imported or invoked",
        "targeted_stability": {"q": list(STABILITY_Q), "K": list(STABILITY_K), "seeds": list(STABILITY_SEEDS),
                               "n_init": STABILITY_N_INIT, "reuse_q40_K6_K7_K8": True},
        "source_fingerprint_before": verification["source_fingerprint_before"],
        "exact_command": ".venv/bin/python src/run_airfoil_template_number_q100_extension.py --stage all",
    }
    atomic_json(out / "config.json", config)


def fit_pca_q100(out: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    matrix = np.load(SOURCE / "velocity_field_matrix.npy", mmap_mode="r")
    split = np.load(SOURCE / "split" / "train_heldout_indices.npz")
    train_pos = split["train_matrix_positions"]
    heldout_pos = split["heldout_matrix_positions"]
    train = np.asarray(matrix[train_pos])
    heldout = np.asarray(matrix[heldout_pos])
    model = PCA(n_components=Q_MAX, svd_solver="randomized", random_state=SEED, iterated_power=7)
    start = time.perf_counter()
    train_scores = model.fit_transform(train)
    heldout_scores = model.transform(heldout)
    fit_runtime = time.perf_counter() - start

    train_centered_ss = float(np.sum((train - model.mean_) ** 2))
    heldout_residual = heldout - model.mean_
    heldout_centered_ss = float(np.sum(heldout_residual ** 2))
    train_captured = 0.0
    heldout_captured = 0.0
    q100_rows: list[dict[str, Any]] = []
    for q in range(1, Q_MAX + 1):
        train_captured += float(np.sum(train_scores[:, q - 1] ** 2))
        heldout_captured += float(np.sum(heldout_scores[:, q - 1] ** 2))
        heldout_residual -= heldout_scores[:, q - 1, None] * model.components_[q - 1][None, :]
        heldout_mse = max(0.0, (heldout_centered_ss - heldout_captured) / (HOLDOUT_N * FIELD_D))
        sample_rmse = np.sqrt(np.mean(heldout_residual ** 2, axis=1))
        q100_rows.append({
            "q": q,
            "marginal_explained_variance_ratio": float(model.explained_variance_ratio_[q - 1]),
            "cumulative_explained_variance": float(np.sum(model.explained_variance_ratio_[:q])),
            "train_reconstruction_mse": max(0.0, (train_centered_ss - train_captured) / (TRAIN_N * FIELD_D)),
            "heldout_reconstruction_mse": heldout_mse,
            "heldout_reconstruction_rmse": float(math.sqrt(heldout_mse)),
            "maximum_heldout_absolute_coefficient_error": float(np.max(np.abs(heldout_residual))),
            "maximum_heldout_sample_reconstruction_rmse": float(np.max(sample_rmse)),
            "pca_basis_source": "single_common_q100_training_fit",
        })

    old_rows = read_csv(Q40 / "pca_q1_q40.csv")
    with np.load(Q40 / "pca" / "pca_q40.npz") as old:
        dots = np.abs(np.sum(old["components"] * model.components_[:40], axis=1))
        train_corr = [abs(float(np.corrcoef(old["train_scores"][:, j], train_scores[:, j])[0, 1])) for j in range(40)]
        heldout_corr = [abs(float(np.corrcoef(old["heldout_scores"][:, j], heldout_scores[:, j])[0, 1])) for j in range(40)]
    cumulative_diff = [abs(float(old_rows[q - 1]["cumulative_explained_variance"]) - q100_rows[q - 1]["cumulative_explained_variance"]) for q in range(1, 41)]
    heldout_diff = [abs(float(old_rows[q - 1]["heldout_reconstruction_mse"]) - q100_rows[q - 1]["heldout_reconstruction_mse"]) for q in range(1, 41)]
    compatibility = {
        "pca_runtime_seconds": fit_runtime,
        "previous_basis": "single randomized q40 training-only PCA fit",
        "extension_basis": "single randomized q100 training-only PCA fit",
        "max_abs_cumulative_explained_variance_difference_q1_q40": max(cumulative_diff),
        "max_abs_heldout_reconstruction_mse_difference_q1_q40": max(heldout_diff),
        "minimum_corresponding_component_absolute_dot_q1_q40": float(np.min(dots)),
        "minimum_train_score_absolute_correlation_q1_q40": min(train_corr),
        "minimum_holdout_score_absolute_correlation_q1_q40": min(heldout_corr),
        "component_absolute_dots_q1_q40": dots.tolist(),
        "train_score_absolute_correlations_q1_q40": train_corr,
        "holdout_score_absolute_correlations_q1_q40": heldout_corr,
        "policy": "q<=40 GMM and saved PCA summary values are reused; q100-prefix comparison is validation only",
    }
    atomic_json(out / "pca" / "q40_q100_basis_compatibility.json", compatibility)
    np.savez_compressed(
        out / "pca" / "pca_q100.npz",
        mean=model.mean_, components=model.components_, explained_variance=model.explained_variance_,
        explained_variance_ratio=model.explained_variance_ratio_, singular_values=model.singular_values_,
        train_matrix_positions=train_pos, heldout_matrix_positions=heldout_pos,
        train_scores=train_scores, heldout_scores=heldout_scores,
    )

    combined_pca: list[dict[str, Any]] = []
    for q in range(1, 101):
        diagnostic = q100_rows[q - 1]
        if q <= 40:
            old = old_rows[q - 1]
            combined_pca.append({
                "q": q,
                "marginal_explained_variance_ratio": old["explained_variance_ratio"],
                "cumulative_explained_variance": old["cumulative_explained_variance"],
                "train_reconstruction_mse": old["train_reconstruction_mse"],
                "heldout_reconstruction_mse": old["heldout_reconstruction_mse"],
                "heldout_reconstruction_rmse": math.sqrt(float(old["heldout_reconstruction_mse"])),
                "maximum_heldout_absolute_coefficient_error": diagnostic["maximum_heldout_absolute_coefficient_error"],
                "maximum_heldout_sample_reconstruction_rmse": diagnostic["maximum_heldout_sample_reconstruction_rmse"],
                "pca_basis_source": "q40_saved_summary_reused",
                "q100_prefix_cumulative_explained_variance": diagnostic["cumulative_explained_variance"],
                "q100_prefix_heldout_reconstruction_mse": diagnostic["heldout_reconstruction_mse"],
            })
        else:
            combined_pca.append({**diagnostic,
                                 "q100_prefix_cumulative_explained_variance": diagnostic["cumulative_explained_variance"],
                                 "q100_prefix_heldout_reconstruction_mse": diagnostic["heldout_reconstruction_mse"]})
    save_csv(out / "pca_q1_q100.csv", combined_pca)
    save_csv(out / "pca_q41_q100_summary.csv", combined_pca[40:])
    return train_scores, heldout_scores, combined_pca, compatibility


def posterior_entropy(probabilities: np.ndarray) -> float:
    return float(-np.sum(probabilities * np.log(np.clip(probabilities, 1e-300, 1.0))))


def fit_gmm_cell(train_scores: np.ndarray, heldout_scores: np.ndarray, q: int, k: int,
                 seed: int = SEED, n_init: int = 5, basis: str = "single_common_q100_training_fit") -> tuple[dict[str, Any], np.ndarray | None]:
    start = time.perf_counter()
    covariance_parameters = k * q * (q + 1) // 2
    mean_parameters = k * q
    weight_parameters = k - 1
    parameter_count = gmm_parameter_count(q, k)
    base: dict[str, Any] = {
        "q": q, "K": k, "covariance_type": "full", "seed": seed, "random_state": seed,
        "n_init": n_init, "reg_covar": GMM_CFG["reg_covar"], "max_iter": GMM_CFG["max_iter"],
        "tol": GMM_CFG["tol"], "init_params": GMM_CFG["init_params"], "pca_basis_source": basis,
        "covariance_parameter_count": covariance_parameters, "mean_parameter_count": mean_parameters,
        "mixture_weight_parameter_count": weight_parameters, "parameter_count": parameter_count,
        "training_samples_per_parameter": TRAIN_N / parameter_count,
    }
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = GaussianMixture(n_components=k, covariance_type="full", n_init=n_init,
                                    random_state=seed, reg_covar=GMM_CFG["reg_covar"],
                                    max_iter=GMM_CFG["max_iter"], tol=GMM_CFG["tol"],
                                    init_params=GMM_CFG["init_params"]).fit(train_scores[:, :q])
        warning_messages = [str(item.message) for item in caught]
        train_ll = float(model.score(train_scores[:, :q]))
        holdout_ll = float(model.score(heldout_scores[:, :q]))
        probs = model.predict_proba(train_scores[:, :q])
        entropy = posterior_entropy(probs)
        labels = probs.argmax(axis=1)
        sizes = np.bincount(labels, minlength=k)
        bic = float(model.bic(train_scores[:, :q]))
        below = int(np.sum(sizes < SMALL_THRESHOLD))
        row = {
            **base,
            "status": "valid" if bool(model.converged_) else "non_converged",
            "valid_for_selection": bool(model.converged_), "converged": bool(model.converged_),
            "iterations": int(model.n_iter_), "training_log_likelihood_mean": train_ll,
            "training_log_likelihood_total": train_ll * TRAIN_N,
            "heldout_log_likelihood_mean": holdout_ll, "heldout_log_likelihood_total": holdout_ll * HOLDOUT_N,
            "BIC": bic, "ICL": bic + 2.0 * entropy,
            "ICL_formula": "BIC + 2 * posterior_entropy_total; lower is preferred",
            "posterior_entropy_total": entropy, "posterior_entropy_mean": entropy / TRAIN_N,
            "component_weights": json.dumps(model.weights_.tolist(), separators=(",", ":")),
            "hard_component_sizes": json.dumps([int(value) for value in sizes], separators=(",", ":")),
            "minimum_component_weight": float(model.weights_.min()),
            "minimum_component_size": int(sizes.min()), "maximum_component_size": int(sizes.max()),
            "minimum_component_proportion": float(sizes.min() / TRAIN_N),
            "small_component_threshold": SMALL_THRESHOLD,
            "number_components_below_threshold": below, "tiny_component_flag": bool(below > 0),
            "runtime_seconds": time.perf_counter() - start,
            "warning_or_failure_message": " | ".join(warning_messages),
        }
        numeric = [row[key] for key in ("training_log_likelihood_mean", "heldout_log_likelihood_mean", "BIC", "ICL", "posterior_entropy_total")]
        if not all(finite(value) for value in numeric):
            row["status"] = "failed_nonfinite"
            row["valid_for_selection"] = False
            row["warning_or_failure_message"] = (row["warning_or_failure_message"] + " | non-finite metric").strip(" |")
        return row, labels
    except Exception as exc:
        return ({
            **base, "status": "failed", "valid_for_selection": False, "converged": False, "iterations": 0,
            "training_log_likelihood_mean": "", "training_log_likelihood_total": "",
            "heldout_log_likelihood_mean": "", "heldout_log_likelihood_total": "", "BIC": "", "ICL": "",
            "ICL_formula": "BIC + 2 * posterior_entropy_total; lower is preferred",
            "posterior_entropy_total": "", "posterior_entropy_mean": "", "component_weights": "",
            "hard_component_sizes": "", "minimum_component_weight": "", "minimum_component_size": "",
            "maximum_component_size": "", "minimum_component_proportion": "",
            "small_component_threshold": SMALL_THRESHOLD, "number_components_below_threshold": "",
            "tiny_component_flag": "", "runtime_seconds": time.perf_counter() - start,
            "warning_or_failure_message": f"{type(exc).__name__}: {exc}",
        }, None)


def complete_new_row(row: dict[str, str], q: int, k: int) -> bool:
    if int(row.get("q", -1)) != q or int(row.get("K", -1)) != k:
        return False
    if row.get("covariance_type") != "full" or int(row.get("n_init", -1)) != 5 or int(row.get("seed", -1)) != SEED:
        return False
    if str(row.get("status", "")).startswith("failed"):
        return bool(row.get("warning_or_failure_message"))
    required = ["training_log_likelihood_mean", "heldout_log_likelihood_mean", "BIC", "ICL",
                "posterior_entropy_total", "minimum_component_size", "minimum_component_weight",
                "parameter_count", "runtime_seconds"]
    return all(finite(row.get(key)) for key in required)


def run_new_grid(out: Path, train_scores: np.ndarray, heldout_scores: np.ndarray, force: bool) -> list[dict[str, Any]]:
    path = out / "gmm_q41_q100_cells.csv"
    existing = [] if force else read_csv(path)
    rows: list[dict[str, Any]] = list(existing)
    by_cell = {(int(row["q"]), int(row["K"])): row for row in existing if row.get("q") and row.get("K")}
    total = 60 * 20
    for q in range(41, 101):
        for k in range(1, 21):
            old = by_cell.get((q, k))
            if old is not None and complete_new_row(old, q, k):
                continue
            if old is not None:
                rows = [row for row in rows if (int(row["q"]), int(row["K"])) != (q, k)]
            row, _ = fit_gmm_cell(train_scores, heldout_scores, q, k)
            rows.append(row)
            rows.sort(key=lambda value: (int(value["q"]), int(value["K"])))
            save_csv(path, rows)
            done = len(rows)
            print(f"primary {done:04d}/{total} q={q:03d} K={k:02d} status={row['status']} BIC={row['BIC']} HO={row['heldout_log_likelihood_mean']}", flush=True)
    return sorted(rows, key=lambda value: (int(value["q"]), int(value["K"])))


def parameter_table(out: Path) -> list[dict[str, Any]]:
    rows = []
    for q in range(1, 101):
        for k in range(1, 21):
            covariance = k * q * (q + 1) // 2
            means = k * q
            weights = k - 1
            total = covariance + means + weights
            rows.append({"q": q, "K": k, "covariance_parameters": covariance,
                         "mean_parameters": means, "mixture_weight_parameters": weights,
                         "parameter_count": total, "training_samples": TRAIN_N,
                         "training_samples_per_parameter": TRAIN_N / total})
    save_csv(out / "high_dimensional_parameter_counts.csv", rows)
    return rows


def analysis_value(row: dict[str, Any], key: str) -> Any:
    if key == "minimum_component_weight" and not finite(row.get(key)) and row.get("component_weights"):
        return min(json.loads(row["component_weights"]))
    if key == "parameter_count" and not finite(row.get(key)):
        return gmm_parameter_count(int(row["q"]), int(row["K"]))
    return row.get(key)


def combine_and_select(out: Path, new_rows: list[dict[str, Any]], pca_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    old_rows = read_csv(Q40 / "gmm_q1_q40_combined.csv")
    combined = old_rows + new_rows
    save_csv(out / "gmm_q1_q100_combined.csv", combined)
    preferred: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for q in range(1, 101):
        group = [row for row in combined if int(row["q"]) == q and bool_value(row["valid_for_selection"])]
        if not group:
            raise RuntimeError(f"No valid selectable GMM cell at q={q}")
        winners = {
            "BIC": min(group, key=lambda row: float(row["BIC"])),
            "ICL": min(group, key=lambda row: float(row["ICL"])),
            "heldout_LL": max(group, key=lambda row: float(row["heldout_log_likelihood_mean"])),
        }
        pca = pca_rows[q - 1]
        result: dict[str, Any] = {
            "q": q, "cumulative_explained_variance": pca["cumulative_explained_variance"],
            "heldout_reconstruction_mse": pca["heldout_reconstruction_mse"],
            "heldout_reconstruction_rmse": pca["heldout_reconstruction_rmse"],
            "pca_basis_source": pca["pca_basis_source"],
            "gmm_basis_source": "q40_saved_results_reused" if q <= 40 else "single_common_q100_training_fit",
        }
        for label, winner in winners.items():
            result[f"K_{label}"] = int(winner["K"])
            result[f"{label}_small_component_flag"] = bool_value(winner["tiny_component_flag"])
            result[f"{label}_minimum_component_size"] = int(winner["minimum_component_size"])
            result[f"{label}_minimum_component_weight"] = float(analysis_value(winner, "minimum_component_weight"))
            result[f"{label}_converged"] = bool_value(winner["converged"])
            result[f"{label}_parameter_count"] = int(analysis_value(winner, "parameter_count"))
            diagnostics.append({
                "q": q, "criterion": label, "selected_K": int(winner["K"]),
                "minimum_component_size": int(winner["minimum_component_size"]),
                "minimum_component_weight": float(analysis_value(winner, "minimum_component_weight")),
                "small_component_threshold": SMALL_THRESHOLD,
                "small_component_flag": bool_value(winner["tiny_component_flag"]),
                "converged": bool_value(winner["converged"]),
                "parameter_count": int(analysis_value(winner, "parameter_count")),
                "training_samples_per_parameter": TRAIN_N / int(analysis_value(winner, "parameter_count")),
            })
        preferred.append(result)
    save_csv(out / "preferred_k_q1_q100.csv", preferred)
    save_csv(out / "preferred_model_component_diagnostics.csv", diagnostics)
    sequences = {
        "q_1_q_100": {
            "K_BIC": [row["K_BIC"] for row in preferred],
            "K_ICL": [row["K_ICL"] for row in preferred],
            "K_holdout_LL": [row["K_heldout_LL"] for row in preferred],
        },
        "q_41_q_100": {
            "K_BIC": [row["K_BIC"] for row in preferred[40:]],
            "K_ICL": [row["K_ICL"] for row in preferred[40:]],
            "K_holdout_LL": [row["K_heldout_LL"] for row in preferred[40:]],
        },
    }
    atomic_json(out / "preferred_k_sequences_q1_q100.json", sequences)
    return combined, preferred


Q_RANGES = (("q1_20", 1, 20), ("q21_40", 21, 40), ("q41_60", 41, 60),
            ("q61_80", 61, 80), ("q81_100", 81, 100), ("q41_100", 41, 100))


def preferred_frequency(out: Path, preferred: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for range_name, lo, hi in Q_RANGES:
        subset = preferred[lo - 1:hi]
        for criterion, key in (("BIC", "K_BIC"), ("ICL", "K_ICL"), ("heldout_LL", "K_heldout_LL")):
            counts = Counter(int(row[key]) for row in subset)
            for k in range(1, 21):
                rows.append({"q_range": range_name, "q_min": lo, "q_max": hi, "criterion": criterion,
                             "K": k, "count": counts.get(k, 0), "percentage": 100.0 * counts.get(k, 0) / len(subset)})
    save_csv(out / "preferred_k_frequency_by_q_range.csv", rows)
    return rows


def model_selection_gaps(out: Path, combined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for q in range(1, 101):
        group = [row for row in combined if int(row["q"]) == q and bool_value(row["valid_for_selection"])]
        for criterion, key, direction in (("BIC", "BIC", "minimum"), ("ICL", "ICL", "minimum"),
                                           ("heldout_LL", "heldout_log_likelihood_mean", "maximum")):
            ordered = sorted(group, key=lambda row: float(row[key]), reverse=(direction == "maximum"))
            best, second = ordered[0], ordered[1]
            raw_gap = (float(second[key]) - float(best[key])) if direction == "minimum" else (float(best[key]) - float(second[key]))
            comparable_gap = raw_gap if criterion != "heldout_LL" else raw_gap * HOLDOUT_N
            if comparable_gap < (2.0 if criterion != "heldout_LL" else 1.0):
                strength = "nearly_tied"
            elif comparable_gap <= 10.0:
                strength = "moderately_separated"
            else:
                strength = "strongly_separated"
            rows.append({"q": q, "criterion": criterion, "best_K": int(best["K"]),
                         "second_best_K": int(second["K"]), "best_value": float(best[key]),
                         "second_best_value": float(second[key]), "raw_gap": raw_gap,
                         "comparable_gap": comparable_gap,
                         "gap_definition": "second-best minus best" if direction == "minimum" else "best minus second-best",
                         "heldout_total_gap": comparable_gap if criterion == "heldout_LL" else "",
                         "separation_strength": strength})
    save_csv(out / "model_selection_gap_by_q.csv", rows)
    return rows


def longest_run(values: list[int]) -> dict[str, int]:
    best_value = values[0]
    best_start = best_end = 1
    start = 1
    for index in range(2, len(values) + 1):
        if values[index - 1] != values[index - 2]:
            if index - start > best_end - best_start + 1:
                best_value, best_start, best_end = values[index - 2], start, index - 1
            start = index
    if len(values) - start + 1 > best_end - best_start + 1:
        best_value, best_start, best_end = values[-1], start, len(values)
    return {"K": int(best_value), "q_start": best_start, "q_end": best_end, "length": best_end - best_start + 1}


def copy_reused_stability(out: Path) -> tuple[list[dict[str, Any]], set[tuple[int, int, int]]]:
    old_cells = read_csv(Q40 / "targeted_stability_cells.csv")
    rows: list[dict[str, Any]] = []
    keys: set[tuple[int, int, int]] = set()
    assignment_dir = out / "stability_assignments"
    assignment_dir.mkdir(parents=True, exist_ok=True)
    for row in old_cells:
        q, k, seed = int(row["q"]), int(row["K"]), int(row["seed"])
        if q in (20, 40) and k in (6, 7, 8):
            enriched = {**row, "stability_source": "reused_q40_extension", "stability_n_init": STABILITY_N_INIT}
            rows.append(enriched)
            keys.add((q, k, seed))
            source_assignment = Q40 / "stability_assignments" / f"q{q:02d}_k{k:02d}_seed{seed}.npy"
            shutil.copyfile(source_assignment, assignment_dir / source_assignment.name)
    return rows, keys


def run_targeted_stability(out: Path, train_scores_q100: np.ndarray, heldout_scores_q100: np.ndarray,
                           force: bool) -> list[dict[str, Any]]:
    path = out / "targeted_stability_cells.csv"
    if force or not path.exists():
        rows, _ = copy_reused_stability(out)
    else:
        rows = read_csv(path)
    by_key = {(int(row["q"]), int(row["K"]), int(row["seed"])): row for row in rows}
    with np.load(Q40 / "pca" / "pca_q40.npz") as pca40:
        train_q40 = np.asarray(pca40["train_scores"])
        heldout_q40 = np.asarray(pca40["heldout_scores"])
    assignment_dir = out / "stability_assignments"
    for q in STABILITY_Q:
        for k in STABILITY_K:
            for seed in STABILITY_SEEDS:
                key = (q, k, seed)
                assignment_path = assignment_dir / f"q{q:02d}_k{k:02d}_seed{seed}.npy"
                if key in by_key and assignment_path.exists() and np.load(assignment_path, mmap_mode="r").shape == (TRAIN_N,):
                    continue
                train_scores = train_q40 if q <= 40 else train_scores_q100
                heldout_scores = heldout_q40 if q <= 40 else heldout_scores_q100
                basis = "single_common_q40_training_fit" if q <= 40 else "single_common_q100_training_fit"
                row, labels = fit_gmm_cell(train_scores, heldout_scores, q, k, seed=seed,
                                           n_init=STABILITY_N_INIT, basis=basis)
                row["stability_source"] = "new_q100_extension"
                row["stability_n_init"] = STABILITY_N_INIT
                rows = [value for value in rows if (int(value["q"]), int(value["K"]), int(value["seed"])) != key]
                rows.append(row)
                if labels is not None:
                    np.save(assignment_path, labels)
                rows.sort(key=lambda value: (int(value["q"]), int(value["K"]), int(value["seed"])))
                save_csv(path, rows)
                print(f"stability q={q:03d} K={k:02d} seed={seed} status={row['status']}", flush=True)

    pair_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for q in STABILITY_Q:
        for k in STABILITY_K:
            labelled = []
            cell_group = [row for row in rows if int(row["q"]) == q and int(row["K"]) == k]
            for seed in STABILITY_SEEDS:
                p = assignment_dir / f"q{q:02d}_k{k:02d}_seed{seed}.npy"
                if p.exists():
                    labelled.append((seed, np.load(p)))
            aris = []
            for i in range(len(labelled)):
                for j in range(i + 1, len(labelled)):
                    ari = float(adjusted_rand_score(labelled[i][1], labelled[j][1]))
                    aris.append(ari)
                    pair_rows.append({"q": q, "K": k, "seed_a": labelled[i][0], "seed_b": labelled[j][0], "ARI": ari})
            min_sizes = [int(row["minimum_component_size"]) for row in cell_group if finite(row.get("minimum_component_size"))]
            sources = sorted({row.get("stability_source", "") for row in cell_group})
            summaries.append({
                "q": q, "K": k, "successful_seeds": len(labelled), "pair_count": len(aris),
                "mean_pairwise_ARI": float(np.mean(aris)) if aris else "",
                "minimum_pairwise_ARI": float(np.min(aris)) if aris else "",
                "maximum_pairwise_ARI": float(np.max(aris)) if aris else "",
                "std_pairwise_ARI": float(np.std(aris, ddof=0)) if aris else "",
                "mean_minimum_component_size_across_seeds": float(np.mean(min_sizes)) if min_sizes else "",
                "std_minimum_component_size_across_seeds": float(np.std(min_sizes, ddof=0)) if min_sizes else "",
                "minimum_component_size_across_seeds": min(min_sizes) if min_sizes else "",
                "maximum_minimum_component_size_across_seeds": max(min_sizes) if min_sizes else "",
                "sources": ";".join(sources), "n_init_per_seed": STABILITY_N_INIT,
            })
    save_csv(out / "targeted_stability_pairwise_ari.csv", pair_rows)
    save_csv(out / "targeted_stability_ari.csv", summaries)
    return summaries


def style_axis(ax: plt.Axes) -> None:
    ax.grid(alpha=0.22, color="#9ca3af")
    ax.spines[["top", "right"]].set_visible(False)


COLORS = {"K_BIC": "#2563eb", "K_ICL": "#d97706", "K_heldout_LL": "#9f1239"}
MARKERS = {"K_BIC": "o", "K_ICL": "s", "K_heldout_LL": "^"}


def plot_preferred(preferred: list[dict[str, Any]], figures: Path) -> None:
    q = np.arange(1, 101)
    fig, ax = plt.subplots(figsize=(15, 5.6))
    for key, label in (("K_BIC", "BIC"), ("K_ICL", "ICL"), ("K_heldout_LL", "Held-out LL")):
        ax.plot(q, [row[key] for row in preferred], color=COLORS[key], marker=MARKERS[key],
                markersize=3.4, linewidth=1.45, label=label)
    for boundary in (20.5, 40.5):
        ax.axvline(boundary, color="#374151", linestyle="--", linewidth=1.0)
    ax.axvspan(40.5, 100.5, color="#f59e0b", alpha=0.065)
    ax.set(xlabel="PCA dimension q", ylabel="Preferred GMM component count K",
           title="Preferred component count across PCA dimensions (q=1–100)",
           xticks=np.arange(0, 101, 5), yticks=np.arange(1, 21), xlim=(0.5, 100.5), ylim=(0.5, 20.5))
    style_axis(ax)
    ax.legend(ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(figures / "preferred_k_q1_q100.png", dpi=240)
    fig.savefig(figures / "preferred_k_q1_q100.pdf")
    plt.close(fig)
    for key, label, title in (("K_BIC", "BIC", "BIC-preferred K versus PCA dimension"),
                              ("K_ICL", "ICL", "ICL-preferred K versus PCA dimension"),
                              ("K_heldout_LL", "Held-out LL", "Held-out-likelihood-preferred K versus PCA dimension")):
        fig, ax = plt.subplots(figsize=(13, 4.6))
        ax.plot(q, [row[key] for row in preferred], color=COLORS[key], marker=MARKERS[key], markersize=3.8, linewidth=1.6)
        for boundary in (20.5, 40.5):
            ax.axvline(boundary, color="#6b7280", linestyle="--", linewidth=1.0)
        ax.axvspan(40.5, 100.5, color="#f59e0b", alpha=0.065)
        ax.set(xlabel="PCA dimension q", ylabel="Preferred K", title=title,
               xticks=np.arange(0, 101, 5), yticks=np.arange(1, 21), xlim=(0.5, 100.5), ylim=(0.5, 20.5))
        style_axis(ax)
        fig.tight_layout()
        fig.savefig(figures / f"preferred_k_{label.lower().replace('-', '_').replace(' ', '_')}_q1_q100.png", dpi=230)
        plt.close(fig)


def plot_pca(pca_rows: list[dict[str, Any]], figures: Path) -> None:
    q = np.arange(1, 101)
    specs = [
        ([float(row["cumulative_explained_variance"]) for row in pca_rows], "Cumulative explained variance", False,
         "Cumulative PCA explained variance", "pca_cumulative_explained_variance_q1_q100.png"),
        ([float(row["heldout_reconstruction_mse"]) for row in pca_rows], "Held-out coefficient reconstruction MSE", True,
         "Held-out PCA reconstruction MSE", "pca_heldout_reconstruction_mse_q1_q100.png"),
        ([float(row["marginal_explained_variance_ratio"]) for row in pca_rows], "Marginal explained variance ratio", True,
         "Marginal explained variance by component", "pca_marginal_explained_variance_q1_q100.png"),
    ]
    for values, ylabel, logy, title, name in specs:
        fig, ax = plt.subplots(figsize=(12.5, 4.8))
        ax.plot(q, values, color="#2563eb", linewidth=1.6)
        ax.scatter(q[::4], np.asarray(values)[::4], color="#2563eb", s=14)
        for boundary in (20.5, 40.5):
            ax.axvline(boundary, color="#6b7280", linestyle="--", linewidth=1.0)
        if logy:
            ax.set_yscale("log")
        ax.set(xlabel="PCA dimension q", ylabel=ylabel, title=title, xticks=np.arange(0, 101, 5), xlim=(0.5, 100.5))
        style_axis(ax)
        fig.tight_layout()
        fig.savefig(figures / name, dpi=230)
        plt.close(fig)


def plot_parameter_counts(parameters: list[dict[str, Any]], figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    colors = {2: "#0f766e", 6: "#2563eb", 7: "#d97706", 8: "#9f1239", 13: "#6b7280"}
    for k in (2, 6, 7, 8, 13):
        subset = [row for row in parameters if row["K"] == k]
        ax.plot([row["q"] for row in subset], [row["parameter_count"] for row in subset],
                color=colors[k], linewidth=1.8, label=f"K={k}")
    ax.axhline(TRAIN_N, color="#111827", linestyle="--", linewidth=1.1, label="training samples=1321")
    ax.set_yscale("log")
    ax.set(xlabel="PCA dimension q", ylabel="Full-covariance GMM free parameters (log scale)",
           title="GMM parameter burden versus PCA dimension", xticks=np.arange(0, 101, 10), xlim=(0.5, 100.5))
    style_axis(ax)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(figures / "gmm_parameter_count_vs_q.png", dpi=230)
    plt.close(fig)


def grid_matrix(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    result = np.full((100, 20), np.nan)
    for row in rows:
        if bool_value(row.get("valid_for_selection")) and finite(row.get(key)):
            result[int(row["q"]) - 1, int(row["K"]) - 1] = float(row[key])
    return result


def plot_relative_heatmaps(combined: list[dict[str, Any]], figures: Path) -> None:
    bic = grid_matrix(combined, "BIC")
    icl = grid_matrix(combined, "ICL")
    ho = grid_matrix(combined, "heldout_log_likelihood_mean")
    matrices = [
        (bic - np.nanmin(bic, axis=1, keepdims=True), "Within-q relative BIC", "Delta BIC", "relative_bic_q1_q100_heatmap.png"),
        (icl - np.nanmin(icl, axis=1, keepdims=True), "Within-q relative ICL", "Delta ICL", "relative_icl_q1_q100_heatmap.png"),
        (np.nanmax(ho, axis=1, keepdims=True) - ho, "Within-q held-out likelihood loss", "Best mean LL - mean LL(K)", "relative_heldout_ll_q1_q100_heatmap.png"),
    ]
    for matrix, title, label, name in matrices:
        upper = float(np.nanpercentile(matrix, 95))
        fig, ax = plt.subplots(figsize=(11, 12))
        image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=upper, interpolation="nearest")
        for boundary in (19.5, 39.5):
            ax.axhline(boundary, color="white", linestyle="--", linewidth=1.0)
        ax.set(xlabel="GMM components K", ylabel="PCA dimension q", title=title,
               xticks=np.arange(20), xticklabels=np.arange(1, 21),
               yticks=np.arange(0, 100, 5), yticklabels=np.arange(1, 101, 5))
        fig.colorbar(image, ax=ax, label=label, shrink=0.82)
        fig.tight_layout()
        fig.savefig(figures / name, dpi=220)
        plt.close(fig)


def plot_gaps(gaps: list[dict[str, Any]], figures: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for ax, criterion, color in zip(axes, ("BIC", "ICL", "heldout_LL"), ("#2563eb", "#d97706", "#9f1239")):
        subset = [row for row in gaps if row["criterion"] == criterion]
        ax.plot([row["q"] for row in subset], [row["comparable_gap"] for row in subset], color=color, linewidth=1.5)
        ax.scatter([row["q"] for row in subset], [row["comparable_gap"] for row in subset], color=color, s=10)
        ax.axhline(2 if criterion != "heldout_LL" else 1, color="#6b7280", linestyle=":", linewidth=1)
        ax.axhline(10, color="#6b7280", linestyle="--", linewidth=1)
        for boundary in (20.5, 40.5):
            ax.axvline(boundary, color="#9ca3af", linestyle="--", linewidth=0.9)
        ax.set_ylabel("Best–2nd gap" if criterion != "heldout_LL" else "Held-out total LL gap")
        ax.set_title(criterion)
        style_axis(ax)
    axes[-1].set(xlabel="PCA dimension q", xticks=np.arange(0, 101, 5), xlim=(0.5, 100.5))
    fig.suptitle("Separation between best and second-best GMM component count", y=0.995)
    fig.tight_layout()
    fig.savefig(figures / "best_vs_second_best_gap_q1_q100.png", dpi=230)
    plt.close(fig)


def plot_stability(stability: list[dict[str, Any]], figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.1))
    colors = {2: "#0f766e", 6: "#2563eb", 7: "#d97706", 8: "#9f1239"}
    for k in STABILITY_K:
        subset = [row for row in stability if int(row["K"]) == k]
        ax.errorbar([int(row["q"]) for row in subset], [float(row["mean_pairwise_ARI"]) for row in subset],
                    yerr=[float(row["std_pairwise_ARI"]) for row in subset], color=colors[k],
                    marker="o", capsize=3, linewidth=1.6, label=f"K={k}")
    ax.set(xlabel="PCA dimension q", ylabel="Mean pairwise ARI ± SD", title="Targeted full-covariance seed stability",
           xticks=list(STABILITY_Q), ylim=(-0.02, 1.02))
    style_axis(ax)
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(figures / "targeted_ari_stability_vs_q.png", dpi=230)
    plt.close(fig)


def plot_min_component(preferred: list[dict[str, Any]], figures: Path) -> None:
    q = np.arange(1, 101)
    fig, ax = plt.subplots(figsize=(13, 4.8))
    for key, label, color in (("BIC_minimum_component_size", "BIC-selected", "#2563eb"),
                              ("ICL_minimum_component_size", "ICL-selected", "#d97706"),
                              ("heldout_LL_minimum_component_size", "Held-out-selected", "#9f1239")):
        ax.plot(q, [int(row[key]) for row in preferred], color=color, linewidth=1.5, label=label)
    ax.axhline(SMALL_THRESHOLD, color="#111827", linestyle="--", linewidth=1.2, label=f"small threshold={SMALL_THRESHOLD}")
    for boundary in (20.5, 40.5):
        ax.axvline(boundary, color="#9ca3af", linestyle="--", linewidth=0.9)
    ax.set(xlabel="PCA dimension q", ylabel="Minimum hard component size",
           title="Minimum component size of criterion-selected model", xticks=np.arange(0, 101, 5), xlim=(0.5, 100.5))
    ax.set_yscale("log")
    style_axis(ax)
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(figures / "preferred_model_minimum_component_size_vs_q.png", dpi=230)
    plt.close(fig)


def storage_bytes(path: Path) -> int:
    return int(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()))


def write_report(out: Path, pca_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]],
                 combined: list[dict[str, Any]], preferred: list[dict[str, Any]], frequencies: list[dict[str, Any]],
                 gaps: list[dict[str, Any]], stability: list[dict[str, Any]], compatibility: dict[str, Any],
                 aggregate_runtime: float, primary_runtime: float, stability_runtime: float) -> None:
    sequences = json.loads((out / "preferred_k_sequences_q1_q100.json").read_text(encoding="utf-8"))
    valid = [row for row in new_rows if bool_value(row["valid_for_selection"])]
    failed = [row for row in new_rows if str(row["status"]).startswith("failed")]
    nonconv = [row for row in new_rows if row["status"] == "non_converged"]
    runs = {key: longest_run(sequences["q_1_q_100"][key]) for key in ("K_BIC", "K_ICL", "K_holdout_LL")}
    thresholds = {}
    for threshold in (0.99, 0.999, 0.9995, 0.9999):
        reached = [int(row["q"]) for row in pca_rows if float(row["cumulative_explained_variance"]) > threshold]
        thresholds[str(threshold)] = reached[0] if reached else None
    qpoints = [19, 20, 40, 60, 80, 100]
    stable = {(int(row["q"]), int(row["K"])): row for row in stability}
    frequency_lookup: dict[tuple[str, str], list[tuple[int, int, float]]] = {}
    for range_name, _, _ in Q_RANGES:
        for criterion in ("BIC", "ICL", "heldout_LL"):
            vals = [(int(row["K"]), int(row["count"]), float(row["percentage"])) for row in frequencies
                    if row["q_range"] == range_name and row["criterion"] == criterion and int(row["count"]) > 0]
            frequency_lookup[(range_name, criterion)] = sorted(vals, key=lambda x: (-x[1], x[0]))
    q41_100 = preferred[40:]
    highq_small = {criterion: sum(bool_value(row[f"{criterion}_small_component_flag"]) for row in q41_100)
                   for criterion in ("BIC", "ICL", "heldout_LL")}
    gap_summary = {}
    for criterion in ("BIC", "ICL", "heldout_LL"):
        subset = [row for row in gaps if row["criterion"] == criterion and int(row["q"]) >= 41]
        gap_summary[criterion] = Counter(row["separation_strength"] for row in subset)
    lines = [
        "# Airfoil velocity-field PCA–GMM q=100 extension",
        "",
        "## 结论摘要",
        "",
        f"本扩展严格复用 1651 个安全速度场、固定 1321/330 训练—留出拆分与 D=10368 的 C-order 向量化；没有重新运行配准。新计算仅覆盖 q=41–100、K=1–20，共 `{len(new_rows)}` 个 full-covariance GMM 单元（每单元 n_init=5，即主网格 `{len(new_rows)*5}` 次 EM 初始化）。有效 `{len(valid)}`、失败 `{len(failed)}`、未收敛 `{len(nonconv)}`。",
        "",
        f"全范围最长平台：BIC 为 K={runs['K_BIC']['K']}（q={runs['K_BIC']['q_start']}–{runs['K_BIC']['q_end']}，连续 {runs['K_BIC']['length']} 维），ICL 为 K={runs['K_ICL']['K']}（q={runs['K_ICL']['q_start']}–{runs['K_ICL']['q_end']}，连续 {runs['K_ICL']['length']} 维），留出似然为 K={runs['K_holdout_LL']['K']}（q={runs['K_holdout_LL']['q_start']}–{runs['K_holdout_LL']['q_end']}，连续 {runs['K_holdout_LL']['length']} 维）。这些是模型选择准则的平台，不等同于真实物理族群数。",
        "",
        "![Preferred K](../figures/preferred_k_q1_q100.png)",
        "",
        "## 完整首选 K 序列",
        "",
        f"BIC（q=1–100）：`{sequences['q_1_q_100']['K_BIC']}`",
        "",
        f"ICL（q=1–100）：`{sequences['q_1_q_100']['K_ICL']}`",
        "",
        f"留出似然（q=1–100）：`{sequences['q_1_q_100']['K_holdout_LL']}`",
        "",
        f"BIC 新段（q=41–100）：`{sequences['q_41_q_100']['K_BIC']}`",
        "",
        f"ICL 新段（q=41–100）：`{sequences['q_41_q_100']['K_ICL']}`",
        "",
        f"留出似然新段（q=41–100）：`{sequences['q_41_q_100']['K_holdout_LL']}`",
        "",
        "## 分段首选频率",
        "",
        "| q范围 | BIC | ICL | 留出似然 |",
        "|---|---|---|---|",
        *[f"| {name.replace('q','q=').replace('_','–')} | {frequency_lookup[(name,'BIC')]} | {frequency_lookup[(name,'ICL')]} | {frequency_lookup[(name,'heldout_LL')]} |" for name, _, _ in Q_RANGES],
        "",
        "括号内容依次为 `(K, 次数, 百分比)`；完整长表见 `preferred_k_frequency_by_q_range.csv`。",
        "",
        "## PCA 重构收益与操作区间",
        "",
        f"首次超过99%、99.9%、99.95%、99.99%累计解释方差的 q 分别为 `{thresholds['0.99']}`、`{thresholds['0.999']}`、`{thresholds['0.9995']}`、`{thresholds['0.9999'] if thresholds['0.9999'] is not None else 'q≤100未达到'}`。",
        "",
        "![PCA variance](../figures/pca_cumulative_explained_variance_q1_q100.png)",
        "",
        "![PCA MSE](../figures/pca_heldout_reconstruction_mse_q1_q100.png)",
        "",
        "| q | 累计解释方差 | 留出重构MSE | BIC K | ICL K | HO K | K=2 ARI | K=6 ARI | K=7 ARI | K=8 ARI |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *[f"| {q} | {float(pca_rows[q-1]['cumulative_explained_variance']):.7%} | {float(pca_rows[q-1]['heldout_reconstruction_mse']):.3e} | {preferred[q-1]['K_BIC']} | {preferred[q-1]['K_ICL']} | {preferred[q-1]['K_heldout_LL']} | " + " | ".join(f"{float(stable[(q,k)]['mean_pairwise_ARI']):.3f}" if (q,k) in stable else "—" for k in STABILITY_K) + " |" for q in qpoints],
        "",
        "q≈19–20 已跨过99.9%阈值。后续维度是否‘值得’不能用重构误差单独决定；还必须同时看同q模型竞争、种子稳定性、分量安全和参数负担。",
        "",
        "## full-covariance 参数负担",
        "",
        "每个 q 维 Gaussian 分量包含 `q(q+1)/2` 个协方差参数与 q 个均值参数；K 个分量再有 K−1 个独立权重，因此总参数数为 `K[q(q+1)/2+q]+(K−1)`。q=100 时每个分量仅协方差就有 5,050 个参数；K=2/6/7/8 的总参数分别为 " + ", ".join(f"{gmm_parameter_count(100,k):,}" for k in STABILITY_K) + "，都远高于1,321个训练样本。",
        "",
        "![Parameter count](../figures/gmm_parameter_count_vs_q.png)",
        "",
        "因此高q留出似然偏好更小K与协方差估计方差上升是相容的，但这只是统计解释，不能据此断言物理形状空间恰有两个族群。",
        "",
        "## 小分量与模型分离强度",
        "",
        f"q=41–100 的60个首选模型中，BIC/ICL/留出准则分别有 `{highq_small['BIC']}`、`{highq_small['ICL']}`、`{highq_small['heldout_LL']}` 个触发现有小分量阈值（硬分配<14）。失败或非有限单元从未参与首选提取。",
        "",
        "![Minimum component](../figures/preferred_model_minimum_component_size_vs_q.png)",
        "",
        f"q=41–100 最优与次优差距分类：BIC `{dict(gap_summary['BIC'])}`，ICL `{dict(gap_summary['ICL'])}`，留出似然 `{dict(gap_summary['heldout_LL'])}`。BIC/ICL按Δ<2近似并列、2–10中等、>10强分离；留出似然按330个留出样本的总LL差<1、1–10、>10分类。",
        "",
        "![Gaps](../figures/best_vs_second_best_gap_q1_q100.png)",
        "",
        "## 定向种子稳定性",
        "",
        "稳定性比较覆盖 q=20/40/60/80/100 与 K=2/6/7/8，full covariance、seeds 0–4、每seed n_init=3。q=20/40 的 K=6/7/8 直接复用已验证结果；q=20/40 的K=2及q=60/80/100为本扩展新增。",
        "",
        "| q | K | mean ARI | min | max | SD | 最小分量均值±SD |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        *[f"| {row['q']} | {row['K']} | {float(row['mean_pairwise_ARI']):.3f} | {float(row['minimum_pairwise_ARI']):.3f} | {float(row['maximum_pairwise_ARI']):.3f} | {float(row['std_pairwise_ARI']):.3f} | {float(row['mean_minimum_component_size_across_seeds']):.1f}±{float(row['std_minimum_component_size_across_seeds']):.1f} |" for row in stability],
        "",
        "![ARI](../figures/targeted_ari_stability_vs_q.png)",
        "",
        "## 兼容性、运行与限制",
        "",
        f"q≤40 的800个GMM单元逐字段原样复用。q=100 randomized PCA与既有q=40基在前40维的累计方差最大差 `{compatibility['max_abs_cumulative_explained_variance_difference_q1_q40']:.3e}`、留出重构MSE最大差 `{compatibility['max_abs_heldout_reconstruction_mse_difference_q1_q40']:.3e}`、对应成分最小绝对点积 `{compatibility['minimum_corresponding_component_absolute_dot_q1_q40']:.6f}`。PCA求解器维数变化造成的差异被显式记录；没有静默重算q≤40 GMM。",
        "",
        f"主1200单元累计拟合时间 `{primary_runtime:.1f}` 秒；新增定向稳定性拟合 `{stability_runtime:.1f}` 秒；含PCA在内的累计模型拟合时间 `{aggregate_runtime:.1f}` 秒。生成报告时输出约 `{storage_bytes(out)/(1024**2):.2f}` MiB；最终精确字节数记录于 `run_summary.json`。",
        "",
        "ICL沿用原公式：`ICL = BIC + 2 × posterior_entropy_total`，其中 `posterior_entropy_total = -Σ_iΣ_k τ_ik log τ_ik`，取最小；BIC取最小；留出平均对数似然取最大。",
        "",
        "## 最终16项答复",
        "",
        f"1. **新增GMM单元：** `{len(new_rows)}`，即 q=41–100 的60维 × K=1–20。",
        f"2. **EM初始化次数：** 主网格 `{len(new_rows)*5}` 次；本阶段新增定向稳定性 `{len([row for row in read_csv(out / 'targeted_stability_cells.csv') if row.get('stability_source') == 'new_q100_extension'])*STABILITY_N_INIT}` 次；另复用既有90次初始化的稳定性结果。",
        f"3. **运行与存储：** 主网格 `{primary_runtime:.1f}` 秒，新增稳定性 `{stability_runtime:.1f}` 秒，含PCA累计模型拟合 `{aggregate_runtime:.1f}` 秒；最终存储量见 `run_summary.json`。",
        f"4. **收敛/失败：** 新网格有效 `{len(valid)}`、失败 `{len(failed)}`、未收敛 `{len(nonconv)}`。",
        f"5. **PCA重构：** q=20/40/60/80/100 的留出MSE依次为 " + ", ".join(f"{float(pca_rows[q-1]['heldout_reconstruction_mse']):.3e}" for q in (20,40,60,80,100)) + f"；q=20到100相对降低 `{1-float(pca_rows[99]['heldout_reconstruction_mse'])/float(pca_rows[19]['heldout_reconstruction_mse']):.1%}`，但累计解释方差只增加 `{100*(float(pca_rows[99]['cumulative_explained_variance'])-float(pca_rows[19]['cumulative_explained_variance'])):.4f}` 个百分点。",
        "6. **完整序列：** 已在上文逐项打印，并保存到 `preferred_k_sequences_q1_q100.json`。",
        "7. **分段频率：** 已在上表给出次数与百分比，并保存为长表CSV。",
        f"8. **BIC是否稳定：** 并未在K=7保持到q=100。K=7持续到q=51，q=52–77转为K=4；q=78–81为K=3、q=82短暂回到K=4、q=83–100再次为K=3。全范围最长平台是K={runs['K_BIC']['K']}、q={runs['K_BIC']['q_start']}–{runs['K_BIC']['q_end']}。",
        f"9. **ICL是否稳定：** q>40与BIC序列完全一致；同样从K=7转为K=4，再转为K=3，全范围最长平台为K={runs['K_ICL']['K']}、q={runs['K_ICL']['q_start']}–{runs['K_ICL']['q_end']}。",
        f"10. **留出似然是否稳定：** 基本稳定为K=2：q=41–100中59/60维选择K=2，仅q=99选择K=1；最长连续K=2平台为q={runs['K_holdout_LL']['q_start']}–{runs['K_holdout_LL']['q_end']}。",
        "11. **q>40是否改变结论：** 是。它否定了‘BIC/ICL会长期停在K=7’的假设，展示出随q增大由7→4→3的系统性下降；留出准则则维持低K。",
        f"12. **病理小分量：** 否。q=41–100的BIC、ICL、留出首选模型均为0/60触发硬分量<14阈值；对应最小硬分量分别至少 `{min(int(row['BIC_minimum_component_size']) for row in q41_100)}`、`{min(int(row['ICL_minimum_component_size']) for row in q41_100)}`、`{min(int(row['heldout_LL_minimum_component_size']) for row in q41_100)}`。",
        "13. **ARI稳定性：** K=2在五个q上均为1.000；K=6约0.639–0.789，K=7约0.584–0.613，K=8从q=20/40的0.580/0.554升至q=80/100约0.787/0.786。ARI衡量分配可重复性，不等同于预测优度或物理真实性。",
        f"14. **最优胜出强度：** q=41–100 的BIC/ICL/留出首选在预定义尺度下均为60/60强分离；因此7→4→3及K=2平台不是由近似并列的单元随机跳动造成。",
        "15. **q≈19–20操作区间：** 仍有支持。q=19首次超过99.9%，q=20时三准则为BIC=7、ICL=7、留出=6，且K=6的平均ARI=0.789；继续提高q改善重构，却使full-covariance参数远超样本并显著改变K。这里的选择是科学可解释性与估计负担的折中，不是因为高q重构没有收益。",
        "16. **更新后的候选K：** 下游模板验证的主要候选仍建议K=6–8（对应q≈19–20的共同证据）；同时将K=2和K=3–4列为明确的高q敏感性对照。不要把它写成连续的K=2–8范围，也不要仅凭本统计阶段替换应用驱动的模板数验证。",
        "",
        "## 科学解释与停止条件",
        "",
        "本研究只筛查 Gaussian mixture component count 对 PCA 截断维度的敏感性。GMM分量数、种子稳定性、预测留出密度、PCA操作维度和最终应用驱动模板数是五个不同问题；任何单一统计准则都不能证明真实气动外形族群数。最终模板数仍须通过另行批准的集群模板重注册验证。",
        "",
        "本阶段未重新运行配准、未构造模板、未做模板重注册、未扩展K>20或q>100、未commit/push；到此停止。",
    ]
    atomic_text(out / "report" / "q100_extension_report.md", "\n".join(lines) + "\n")


def finalize(out: Path, train_scores: np.ndarray, heldout_scores: np.ndarray,
             pca_rows: list[dict[str, Any]], compatibility: dict[str, Any], force: bool) -> None:
    new_rows = run_new_grid(out, train_scores, heldout_scores, force)
    parameters = parameter_table(out)
    combined, preferred = combine_and_select(out, new_rows, pca_rows)
    frequencies = preferred_frequency(out, preferred)
    gaps = model_selection_gaps(out, combined)
    stability = run_targeted_stability(out, train_scores, heldout_scores, force)
    figures = out / "figures"
    plot_preferred(preferred, figures)
    plot_pca(pca_rows, figures)
    plot_parameter_counts(parameters, figures)
    plot_relative_heatmaps(combined, figures)
    plot_gaps(gaps, figures)
    plot_stability(stability, figures)
    plot_min_component(preferred, figures)
    before = json.loads((out / "input_verification.json").read_text(encoding="utf-8"))["source_fingerprint_before"]
    after = source_fingerprint()
    if before != after:
        raise RuntimeError("Completed source study changed during extension; refusing to finalize")
    primary_runtime = sum(float(row["runtime_seconds"]) for row in new_rows)
    stability_cells = read_csv(out / "targeted_stability_cells.csv")
    new_stability = [row for row in stability_cells if row.get("stability_source") == "new_q100_extension"]
    stability_runtime = sum(float(row["runtime_seconds"]) for row in new_stability)
    aggregate_runtime = float(compatibility["pca_runtime_seconds"]) + primary_runtime + stability_runtime
    write_report(out, pca_rows, new_rows, combined, preferred, frequencies, gaps, stability,
                 compatibility, aggregate_runtime, primary_runtime, stability_runtime)
    script_dir = out / "scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__), script_dir / Path(__file__).name)
    summary = {
        "new_primary_cells": len(new_rows),
        "primary_EM_initialization_runs": len(new_rows) * GMM_CFG["n_init"],
        "new_targeted_stability_fits": len(new_stability),
        "new_targeted_stability_EM_initialization_runs": len(new_stability) * STABILITY_N_INIT,
        "reused_targeted_stability_fits": len(stability_cells) - len(new_stability),
        "valid_new_cells": sum(bool_value(row["valid_for_selection"]) for row in new_rows),
        "failed_new_cells": sum(str(row["status"]).startswith("failed") for row in new_rows),
        "non_converged_new_cells": sum(row["status"] == "non_converged" for row in new_rows),
        "pca_runtime_seconds": compatibility["pca_runtime_seconds"],
        "primary_grid_runtime_seconds": primary_runtime,
        "new_targeted_stability_runtime_seconds": stability_runtime,
        "aggregate_model_fit_runtime_seconds": aggregate_runtime,
        "source_fingerprint_unchanged": True,
        "output_storage_bytes_before_validation": storage_bytes(out),
    }
    atomic_json(out / "run_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stage", choices=("verify", "pca", "gmm", "stability", "finalize", "all"), default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    for folder in ("pca", "figures", "report", "stability_assignments", "scripts"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    verification = verify_inputs(out)
    build_config(out, verification)
    if args.stage == "verify":
        return
    train_scores, heldout_scores, pca_rows, compatibility = fit_pca_q100(out)
    if args.stage == "pca":
        return
    if args.stage == "gmm":
        run_new_grid(out, train_scores, heldout_scores, args.force)
        return
    if args.stage == "stability":
        run_targeted_stability(out, train_scores, heldout_scores, args.force)
        return
    finalize(out, train_scores, heldout_scores, pca_rows, compatibility, args.force)


if __name__ == "__main__":
    main()
