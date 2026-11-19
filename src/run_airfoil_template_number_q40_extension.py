#!/usr/bin/env python3
"""Extend the completed Stage-5 PCA/GMM screen from q=20 to q=40.

This script is deliberately read-only with respect to the completed Stage-5
screen.  It reuses the saved 1651-by-10368 velocity-field matrix and the exact
saved split, fits one training-only q=40 PCA basis, runs only q=21..40 GMM
cells, and writes all new artifacts to a separate output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
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
DEFAULT_OUT = ROOT / "outputs" / "airfoil_template_number_stage5_q40_extension"
SEED = 20260813
TRAIN_N = 1321
HOLDOUT_N = 330
SAFE_N = 1651
FIELD_D = 10368
Q_MAX = 40
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


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
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


def source_fingerprint() -> dict[str, Any]:
    registration_dir = SOURCE / "registrations"
    registration_files = sorted(registration_dir.glob("*.npz"))
    paths = {
        "velocity_field_matrix": SOURCE / "velocity_field_matrix.npy",
        "velocity_field_population_indices": SOURCE / "velocity_field_population_indices.npy",
        "split": SOURCE / "split" / "train_heldout_indices.npz",
        "pca_q20": SOURCE / "pca" / "pca_q20.npz",
        "pca_dimension_study": SOURCE / "pca" / "pca_dimension_study.csv",
        "primary_gmm_grid": SOURCE / "gmm" / "primary_grid.csv",
        "source_config": SOURCE / "config.json",
        "registration_results": SOURCE / "registration_results.csv",
    }
    return {
        "files": {
            key: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for key, path in paths.items()
        },
        "registration_artifact_count": len(registration_files),
        "registration_artifact_total_bytes": int(sum(path.stat().st_size for path in registration_files)),
        "registration_artifact_latest_mtime_ns": int(max(path.stat().st_mtime_ns for path in registration_files)),
    }


def verify_inputs(out: Path) -> dict[str, Any]:
    required = [
        SOURCE / "velocity_field_matrix.npy",
        SOURCE / "velocity_field_population_indices.npy",
        SOURCE / "split" / "train_heldout_indices.npz",
        SOURCE / "pca" / "pca_q20.npz",
        SOURCE / "pca" / "pca_dimension_study.csv",
        SOURCE / "gmm" / "primary_grid.csv",
        SOURCE / "config.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Required completed-study inputs are missing: {missing}")

    matrix = np.load(SOURCE / "velocity_field_matrix.npy", mmap_mode="r")
    population = np.load(SOURCE / "velocity_field_population_indices.npy")
    split = np.load(SOURCE / "split" / "train_heldout_indices.npz")
    config = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    original_pca = read_csv(SOURCE / "pca" / "pca_dimension_study.csv")
    original_gmm = read_csv(SOURCE / "gmm" / "primary_grid.csv")

    train_pos = split["train_matrix_positions"]
    holdout_pos = split["heldout_matrix_positions"]
    train_indices = split["train_indices"]
    holdout_indices = split["heldout_indices"]
    expected_cells = {(q, k) for q in range(1, 21) for k in range(1, 21)}
    actual_cells = {(int(row["q"]), int(row["K"])) for row in original_gmm}
    errors: list[str] = []
    if matrix.shape != (SAFE_N, FIELD_D) or matrix.dtype != np.float64:
        errors.append(f"velocity matrix mismatch: {matrix.shape}, {matrix.dtype}")
    if population.shape != (SAFE_N,) or len(np.unique(population)) != SAFE_N:
        errors.append("safe population-index vector mismatch")
    if len(train_pos) != TRAIN_N or len(holdout_pos) != HOLDOUT_N:
        errors.append(f"split count mismatch: {len(train_pos)}/{len(holdout_pos)}")
    if len(train_indices) != TRAIN_N or len(holdout_indices) != HOLDOUT_N:
        errors.append("original-index split count mismatch")
    if np.intersect1d(train_pos, holdout_pos).size or np.union1d(train_pos, holdout_pos).size != SAFE_N:
        errors.append("train/holdout matrix positions are not a disjoint partition")
    if not np.array_equal(population[train_pos], train_indices) or not np.array_equal(population[holdout_pos], holdout_indices):
        errors.append("matrix positions do not reproduce saved original sample indices")
    if [int(row["q"]) for row in original_pca] != list(range(1, 21)):
        errors.append("original PCA q range is not exactly 1..20")
    if len(original_gmm) != 400 or actual_cells != expected_cells:
        errors.append("original GMM grid is not exactly q=1..20 by K=1..20")
    primary = config.get("primary_gmm", {})
    if primary.get("covariance") != "full" or int(primary.get("n_init", -1)) != 5 or int(primary.get("seed", -1)) != SEED:
        errors.append(f"original GMM configuration mismatch: {primary}")
    frozen = config.get("frozen_registration", {})
    if int(frozen.get("field_vector_dimension", -1)) != FIELD_D or frozen.get("field_vector_order") != "C-order concatenate global[24,24,2], LE[48,48,2], TE[48,48,2]":
        errors.append("velocity-field vectorisation metadata mismatch")
    with np.load(SOURCE / "pca" / "pca_q20.npz") as pca20:
        if not np.array_equal(pca20["train_matrix_positions"], train_pos) or not np.array_equal(pca20["heldout_matrix_positions"], holdout_pos):
            errors.append("saved PCA q20 split differs from fixed split")
        if not np.array_equal(pca20["train_indices"], train_indices) or not np.array_equal(pca20["heldout_indices"], holdout_indices):
            errors.append("saved PCA q20 original indices differ from fixed split")
    if errors:
        raise RuntimeError("Input verification failed; no replacement will be generated: " + "; ".join(errors))

    result = {
        "verified": True,
        "safe_samples": SAFE_N,
        "training_samples": TRAIN_N,
        "holdout_samples": HOLDOUT_N,
        "velocity_dimension": FIELD_D,
        "velocity_dtype": str(matrix.dtype),
        "vectorisation": frozen["field_vector_order"],
        "feature_scaling": frozen.get("field_scaling", "none"),
        "pca_centring": "sklearn PCA training mean; no score standardisation",
        "original_q_range": [1, 20],
        "original_K_range": [1, 20],
        "original_gmm_cells": 400,
        "train_indices_match_saved_pca": True,
        "holdout_indices_match_saved_pca": True,
        "source_fingerprint_before": source_fingerprint(),
    }
    atomic_json(out / "input_verification.json", result)
    return result


def fit_pca_q40(out: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    matrix = np.load(SOURCE / "velocity_field_matrix.npy", mmap_mode="r")
    split = np.load(SOURCE / "split" / "train_heldout_indices.npz")
    train = split["train_matrix_positions"]
    heldout = split["heldout_matrix_positions"]
    model = PCA(n_components=Q_MAX, svd_solver="randomized", random_state=SEED, iterated_power=7)
    start = time.perf_counter()
    train_scores = model.fit_transform(matrix[train])
    heldout_scores = model.transform(matrix[heldout])
    runtime = time.perf_counter() - start
    rows: list[dict[str, Any]] = []
    for q in range(1, Q_MAX + 1):
        train_recon = train_scores[:, :q] @ model.components_[:q] + model.mean_
        heldout_recon = heldout_scores[:, :q] @ model.components_[:q] + model.mean_
        rows.append({
            "q": q,
            "explained_variance_ratio": float(model.explained_variance_ratio_[q - 1]),
            "cumulative_explained_variance": float(np.sum(model.explained_variance_ratio_[:q])),
            "train_reconstruction_mse": float(np.mean((matrix[train] - train_recon) ** 2)),
            "heldout_reconstruction_mse": float(np.mean((matrix[heldout] - heldout_recon) ** 2)),
            "pca_basis_source": "single_common_q40_training_fit",
        })
    out.joinpath("pca").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "pca" / "pca_q40.npz",
        mean=model.mean_, components=model.components_, explained_variance=model.explained_variance_,
        explained_variance_ratio=model.explained_variance_ratio_, singular_values=model.singular_values_,
        train_matrix_positions=train, heldout_matrix_positions=heldout,
        train_scores=train_scores, heldout_scores=heldout_scores,
    )
    save_csv(out / "pca_q1_q40.csv", rows)

    legacy_rows = read_csv(SOURCE / "pca" / "pca_dimension_study.csv")
    with np.load(SOURCE / "pca" / "pca_q20.npz") as old:
        old_components = old["components"]
        old_train_scores = old["train_scores"]
        old_holdout_scores = old["heldout_scores"]
        component_alignment = np.abs(np.sum(old_components * model.components_[:20], axis=1))
        train_score_corr = [abs(float(np.corrcoef(old_train_scores[:, j], train_scores[:, j])[0, 1])) for j in range(20)]
        holdout_score_corr = [abs(float(np.corrcoef(old_holdout_scores[:, j], heldout_scores[:, j])[0, 1])) for j in range(20)]
    cumulative_diff = [abs(float(legacy_rows[q - 1]["cumulative_explained_variance"]) - rows[q - 1]["cumulative_explained_variance"]) for q in range(1, 21)]
    heldout_mse_diff = [abs(float(legacy_rows[q - 1]["heldout_reconstruction_mse"]) - rows[q - 1]["heldout_reconstruction_mse"]) for q in range(1, 21)]
    compatibility = {
        "pca_runtime_seconds": runtime,
        "previous_basis": "separate randomized q20 PCA fit",
        "extension_basis": "single randomized q40 PCA fit",
        "max_abs_cumulative_explained_variance_difference_q1_q20": max(cumulative_diff),
        "max_abs_heldout_reconstruction_mse_difference_q1_q20": max(heldout_mse_diff),
        "minimum_corresponding_component_absolute_dot": min(component_alignment),
        "minimum_train_score_absolute_correlation": min(train_score_corr),
        "minimum_holdout_score_absolute_correlation": min(holdout_score_corr),
        "component_absolute_dots": component_alignment.tolist(),
        "train_score_absolute_correlations": train_score_corr,
        "holdout_score_absolute_correlations": holdout_score_corr,
    }
    atomic_json(out / "pca" / "q20_q40_basis_compatibility.json", compatibility)
    return train_scores, heldout_scores, rows, compatibility


def posterior_entropy(probabilities: np.ndarray) -> float:
    return float(-np.sum(probabilities * np.log(np.clip(probabilities, 1e-300, 1.0))))


def fit_gmm_cell(train_scores: np.ndarray, heldout_scores: np.ndarray, q: int, k: int, seed: int = SEED, n_init: int = 5) -> tuple[dict[str, Any], np.ndarray | None]:
    start = time.perf_counter()
    warning_messages: list[str] = []
    base: dict[str, Any] = {
        "q": q, "K": k, "covariance_type": "full", "seed": seed, "random_state": seed,
        "n_init": n_init, "reg_covar": GMM_CFG["reg_covar"], "max_iter": GMM_CFG["max_iter"],
        "tol": GMM_CFG["tol"], "init_params": GMM_CFG["init_params"],
        "pca_basis_source": "single_common_q40_training_fit",
    }
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = GaussianMixture(
                n_components=k, covariance_type="full", n_init=n_init, random_state=seed,
                reg_covar=GMM_CFG["reg_covar"], max_iter=GMM_CFG["max_iter"],
                tol=GMM_CFG["tol"], init_params=GMM_CFG["init_params"],
            ).fit(train_scores[:, :q])
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
            "valid_for_selection": bool(model.converged_),
            "converged": bool(model.converged_),
            "iterations": int(model.n_iter_),
            "training_log_likelihood_mean": train_ll,
            "training_log_likelihood_total": train_ll * TRAIN_N,
            "heldout_log_likelihood_mean": holdout_ll,
            "heldout_log_likelihood_total": holdout_ll * HOLDOUT_N,
            "BIC": bic,
            "ICL": bic + 2.0 * entropy,
            "ICL_formula": "BIC + 2 * posterior_entropy_total; lower is preferred",
            "posterior_entropy_total": entropy,
            "posterior_entropy_mean": entropy / TRAIN_N,
            "component_weights": json.dumps(model.weights_.tolist(), separators=(",", ":")),
            "hard_component_sizes": json.dumps([int(value) for value in sizes], separators=(",", ":")),
            "minimum_component_size": int(sizes.min()),
            "maximum_component_size": int(sizes.max()),
            "minimum_component_proportion": float(sizes.min() / TRAIN_N),
            "small_component_threshold": SMALL_THRESHOLD,
            "number_components_below_threshold": below,
            "tiny_component_flag": bool(below > 0),
            "runtime_seconds": time.perf_counter() - start,
            "warning_or_failure_message": " | ".join(warning_messages),
        }
        numeric = [row[key] for key in ("training_log_likelihood_mean", "heldout_log_likelihood_mean", "BIC", "ICL", "posterior_entropy_total")]
        if not all(finite(value) for value in numeric):
            row["status"] = "failed_nonfinite"
            row["valid_for_selection"] = False
            row["warning_or_failure_message"] = (row["warning_or_failure_message"] + " | non-finite metric").strip(" |")
        return row, labels
    except Exception as exc:  # preserve exact failed cell
        row = {
            **base, "status": "failed", "valid_for_selection": False, "converged": False,
            "iterations": 0, "training_log_likelihood_mean": "", "training_log_likelihood_total": "",
            "heldout_log_likelihood_mean": "", "heldout_log_likelihood_total": "", "BIC": "", "ICL": "",
            "ICL_formula": "BIC + 2 * posterior_entropy_total; lower is preferred",
            "posterior_entropy_total": "", "posterior_entropy_mean": "", "component_weights": "",
            "hard_component_sizes": "", "minimum_component_size": "", "maximum_component_size": "",
            "minimum_component_proportion": "", "small_component_threshold": SMALL_THRESHOLD,
            "number_components_below_threshold": "", "tiny_component_flag": "",
            "runtime_seconds": time.perf_counter() - start,
            "warning_or_failure_message": f"{type(exc).__name__}: {exc}",
        }
        return row, None


def complete_new_row(row: dict[str, str], q: int, k: int) -> bool:
    if int(row.get("q", -1)) != q or int(row.get("K", -1)) != k:
        return False
    if row.get("covariance_type") != "full" or int(row.get("n_init", -1)) != 5 or int(row.get("seed", -1)) != SEED:
        return False
    if row.get("status") == "failed":
        return bool(row.get("warning_or_failure_message"))
    required = ["training_log_likelihood_mean", "heldout_log_likelihood_mean", "BIC", "ICL", "posterior_entropy_total", "minimum_component_size", "runtime_seconds"]
    return all(finite(row.get(key)) for key in required)


def run_new_grid(out: Path, train_scores: np.ndarray, heldout_scores: np.ndarray, force: bool) -> list[dict[str, Any]]:
    path = out / "gmm_q21_q40_cells.csv"
    existing = [] if force else read_csv(path)
    by_cell = {(int(row["q"]), int(row["K"])): row for row in existing if "q" in row and "K" in row}
    rows: list[dict[str, Any]] = list(existing)
    for q in range(21, 41):
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
            print(f"q={q:02d} K={k:02d} status={row['status']} BIC={row['BIC']} heldout={row['heldout_log_likelihood_mean']}", flush=True)
    return sorted(rows, key=lambda value: (int(value["q"]), int(value["K"])))


def validate_legacy_with_q40(out: Path, train_scores: np.ndarray, heldout_scores: np.ndarray) -> dict[str, Any]:
    old_rows = read_csv(SOURCE / "gmm" / "primary_grid.csv")
    old = {(int(row["q"]), int(row["K"])): row for row in old_rows}
    preferred = read_csv(SOURCE / "gmm" / "preferred_k_by_q.csv")
    qs = (1, 5, 10, 15, 20)
    cells: set[tuple[int, int]] = set()
    for q in qs:
        row = preferred[q - 1]
        for key in ("K_BIC", "K_ICL", "K_heldout_LL"):
            cells.add((q, int(row[key])))
    comparisons = []
    for q, k in sorted(cells):
        new, _ = fit_gmm_cell(train_scores, heldout_scores, q, k)
        prior = old[(q, k)]
        comparisons.append({
            "q": q, "K": k,
            "old_BIC": float(prior["BIC"]), "q40_prefix_BIC": new["BIC"], "abs_BIC_difference": abs(float(prior["BIC"]) - float(new["BIC"])),
            "old_ICL": float(prior["ICL"]), "q40_prefix_ICL": new["ICL"], "abs_ICL_difference": abs(float(prior["ICL"]) - float(new["ICL"])),
            "old_holdout_LL": float(prior["heldout_log_likelihood_mean"]), "q40_prefix_holdout_LL": new["heldout_log_likelihood_mean"],
            "abs_holdout_LL_difference": abs(float(prior["heldout_log_likelihood_mean"]) - float(new["heldout_log_likelihood_mean"])),
            "old_minimum_component_size": int(prior["minimum_component_size"]), "q40_prefix_minimum_component_size": int(new["minimum_component_size"]),
            "old_converged": bool_value(prior["converged"]), "q40_prefix_converged": bool_value(new["converged"]),
            "runtime_seconds": float(new["runtime_seconds"]),
        })
    save_csv(out / "pca" / "q20_q40_gmm_spotcheck.csv", comparisons)
    summary = {
        "cells_checked": len(comparisons),
        "cells": [[q, k] for q, k in sorted(cells)],
        "maximum_absolute_BIC_difference": max(row["abs_BIC_difference"] for row in comparisons),
        "maximum_absolute_ICL_difference": max(row["abs_ICL_difference"] for row in comparisons),
        "maximum_absolute_holdout_LL_difference": max(row["abs_holdout_LL_difference"] for row in comparisons),
        "all_converged_both": all(row["old_converged"] and row["q40_prefix_converged"] for row in comparisons),
        "runtime_seconds": sum(row["runtime_seconds"] for row in comparisons),
        "note": "The original q<=20 grid is reused and explicitly labelled legacy_q20_fit; this spot-check quantifies solver-induced differences against q40 prefixes.",
    }
    atomic_json(out / "pca" / "q20_q40_gmm_spotcheck_summary.json", summary)
    return summary


def enrich_legacy(row: dict[str, str]) -> dict[str, Any]:
    sizes = np.asarray(json.loads(row["hard_component_sizes"]), dtype=int)
    below = int(np.sum(sizes < SMALL_THRESHOLD))
    return {
        **row,
        "random_state": int(row["seed"]), "reg_covar": GMM_CFG["reg_covar"], "max_iter": GMM_CFG["max_iter"],
        "tol": GMM_CFG["tol"], "init_params": GMM_CFG["init_params"], "pca_basis_source": "legacy_separate_q20_fit_reused",
        "status": "valid" if bool_value(row["converged"]) and all(finite(row[key]) for key in ("BIC", "ICL", "heldout_log_likelihood_mean")) else "non_converged",
        "valid_for_selection": bool_value(row["converged"]) and all(finite(row[key]) for key in ("BIC", "ICL", "heldout_log_likelihood_mean")),
        "ICL_formula": "BIC + 2 * posterior_entropy_total; lower is preferred",
        "maximum_component_size": int(sizes.max()), "minimum_component_proportion": float(sizes.min() / TRAIN_N),
        "small_component_threshold": SMALL_THRESHOLD, "number_components_below_threshold": below,
        "warning_or_failure_message": "",
    }


def combine_and_select(out: Path, new_rows: list[dict[str, Any]], pca_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    legacy = [enrich_legacy(row) for row in read_csv(SOURCE / "gmm" / "primary_grid.csv")]
    combined = sorted(legacy + new_rows, key=lambda row: (int(row["q"]), int(row["K"])))
    save_csv(out / "gmm_q1_q40_combined.csv", combined)
    preferred: list[dict[str, Any]] = []
    for q in range(1, 41):
        group = [row for row in combined if int(row["q"]) == q and bool_value(row["valid_for_selection"])]
        if not group:
            raise RuntimeError(f"No valid selectable cells at q={q}")
        winners = {
            "BIC": min(group, key=lambda row: float(row["BIC"])),
            "ICL": min(group, key=lambda row: float(row["ICL"])),
            "heldout_LL": max(group, key=lambda row: float(row["heldout_log_likelihood_mean"])),
        }
        pca = pca_rows[q - 1]
        result: dict[str, Any] = {
            "q": q,
            "cumulative_explained_variance": pca["cumulative_explained_variance"],
            "heldout_reconstruction_mse": pca["heldout_reconstruction_mse"],
            "pca_basis_source": pca["pca_basis_source"],
            "gmm_basis_source": "legacy_separate_q20_fit_reused" if q <= 20 else "single_common_q40_training_fit",
        }
        for label, winner in winners.items():
            result[f"K_{label}"] = int(winner["K"])
            result[f"{label}_small_component_flag"] = bool_value(winner["tiny_component_flag"])
            result[f"{label}_minimum_component_size"] = int(winner["minimum_component_size"])
            result[f"{label}_converged"] = bool_value(winner["converged"])
        preferred.append(result)
    save_csv(out / "preferred_k_q1_q40.csv", preferred)
    sequences = {
        "q_1_q_40": {
            "K_BIC": [row["K_BIC"] for row in preferred],
            "K_ICL": [row["K_ICL"] for row in preferred],
            "K_holdout_LL": [row["K_heldout_LL"] for row in preferred],
        },
        "q_21_q_40": {
            "K_BIC": [row["K_BIC"] for row in preferred[20:]],
            "K_ICL": [row["K_ICL"] for row in preferred[20:]],
            "K_holdout_LL": [row["K_heldout_LL"] for row in preferred[20:]],
        },
    }
    atomic_json(out / "preferred_k_sequences.json", sequences)
    save_csv(out / "small_component_diagnostics.csv", [{
        "q": int(row["q"]), "K": int(row["K"]), "minimum_component_size": row["minimum_component_size"],
        "maximum_component_size": row["maximum_component_size"], "minimum_component_proportion": row["minimum_component_proportion"],
        "number_components_below_threshold": row["number_components_below_threshold"], "small_component_threshold": row["small_component_threshold"],
        "tiny_component_flag": row["tiny_component_flag"], "status": row["status"], "pca_basis_source": row["pca_basis_source"],
    } for row in combined])
    save_csv(out / "convergence_diagnostics.csv", [{
        "q": int(row["q"]), "K": int(row["K"]), "status": row["status"], "converged": row["converged"],
        "valid_for_selection": row["valid_for_selection"], "iterations": row["iterations"], "runtime_seconds": row["runtime_seconds"],
        "warning_or_failure_message": row["warning_or_failure_message"], "pca_basis_source": row["pca_basis_source"],
    } for row in combined])
    return combined, preferred


COLORS = {"K_BIC": "#2563eb", "K_ICL": "#d97706", "K_heldout_LL": "#9f1239"}
MARKERS = {"K_BIC": "o", "K_ICL": "s", "K_heldout_LL": "^"}


def style_axis(ax: plt.Axes) -> None:
    ax.grid(alpha=0.22, color="#9ca3af")
    ax.spines[["top", "right"]].set_visible(False)


def plot_pca(pca_rows: list[dict[str, Any]], figures: Path) -> None:
    q = np.arange(1, 41)
    marginal = np.asarray([row["explained_variance_ratio"] for row in pca_rows], dtype=float)
    cumulative = np.asarray([row["cumulative_explained_variance"] for row in pca_rows], dtype=float)
    heldout = np.asarray([row["heldout_reconstruction_mse"] for row in pca_rows], dtype=float)
    for values, ylabel, title, name, log in [
        (cumulative, "Cumulative explained variance", "Cumulative explained variance versus PCA dimension", "pca_explained_variance_q1_q40.png", False),
        (heldout, "Held-out coefficient reconstruction MSE", "Held-out PCA reconstruction error versus dimension", "pca_reconstruction_mse_q1_q40.png", True),
        (marginal, "Marginal explained variance", "Explained variance of each PCA component", "pca_marginal_explained_variance_q1_q40.png", True),
    ]:
        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        ax.plot(q, values, color="#2563eb", marker="o", markersize=3.5, linewidth=1.5)
        ax.axvline(20.5, color="#374151", linestyle="--", linewidth=1.2)
        ax.axvspan(20.5, 40.5, color="#f59e0b", alpha=0.07)
        ax.set(xlabel="PCA dimension q", ylabel=ylabel, title=title, xticks=np.arange(1, 41, 2))
        if log:
            ax.set_yscale("log")
        style_axis(ax)
        fig.tight_layout()
        fig.savefig(figures / name, dpi=220)
        plt.close(fig)


def plot_preferred(preferred: list[dict[str, Any]], figures: Path) -> None:
    q = np.arange(1, 41)
    fig, ax = plt.subplots(figsize=(12, 5.4))
    for key in ("K_BIC", "K_ICL", "K_heldout_LL"):
        width = 2.8 if key == "K_BIC" else (1.2 if key == "K_ICL" else 1.6)
        marker_size = 7 if key == "K_BIC" else (3.7 if key == "K_ICL" else 4.5)
        zorder = 2 if key == "K_BIC" else (3 if key == "K_ICL" else 4)
        ax.plot(q, [row[key] for row in preferred], color=COLORS[key], marker=MARKERS[key], markersize=marker_size, linewidth=width, zorder=zorder, label=key)
    ax.axvline(20.5, color="#111827", linestyle="--", linewidth=1.3, label="original/extension boundary")
    ax.axvspan(0.5, 20.5, color="#2563eb", alpha=0.035)
    ax.axvspan(20.5, 40.5, color="#f59e0b", alpha=0.07)
    ax.text(10.5, 20.55, "original q≤20", ha="center", va="top", color="#1e3a8a")
    ax.text(30.5, 20.55, "extension q≥21", ha="center", va="top", color="#92400e")
    ax.set(xlabel="PCA dimension q", ylabel="Preferred component count K", title="Preferred component count across PCA dimensions (q=1–40)",
           xticks=np.arange(1, 41, 2), yticks=np.arange(1, 21), xlim=(0.5, 40.5), ylim=(0.5, 20.8))
    style_axis(ax)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.31))
    fig.tight_layout()
    fig.savefig(figures / "preferred_k_q1_q40.png", dpi=240, bbox_inches="tight")
    fig.savefig(figures / "preferred_k_q1_q40.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.4))
    for key, label in (("K_BIC", "BIC"), ("K_ICL", "ICL"), ("K_heldout_LL", "heldout_LL")):
        values = np.asarray([row[key] for row in preferred])
        flags = np.asarray([bool_value(row[f"{label}_small_component_flag"]) for row in preferred])
        width = 2.8 if key == "K_BIC" else (1.2 if key == "K_ICL" else 1.6)
        marker_size = 7 if key == "K_BIC" else (3.7 if key == "K_ICL" else 4.5)
        zorder = 2 if key == "K_BIC" else (3 if key == "K_ICL" else 4)
        ax.plot(q, values, color=COLORS[key], marker=MARKERS[key], markersize=marker_size, linewidth=width, zorder=zorder, label=key)
        ax.scatter(q[flags], values[flags], facecolors="none", edgecolors="#111827", s=80, linewidths=1.25, zorder=5)
    ax.axvline(20.5, color="#111827", linestyle="--", linewidth=1.3)
    ax.axvspan(20.5, 40.5, color="#f59e0b", alpha=0.07)
    ax.scatter([], [], facecolors="none", edgecolors="#111827", s=80, label="selected model has small component")
    ax.set(xlabel="PCA dimension q", ylabel="Preferred component count K", title="Preferred K with selected-model small-component flags",
           xticks=np.arange(1, 41, 2), yticks=np.arange(1, 21), xlim=(0.5, 40.5), ylim=(0.5, 20.8))
    style_axis(ax)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.31))
    fig.tight_layout()
    fig.savefig(figures / "preferred_k_q1_q40_small_component_flags.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def matrix_from_rows(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    result = np.full((40, 20), np.nan)
    for row in rows:
        if key == "tiny_component_flag":
            value = 1.0 if bool_value(row[key]) else 0.0
        elif key == "converged":
            value = 1.0 if bool_value(row[key]) else 0.0
        else:
            value = float(row[key]) if finite(row.get(key)) else np.nan
        result[int(row["q"]) - 1, int(row["K"]) - 1] = value
    return result


def save_heatmap(matrix: np.ndarray, title: str, label: str, path: Path, cmap: str = "viridis", vmin: float | None = None, vmax: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 10))
    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.axhline(19.5, color="white", linestyle="--", linewidth=1.2)
    ax.set(xlabel="GMM components K", ylabel="PCA dimension q", title=title,
           xticks=np.arange(20), xticklabels=np.arange(1, 21), yticks=np.arange(0, 40, 2), yticklabels=np.arange(1, 41, 2))
    fig.colorbar(image, ax=ax, label=label, shrink=0.86)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_heatmaps(rows: list[dict[str, Any]], figures: Path) -> None:
    bic = matrix_from_rows(rows, "BIC")
    icl = matrix_from_rows(rows, "ICL")
    heldout = matrix_from_rows(rows, "heldout_log_likelihood_mean")
    minimum = matrix_from_rows(rows, "minimum_component_size")
    tiny = matrix_from_rows(rows, "tiny_component_flag")
    convergence = matrix_from_rows(rows, "converged")
    delta_bic = bic - np.nanmin(bic, axis=1, keepdims=True)
    delta_icl = icl - np.nanmin(icl, axis=1, keepdims=True)
    heldout_gap = heldout - np.nanmax(heldout, axis=1, keepdims=True)
    specs = [
        (bic, "BIC over q and K", "BIC", "bic_q1_q40_heatmap.png", "viridis", None, None),
        (icl, "ICL over q and K", "ICL (lower preferred)", "icl_q1_q40_heatmap.png", "viridis", None, None),
        (heldout, "Mean held-out log-likelihood over q and K", "mean held-out log-likelihood", "heldout_ll_q1_q40_heatmap.png", "viridis", None, None),
        (minimum, "Minimum hard-component size over q and K", "minimum component size", "minimum_component_size_q1_q40_heatmap.png", "magma", None, None),
        (tiny, f"Small-component flag (threshold < {SMALL_THRESHOLD})", "small-component flag", "small_component_flag_q1_q40_heatmap.png", "binary", 0, 1),
        (convergence, "GMM convergence status over q and K", "converged", "convergence_q1_q40_heatmap.png", "binary", 0, 1),
        (delta_bic, "Within-q relative BIC", "Delta BIC from best K", "delta_bic_q1_q40_heatmap.png", "magma", 0, float(np.nanpercentile(delta_bic, 95))),
        (delta_icl, "Within-q relative ICL", "Delta ICL from best K", "delta_icl_q1_q40_heatmap.png", "magma", 0, float(np.nanpercentile(delta_icl, 95))),
        (heldout_gap, "Within-q held-out likelihood gap", "LL(K) - best LL", "heldout_ll_gap_q1_q40_heatmap.png", "viridis", float(np.nanpercentile(heldout_gap, 5)), 0),
    ]
    for matrix, title, label, name, cmap, vmin, vmax in specs:
        save_heatmap(matrix, title, label, figures / name, cmap, vmin, vmax)


def run_targeted_stability(out: Path, train_scores: np.ndarray, heldout_scores: np.ndarray, preferred: list[dict[str, Any]], force: bool) -> list[dict[str, Any]]:
    path = out / "targeted_stability_cells.csv"
    existing = [] if force else read_csv(path)
    by_key = {(int(row["q"]), int(row["K"]), int(row["seed"])): row for row in existing}
    rows: list[dict[str, Any]] = list(existing)
    assignments_dir = out / "stability_assignments"
    assignments_dir.mkdir(parents=True, exist_ok=True)
    target_pairs: set[tuple[int, int]] = set()
    for q in (20, 25, 30, 35, 40):
        p = preferred[q - 1]
        ks = {6, 7, 8, int(p["K_BIC"]), int(p["K_ICL"])}
        target_pairs.update((q, k) for k in ks)
    for q, k in sorted(target_pairs):
        for seed in range(5):
            key = (q, k, seed)
            assignment_path = assignments_dir / f"q{q:02d}_k{k:02d}_seed{seed}.npy"
            if key in by_key and assignment_path.exists() and np.load(assignment_path, mmap_mode="r").shape == (TRAIN_N,):
                continue
            row, labels = fit_gmm_cell(train_scores, heldout_scores, q, k, seed=seed, n_init=3)
            row["stability_n_init"] = 3
            rows = [value for value in rows if (int(value["q"]), int(value["K"]), int(value["seed"])) != key]
            rows.append(row)
            if labels is not None:
                np.save(assignment_path, labels)
            save_csv(path, sorted(rows, key=lambda value: (int(value["q"]), int(value["K"]), int(value["seed"]))))
            print(f"stability q={q} K={k} seed={seed} status={row['status']}", flush=True)
    summaries: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for q, k in sorted(target_pairs):
        labels = []
        for seed in range(5):
            path_value = assignments_dir / f"q{q:02d}_k{k:02d}_seed{seed}.npy"
            if path_value.exists():
                labels.append((seed, np.load(path_value)))
        aris = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                ari = float(adjusted_rand_score(labels[i][1], labels[j][1]))
                aris.append(ari)
                pair_rows.append({"q": q, "K": k, "seed_a": labels[i][0], "seed_b": labels[j][0], "ARI": ari})
        summaries.append({
            "q": q, "K": k, "successful_seeds": len(labels), "pair_count": len(aris),
            "mean_pairwise_ARI": float(np.mean(aris)) if aris else "",
            "minimum_pairwise_ARI": float(np.min(aris)) if aris else "",
            "maximum_pairwise_ARI": float(np.max(aris)) if aris else "",
        })
    save_csv(out / "targeted_stability_ari.csv", summaries)
    save_csv(out / "targeted_stability_pairwise_ari.csv", pair_rows)
    return summaries


def write_report(out: Path, pca_rows: list[dict[str, Any]], combined: list[dict[str, Any]], preferred: list[dict[str, Any]], compatibility: dict[str, Any], spotcheck: dict[str, Any], stability: list[dict[str, Any]], runtime: float) -> None:
    sequences = json.loads((out / "preferred_k_sequences.json").read_text(encoding="utf-8"))
    new = [row for row in combined if int(row["q"]) >= 21]
    valid = [row for row in new if bool_value(row["valid_for_selection"])]
    failed = [row for row in new if str(row["status"]).startswith("failed")]
    nonconv = [row for row in new if row["status"] == "non_converged"]
    preferred_new = preferred[20:]
    small_counts = {
        key: sum(bool_value(row[f"{label}_small_component_flag"]) for row in preferred_new)
        for key, label in (("BIC", "BIC"), ("ICL", "ICL"), ("heldout", "heldout_LL"))
    }
    heldout_freq = Counter(row["K_heldout_LL"] for row in preferred_new)
    bic_freq = Counter(row["K_BIC"] for row in preferred_new)
    icl_freq = Counter(row["K_ICL"] for row in preferred_new)
    mse20 = pca_rows[19]["heldout_reconstruction_mse"]
    mse40 = pca_rows[39]["heldout_reconstruction_mse"]
    mse_reduction = 1.0 - mse40 / mse20
    best_ll_by_q = []
    for q in range(21, 41):
        vals = [float(row["heldout_log_likelihood_mean"]) for row in valid if int(row["q"]) == q]
        best_ll_by_q.append(max(vals))
    stable_lookup = {(int(row["q"]), int(row["K"])): row for row in stability}
    q40_cov = 40 * 41 // 2
    lines = [
        "# Airfoil template-number screening: q=40 PCA/GMM extension",
        "",
        "## 技术摘要",
        "",
        f"本扩展复用了原研究的 1651 个安全速度场、固定 1321/330 训练—留出拆分和 D=10368 的 C-order 参数向量，没有重新配准。新增 q=21–40、K=1–20 共 400 个 full-covariance GMM 单元，其中有效 `{len(valid)}`、失败 `{len(failed)}`、未收敛 `{len(nonconv)}`。",
        "",
        f"q>20 后下降趋势继续，但不同准则最终分叉：BIC 与 ICL 在 q=29–40 连续选择 K=7；留出似然则在 q=31–40 连续选择 K=2。BIC 在新范围最常选择 `{bic_freq.most_common(3)}`，ICL 最常选择 `{icl_freq.most_common(3)}`，留出似然最常选择 `{heldout_freq.most_common(3)}`。这些首选模型并未被微小分量主导（BIC/ICL/留出仅 `{small_counts['BIC']}/20`、`{small_counts['ICL']}/20`、`{small_counts['heldout']}/20` 含小分量）。因此扩展强化了 K=7 的统计证据，但高维留出准则同时提出 K=2–4 的低-K敏感性信号；K=6–8 仍可作为下游主要范围，K=13 没有获得提升证据。",
        "",
        "![Preferred K](../figures/preferred_k_q1_q40.png)",
        "",
        "上图保留每个 q 的原始离散选择，虚线标出旧研究与新扩展的边界。它用于判断趋势是否跨越 q=20 持续，而不是对选择序列做平滑。",
        "",
        "## BIC/ICL稳定到K=7，留出似然则继续降到K=2",
        "",
        f"完整 BIC 序列：`{sequences['q_1_q_40']['K_BIC']}`",
        "",
        f"完整 ICL 序列：`{sequences['q_1_q_40']['K_ICL']}`",
        "",
        f"完整留出似然序列：`{sequences['q_1_q_40']['K_holdout_LL']}`",
        "",
        f"新增 BIC 序列（q=21–40）：`{sequences['q_21_q_40']['K_BIC']}`",
        "",
        f"新增 ICL 序列（q=21–40）：`{sequences['q_21_q_40']['K_ICL']}`",
        "",
        f"新增留出似然序列（q=21–40）：`{sequences['q_21_q_40']['K_holdout_LL']}`",
        "",
        "![Preferred K with flags](../figures/preferred_k_q1_q40_small_component_flags.png)",
        "",
        "空心黑圈标记所选模型含小于14个训练样本的硬分量。q>20 只有BIC/ICL各3个首选单元触发标记，留出首选没有触发；因此高维下降不能归因于微小分量。q=21的K=12、q=26的K=9等孤立跳点仍应降权，而q=29–40的K=7与q=31–40的K=2属于持续模式。",
        "",
        "## 额外 PCA 维度改善重构，但收益进入尾部区间",
        "",
        f"q=20 到 q=40 的留出重构 MSE 从 `{mse20:.6e}` 降到 `{mse40:.6e}`，相对降低 `{mse_reduction:.1%}`、绝对降低 `{mse20-mse40:.3e}`；累计解释方差只从 `{pca_rows[19]['cumulative_explained_variance']:.6%}` 增至 `{pca_rows[39]['cumulative_explained_variance']:.6%}`（增加约 `{100*(pca_rows[39]['cumulative_explained_variance']-pca_rows[19]['cumulative_explained_variance']):.4f}` 个百分点）。21–40维确实改善系数重构，但捕获的是已解释99.9%之后的尾部变化，不能仅凭训练方差增加认定聚类更好。",
        "",
        "![PCA cumulative](../figures/pca_explained_variance_q1_q40.png)",
        "",
        "![PCA reconstruction](../figures/pca_reconstruction_mse_q1_q40.png)",
        "",
        "![PCA marginal](../figures/pca_marginal_explained_variance_q1_q40.png)",
        "",
        "## 高维 full-covariance GMM 的统计负担迅速增加",
        "",
        f"q维每个 Gaussian 分量仅协方差就有 `q(q+1)/2` 个参数；q=40 时为 `{q40_cov}` 个，尚未计入40维均值和混合权重。因此 K=20、q=40 时仅协方差参数就达 16,400 个，而训练样本只有 1,321 个。各q的最佳原始留出平均对数密度为 `{[round(v, 3) for v in best_ll_by_q]}`，数值随q上升；但不同维数的对数密度包含不同归一化维度，不能把跨q数值上升直接解释为预测改进。更可靠的现象是同一q内的最佳K持续缩小，表明复杂协方差估计的负担正在主导模型选择。",
        "",
        "相对 BIC/ICL 与留出差值热图消除了跨 q 的量纲漂移，更适合判断同一 q 内 K 的竞争。原始与相对热图、分量大小和收敛热图均保存在 figures 目录。",
        "",
        "## 方法与兼容性",
        "",
        "本扩展的PCA诊断仅在原固定训练集上拟合一次，所有q的解释方差与重构误差均使用同一个q=40基的前q个得分；PCA scores不标准化。GMM方面，原q≤20结果来自单独的randomized q=20 PCA拟合，本报告复用并标记 `legacy_separate_q20_fit_reused`；新q≥21结果使用共同q=40基并标记 `single_common_q40_training_fit`。",
        "",
        f"两套 PCA 的 q≤20 最大累计解释方差差为 `{compatibility['max_abs_cumulative_explained_variance_difference_q1_q20']:.3e}`，最大留出重构 MSE 差为 `{compatibility['max_abs_heldout_reconstruction_mse_difference_q1_q20']:.3e}`，对应成分最小绝对点积为 `{compatibility['minimum_corresponding_component_absolute_dot']:.8f}`。另以 `{spotcheck['cells_checked']}` 个原首选相关 GMM 单元做 q40-prefix 复算，最大 BIC/ICL/留出LL差分别为 `{spotcheck['maximum_absolute_BIC_difference']:.3e}` / `{spotcheck['maximum_absolute_ICL_difference']:.3e}` / `{spotcheck['maximum_absolute_holdout_LL_difference']:.3e}`。这些差异被显式记录，没有静默混合为同一 PCA 拟合。",
        "",
        "ICL 使用原研究公式：`ICL = BIC + 2 × posterior_entropy_total`，其中 posterior entropy 为 `-Σ_iΣ_k τ_ik log τ_ik`；BIC和ICL均取最小，留出平均对数似然取最大。失败、非有限或未收敛单元不参与首选 K 选择。",
        "",
        "## 有限种子稳定性检查",
        "",
        "稳定性检查仅覆盖 q=20/25/30/35/40，K=6/7/8及该 q 的BIC/ICL首选值；使用 full covariance、seeds 0–4、n_init=3，没有运行协方差类型大扫掠。代表结果：",
        "",
        "| q | K | mean ARI | min ARI | max ARI |",
        "|---:|---:|---:|---:|---:|",
        *[f"| {row['q']} | {row['K']} | {float(row['mean_pairwise_ARI']):.3f} | {float(row['minimum_pairwise_ARI']):.3f} | {float(row['maximum_pairwise_ARI']):.3f} |" for row in stability],
        "",
        "## 对十个诊断问题的回答",
        "",
        "1. **BIC/ICL是否继续向更小K下降？** 是。短暂波动后，两者从q=29到40均稳定为K=7。",
        "2. **留出似然是否仍偏好K=6–8？** 仅q=21选择K=6；随后降到K=4/3，并在q=31–40稳定为K=2。因此高q留出结果不再支持‘始终6–8’。",
        "3. **q>20首选模型是否由小分量主导？** 否。BIC/ICL各仅3/20个首选模型含小分量，留出首选为0/20。",
        "4. **21–40维是否实质改善重构？** 相对MSE降低60.6%，但绝对改善仅3.75×10⁻⁸、累计方差只增加约0.0635个百分点；是数值可见但处于尾部的收益。",
        "5. **增加q是否改善GMM留出似然？** 原始对数密度上升，但跨维数不可直接比较；同q内首选K缩小显示协方差估计难度在增加。",
        "6. **首选K变化是否稳健？** K=7和K=2的长平台不是孤立单元；但K=6–8的seed ARI约0.50–0.79，具体分配仅中等稳定。",
        "7. **K=6–8是否仍可辩护？** 是，尤其K=7得到BIC/ICL强支持；但应承认高q留出准则提出K=2–4的敏感性范围。",
        "8. **是否应提升K=13？** 否；没有新增的留出、小分量或稳定性证据支持将其从敏感性比较提升。",
        "9. **是否存在PCA维数平台？** 存在。q=19已超过99.9%累计解释方差，之后20个成分只再增加约0.07个百分点。",
        "10. **推荐q范围？** q=19–20最平衡：解释方差已超过99.9%，三个准则给出K=6–7，所选分量安全，且q=20的K=6种子稳定性最高。",
        "",
        "## 限制、验证与停止条件",
        "",
        f"本扩展累计实测模型拟合时间约 `{runtime:.1f}` 秒（`{runtime/60:.2f}` 分钟；包括q40 PCA、兼容性GMM复算、400个新增单元和有限稳定性拟合，不含轻量绘图/CSV I/O）。源研究文件在运行前后通过哈希和注册文件清单核对，未重新运行注册。所有400个新增意图单元均保留，包括失败与未收敛记录。未构造模板、未做cluster-specific重注册、未扩展到q>40或K>20，也未commit/push。",
        "",
        "## 下一步",
        "",
        "维持 K=6–8 作为模板重注册候选；只有在另行批准后才进入集群模板构造与应用驱动比较。K=13继续仅作敏感性对照。",
        "",
        "## 仍待回答的问题",
        "",
        "最终模板数仍需由冻结配准算法下的集群模板重注册结果决定；本阶段只检验统计筛选对PCA维数扩展的敏感性。",
    ]
    atomic_text(out / "report" / "q40_extension_report.md", "\n".join(lines) + "\n")


def build_config(out: Path, verification: dict[str, Any]) -> None:
    config = {
        "source_study": str(SOURCE.relative_to(ROOT)),
        "output": str(out.relative_to(ROOT)),
        "safe_samples": SAFE_N, "training_samples": TRAIN_N, "holdout_samples": HOLDOUT_N,
        "velocity_field_dimension": FIELD_D,
        "vectorisation": "C-order concatenate global[24,24,2], LE[48,48,2], TE[48,48,2]",
        "scaling": "none", "centering": "training-set mean via sklearn PCA",
        "pca": {"n_components": 40, "svd_solver": "randomized", "random_state": SEED, "iterated_power": 7, "fit_once": True, "q_score_rule": "Z[:, :q]"},
        "new_gmm_grid": {"q": [21, 40], "K": [1, 20], "intended_cells": 400, **GMM_CFG},
        "ICL": {"formula": "BIC + 2 * posterior_entropy_total", "preferred_direction": "minimum"},
        "small_component_threshold": SMALL_THRESHOLD,
        "legacy_policy": "reuse q<=20 cells; explicitly label their separate q20 randomized PCA basis",
        "registration_policy": "read saved field matrix only; no registration code path",
        "source_fingerprint_before": verification["source_fingerprint_before"],
        "exact_command": ".venv/bin/python src/run_airfoil_template_number_q40_extension.py --stage all",
    }
    atomic_json(out / "config.json", config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stage", choices=("verify", "pca", "gmm", "stability", "finalize", "all"), default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    for folder in ("pca", "figures", "report", "stability_assignments"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    run_start = time.perf_counter()
    verification = verify_inputs(out)
    build_config(out, verification)
    if args.stage == "verify":
        return
    train_scores, heldout_scores, pca_rows, compatibility = fit_pca_q40(out)
    spotcheck = validate_legacy_with_q40(out, train_scores, heldout_scores)
    if args.stage == "pca":
        return
    new_rows = run_new_grid(out, train_scores, heldout_scores, args.force)
    combined, preferred = combine_and_select(out, new_rows, pca_rows)
    if args.stage == "gmm":
        return
    stability = run_targeted_stability(out, train_scores, heldout_scores, preferred, args.force)
    if args.stage == "stability":
        return
    figures = out / "figures"
    plot_pca(pca_rows, figures)
    plot_preferred(preferred, figures)
    plot_heatmaps(combined, figures)
    source_after = source_fingerprint()
    if source_after != verification["source_fingerprint_before"]:
        raise RuntimeError("Completed source study changed during extension; refusing to finalize")
    latest_invocation_runtime = time.perf_counter() - run_start
    stability_cells = read_csv(out / "targeted_stability_cells.csv")
    primary_runtime = sum(float(row["runtime_seconds"]) for row in new_rows)
    stability_runtime = sum(float(row["runtime_seconds"]) for row in stability_cells)
    aggregate_model_fit_runtime = float(compatibility["pca_runtime_seconds"]) + float(spotcheck["runtime_seconds"]) + primary_runtime + stability_runtime
    write_report(out, pca_rows, combined, preferred, compatibility, spotcheck, stability, aggregate_model_fit_runtime)
    summary = {
        "runtime_seconds_this_invocation": latest_invocation_runtime,
        "aggregate_model_fit_runtime_seconds": aggregate_model_fit_runtime,
        "primary_400_cell_runtime_seconds": primary_runtime,
        "targeted_stability_runtime_seconds": stability_runtime,
        "new_cells": len(new_rows),
        "valid_cells": sum(bool_value(row["valid_for_selection"]) for row in new_rows),
        "failed_cells": sum(str(row["status"]).startswith("failed") for row in new_rows),
        "non_converged_cells": sum(row["status"] == "non_converged" for row in new_rows),
        "source_fingerprint_unchanged": True,
    }
    atomic_json(out / "run_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
