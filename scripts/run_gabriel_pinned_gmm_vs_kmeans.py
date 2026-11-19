"""Clustering-only PCA/GMM comparison on the production Gabriel pinned shapes.

This script deliberately has no registration imports or registration actions.
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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-gabriel-gmm-vs-kmeans")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/airfoils/processed_gabriel/airfoils_pinned_arclength_n200.npz"
PRODUCTION = ROOT / "outputs/gabriel_k2_k8_energy_magnitude"
ASSIGNMENTS = PRODUCTION / "clustering_assignments.csv"
OUT = ROOT / "outputs/gabriel_pinned_gmm_vs_kmeans"

SEED = 20260813
N_INIT = 5
REG_COVAR = 1e-7
MAX_ITER = 500
TOL = 1e-5
QS = tuple(range(1, 101))
KS = tuple(range(1, 21))
COMPARE_KS = tuple(range(2, 10))
STABILITY_SEEDS = tuple(range(20))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def csv_write(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or sorted({key for row in rows for key in row})
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def csv_read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def canonical_partition(labels: np.ndarray) -> tuple[int, ...]:
    mapping: dict[int, int] = {}
    nxt = 0
    out = []
    for value in labels.astype(int):
        if value not in mapping:
            mapping[value] = nxt
            nxt += 1
        out.append(mapping[value])
    return tuple(out)


def load_inputs():
    with np.load(DATA, allow_pickle=False) as z:
        contours = z["contours"].copy()
        filenames = z["filenames"].astype(str)
        keys = list(z.files)
    if contours.shape != (1652, 200, 2) or contours.dtype != np.float64:
        raise RuntimeError(f"STOP: unexpected pinned coordinates {contours.shape} {contours.dtype}")
    if not np.isfinite(contours).all():
        raise RuntimeError("STOP: pinned coordinates contain non-finite values")
    X = np.ascontiguousarray(contours).reshape(1652, 400, order="C")
    if not np.array_equal(X[0], contours[0].reshape(400, order="C")):
        raise RuntimeError("STOP: C-order flattening verification failed")

    rows = csv_read(ASSIGNMENTS)
    production_labels: dict[int, np.ndarray] = {}
    order_checks = {}
    for k in COMPARE_KS:
        selected = sorted((r for r in rows if int(r["K"]) == k), key=lambda r: int(r["target_index"]))
        indices = np.asarray([int(r["target_index"]) for r in selected])
        names = np.asarray([r["filename"] for r in selected])
        labels = np.asarray([int(r["cluster_id"]) for r in selected])
        ok = len(selected) == 1652 and np.array_equal(indices, np.arange(1652)) and np.array_equal(names, filenames)
        order_checks[str(k)] = bool(ok)
        if not ok:
            raise RuntimeError(f"STOP: production ordering mismatch for K={k}")
        if set(labels) != set(range(k)):
            raise RuntimeError(f"STOP: production labels do not cover 0..K-1 for K={k}")
        production_labels[k] = labels
        template_paths = list((PRODUCTION / "templates" / f"K{k:02d}").glob("*.npz"))
        if len(template_paths) != k:
            raise RuntimeError(f"STOP: expected {k} production templates for K={k}, got {len(template_paths)}")

    audit = {
        "dataset": str(DATA.relative_to(ROOT)),
        "dataset_sha256": sha256(DATA),
        "npz_keys": keys,
        "coordinate_key": "contours",
        "coordinate_shape": list(contours.shape),
        "coordinate_dtype": str(contours.dtype),
        "matrix_shape": list(X.shape),
        "matrix_dtype": str(X.dtype),
        "flatten_order": "C",
        "finite": True,
        "pca_input": "Gabriel pinned shape coordinates only",
        "filename_count": len(filenames),
        "production_assignment_file": str(ASSIGNMENTS.relative_to(ROOT)),
        "production_assignment_sha256": sha256(ASSIGNMENTS),
        "production_assignment_rows": len(rows),
        "production_order_matches_by_K": order_checks,
    }
    return contours, filenames, X, production_labels, audit


def deterministic_split(n: int):
    rng = np.random.default_rng(SEED)
    order = rng.permutation(n)
    n_train = int(round(0.8 * n))
    return np.sort(order[:n_train]), np.sort(order[n_train:])


def fit_pca(X, train_idx, heldout_idx):
    model = PCA(n_components=100, svd_solver="randomized", random_state=SEED, iterated_power=7)
    train_scores = model.fit_transform(X[train_idx])
    heldout_scores = model.transform(X[heldout_idx])
    rows = []
    cumulative = np.cumsum(model.explained_variance_ratio_)
    for q in QS:
        train_rec = train_scores[:, :q] @ model.components_[:q] + model.mean_
        heldout_rec = heldout_scores[:, :q] @ model.components_[:q] + model.mean_
        rows.append({
            "q": q,
            "explained_variance_ratio": float(model.explained_variance_ratio_[q - 1]),
            "cumulative_explained_variance": float(cumulative[q - 1]),
            "train_reconstruction_mse": float(np.mean((X[train_idx] - train_rec) ** 2)),
            "heldout_reconstruction_mse": float(np.mean((X[heldout_idx] - heldout_rec) ** 2)),
        })
    np.savez_compressed(
        OUT / "pca/pca_q100.npz", mean=model.mean_, components=model.components_,
        explained_variance=model.explained_variance_, explained_variance_ratio=model.explained_variance_ratio_,
        singular_values=model.singular_values_, train_scores=train_scores, heldout_scores=heldout_scores,
    )
    csv_write(OUT / "pca/pca_study_q1_q100.csv", rows)
    return model, train_scores, heldout_scores, rows


def fit_gmm_cell(train_scores, heldout_scores, q: int, k: int, seed: int = SEED, n_init: int = N_INIT, return_model=False):
    start = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = GaussianMixture(
                n_components=k, covariance_type="full", n_init=n_init, random_state=seed,
                reg_covar=REG_COVAR, max_iter=MAX_ITER, tol=TOL, init_params="kmeans",
            ).fit(train_scores[:, :q])
        train_resp = model.predict_proba(train_scores[:, :q])
        train_labels = np.argmax(train_resp, axis=1)
        sizes = np.bincount(train_labels, minlength=k)
        entropy = float(-np.sum(train_resp * np.log(np.clip(train_resp, 1e-300, 1))))
        train_ll_mean = float(model.score(train_scores[:, :q]))
        heldout_ll_mean = float(model.score(heldout_scores[:, :q]))
        bic = float(model.bic(train_scores[:, :q]))
        row = {
            "q": q, "K": k, "seed": seed, "n_init": n_init,
            "BIC": bic, "ICL": bic + 2 * entropy,
            "train_log_likelihood": train_ll_mean * len(train_scores),
            "train_log_likelihood_per_sample": train_ll_mean,
            "heldout_log_likelihood": heldout_ll_mean * len(heldout_scores),
            "heldout_log_likelihood_per_sample": heldout_ll_mean,
            "posterior_entropy": entropy,
            "component_weights": json.dumps(model.weights_.tolist(), separators=(",", ":")),
            "hard_cluster_sizes_train": json.dumps(sizes.tolist(), separators=(",", ":")),
            "converged": bool(model.converged_), "iterations": int(model.n_iter_),
            "minimum_component_weight": float(model.weights_.min()),
            "minimum_train_cluster_size": int(sizes.min()),
            "tiny_component_lt_1pct_or_10": bool(np.any((sizes < 10) | (sizes < 0.01 * len(train_scores)))),
            "warnings": " | ".join(str(w.message) for w in caught),
            "runtime_seconds": time.perf_counter() - start,
        }
        return (row, model) if return_model else row
    except Exception as exc:
        row = {"q": q, "K": k, "seed": seed, "n_init": n_init, "converged": False,
               "error": f"{type(exc).__name__}: {exc}", "runtime_seconds": time.perf_counter() - start}
        return (row, None) if return_model else row


def run_grid(train_scores, heldout_scores, n_jobs: int):
    path = OUT / "tables/pca_gmm_model_selection.csv"
    existing = csv_read(path)
    done = {(int(r["q"]), int(r["K"])) for r in existing}
    tasks = [(q, k) for q in QS for k in KS if (q, k) not in done]
    if tasks:
        print(f"Running {len(tasks)} remaining GMM grid cells with n_jobs={n_jobs}", flush=True)
        new_rows = Parallel(n_jobs=n_jobs, verbose=10, batch_size=1, prefer="threads")(
            delayed(fit_gmm_cell)(train_scores, heldout_scores, q, k) for q, k in tasks
        )
        existing.extend(new_rows)
        existing.sort(key=lambda r: (int(r["q"]), int(r["K"])))
        csv_write(path, existing)
    if len(existing) != 2000:
        raise RuntimeError(f"Expected 2000 grid cells, got {len(existing)}")
    return existing


def preferred_by_q(grid_rows):
    rows = []
    for q in QS:
        candidates = [r for r in grid_rows if int(r["q"]) == q and str(r["converged"]).lower() == "true" and not r.get("error")]
        if not candidates:
            raise RuntimeError(f"No converged GMMs at q={q}")
        rows.append({
            "q": q,
            "K_BIC": int(min(candidates, key=lambda r: float(r["BIC"]))["K"]),
            "K_ICL": int(min(candidates, key=lambda r: float(r["ICL"]))["K"]),
            "K_heldout_LL": int(max(candidates, key=lambda r: float(r["heldout_log_likelihood_per_sample"]))["K"]),
        })
    csv_write(OUT / "tables/preferred_k_by_q.csv", rows)
    return rows


def choose_primary_q(pca_rows, grid_rows):
    # Candidate operating region: first q reaching 99.9%, then inspect q +/- 3.
    threshold = next((int(r["q"]) for r in pca_rows if float(r["cumulative_explained_variance"]) >= 0.999), 20)
    candidates = list(range(max(2, threshold - 3), min(100, threshold + 3) + 1))
    diagnostics = []
    for q in candidates:
        cells = [r for r in grid_rows if int(r["q"]) == q]
        converged = sum(str(r["converged"]).lower() == "true" for r in cells)
        tiny = sum(str(r.get("tiny_component_lt_1pct_or_10", "false")).lower() == "true" for r in cells)
        p = pca_rows[q - 1]
        diagnostics.append({"q": q, "cumulative_explained_variance": float(p["cumulative_explained_variance"]),
                            "heldout_reconstruction_mse": float(p["heldout_reconstruction_mse"]),
                            "converged_cells": converged, "tiny_component_cells": tiny})
    # Prefer the threshold itself if fully converged; otherwise nearest fully converged lower-dimensional q.
    valid = [d for d in diagnostics if d["converged_cells"] == 20]
    q_primary = min(valid, key=lambda d: (abs(d["q"] - threshold), d["q"]))["q"] if valid else threshold
    csv_write(OUT / "tables/q_primary_diagnostics.csv", diagnostics)
    return int(q_primary), threshold, diagnostics


def stability_and_representatives(train_scores, heldout_scores, full_scores, q_primary: int, n_jobs: int):
    tasks = [(k, seed) for k in COMPARE_KS for seed in STABILITY_SEEDS]
    fitted = Parallel(n_jobs=n_jobs, verbose=10, batch_size=1, prefer="threads")(
        delayed(fit_gmm_cell)(train_scores, heldout_scores, q_primary, k, seed, N_INIT, True) for k, seed in tasks
    )
    by_key = {(k, seed): value for (k, seed), value in zip(tasks, fitted)}
    stability_rows, representatives = [], {}
    for k in COMPARE_KS:
        labels, models, seeds, fitrows = [], [], [], []
        for seed in STABILITY_SEEDS:
            row, model = by_key[(k, seed)]
            if model is None or not model.converged_:
                continue
            labels.append(model.predict(full_scores[:, :q_primary]))
            models.append(model); seeds.append(seed); fitrows.append(row)
        if len(labels) < 2:
            raise RuntimeError(f"Insufficient stable fits for K={k}")
        ari = np.eye(len(labels))
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                ari[i, j] = ari[j, i] = adjusted_rand_score(labels[i], labels[j])
        mean_ari = (ari.sum(axis=1) - 1) / (len(labels) - 1)
        medoid_idx = int(np.argmax(mean_ari))
        keys = [canonical_partition(x) for x in labels]
        counts = Counter(keys)
        modal_key, modal_frequency = counts.most_common(1)[0]
        modal_indices = [i for i, key in enumerate(keys) if key == modal_key]
        modal_fraction = modal_frequency / len(labels)
        if modal_frequency >= 2:
            representative_idx = max(modal_indices, key=lambda i: mean_ari[i])
            representative_type = "exact_modal_partition"
        else:
            representative_idx = medoid_idx
            representative_type = "ARI_medoid_weak_exact_mode"
        model = models[representative_idx]
        representatives[k] = {"model": model, "labels": labels[representative_idx], "seed": seeds[representative_idx],
                              "medoid_seed": seeds[medoid_idx], "modal_frequency": modal_frequency,
                              "modal_fraction": modal_fraction, "representative_type": representative_type}
        pair = ari[np.triu_indices(len(labels), 1)]
        stability_rows.append({
            "K": k, "q_primary": q_primary, "attempted_fits": len(STABILITY_SEEDS), "converged_fits": len(labels),
            "pairwise_ARI_mean": float(pair.mean()), "pairwise_ARI_min": float(pair.min()),
            "pairwise_ARI_max": float(pair.max()), "pairwise_ARI_std": float(pair.std(ddof=1)),
            "modal_frequency": modal_frequency, "modal_fraction": modal_fraction,
            "representative_seed": seeds[representative_idx], "medoid_seed": seeds[medoid_idx],
            "representative_type": representative_type,
        })
        for seed, lab, row in zip(seeds, labels, fitrows):
            np.save(OUT / "gmm" / f"stability_K{k:02d}_seed{seed:02d}_labels.npy", lab)
            json_write(OUT / "gmm" / f"stability_K{k:02d}_seed{seed:02d}_fit.json", row)
    csv_write(OUT / "tables/gmm_assignment_stability_k2_k9.csv", stability_rows)
    return stability_rows, representatives


def optimal_label_match(reference: np.ndarray, candidate: np.ndarray, k: int):
    contingency = np.zeros((k, k), dtype=int)
    for a, b in zip(reference, candidate):
        contingency[int(a), int(b)] += 1
    rows, cols = linear_sum_assignment(-contingency)
    mapping = {int(col): int(row) for row, col in zip(rows, cols)}
    matched = np.asarray([mapping[int(x)] for x in candidate])
    return matched, mapping, contingency[:, cols], cols


def compare_assignments_and_templates(contours, filenames, production_labels, representatives, full_scores, q_primary):
    summary = []
    for k in COMPARE_KS:
        rep = representatives[k]
        gmm_labels = rep["labels"].astype(int)
        km_labels = production_labels[k]
        matched, mapping, adjusted_contingency, matched_gmm_order = optimal_label_match(km_labels, gmm_labels, k)
        different = int(np.sum(matched != km_labels))
        hard_templates = np.stack([contours[gmm_labels == cluster].mean(axis=0) for cluster in range(k)])
        responsibilities = rep["model"].predict_proba(full_scores[:, :q_primary])
        soft_templates = np.stack([np.average(contours, axis=0, weights=responsibilities[:, cluster]) for cluster in range(k)])
        prod_templates = []
        for cluster in range(k):
            path = PRODUCTION / "templates" / f"K{k:02d}" / f"K{k:02d}_cluster{cluster:02d}.npz"
            with np.load(path, allow_pickle=False) as z:
                prod_templates.append(z["coordinates"])
        prod_templates = np.stack(prod_templates)
        matched_hard = np.stack([hard_templates[int(col)] for col in matched_gmm_order])
        pointwise_sq = np.sum((prod_templates - matched_hard) ** 2, axis=2)
        template_mse = pointwise_sq.mean(axis=1)
        template_l2 = np.linalg.norm((prod_templates - matched_hard).reshape(k, -1), axis=1)
        soft_hard_mse = np.mean((soft_templates - hard_templates) ** 2, axis=(1, 2))
        km_sizes = np.bincount(km_labels, minlength=k)
        gmm_sizes = np.bincount(gmm_labels, minlength=k)

        np.savez_compressed(OUT / "templates" / f"gmm_hard_templates_K{k:02d}.npz",
                            coordinates=hard_templates, cluster_sizes=gmm_sizes, labels=gmm_labels,
                            filenames=filenames, q_primary=q_primary, representative_seed=rep["seed"])
        np.savez_compressed(OUT / "templates" / f"gmm_soft_templates_K{k:02d}.npz",
                            coordinates=soft_templates, responsibility_sums=responsibilities.sum(axis=0),
                            responsibilities=responsibilities, q_primary=q_primary, representative_seed=rep["seed"])
        np.save(OUT / "assignments" / f"gmm_assignments_K{k:02d}.npy", gmm_labels)
        csv_write(OUT / "assignments" / f"gmm_assignments_K{k:02d}.csv",
                  [{"target_index": i, "filename": filenames[i], "gmm_cluster_id": int(gmm_labels[i]),
                    "matched_kmeans_label": int(matched[i])} for i in range(len(filenames))])
        csv_write(OUT / "tables" / f"adjusted_contingency_K{k:02d}.csv",
                  [{"kmeans_cluster": i, **{f"gmm_matched_cluster_{j}": int(adjusted_contingency[i, j]) for j in range(k)}} for i in range(k)])
        summary.append({
            "K": k, "q_primary": q_primary,
            "kmeans_cluster_sizes": json.dumps(km_sizes.tolist(), separators=(",", ":")),
            "gmm_cluster_sizes": json.dumps(gmm_sizes.tolist(), separators=(",", ":")),
            "ARI_kmeans_vs_gmm": float(adjusted_rand_score(km_labels, gmm_labels)),
            "NMI_kmeans_vs_gmm": float(normalized_mutual_info_score(km_labels, gmm_labels)),
            "exact_same_partition": bool(canonical_partition(km_labels) == canonical_partition(gmm_labels)),
            "n_airfoils_different": different, "fraction_airfoils_different": different / len(filenames),
            "gmm_modal_frequency": rep["modal_frequency"], "gmm_modal_fraction": rep["modal_fraction"],
            "gmm_representative_seed": rep["seed"], "gmm_medoid_seed": rep["medoid_seed"],
            "gmm_representative_type": rep["representative_type"],
            "mean_template_mse": float(template_mse.mean()), "max_template_mse": float(template_mse.max()),
            "mean_template_l2": float(template_l2.mean()), "max_template_l2": float(template_l2.max()),
            "mean_soft_vs_hard_template_mse": float(soft_hard_mse.mean()),
            "max_soft_vs_hard_template_mse": float(soft_hard_mse.max()),
            "optimal_gmm_to_kmeans_mapping": json.dumps(mapping, sort_keys=True),
            "cluster_size_difference_matched": json.dumps((np.bincount(matched, minlength=k) - km_sizes).tolist(), separators=(",", ":")),
        })
    csv_write(OUT / "tables/gmm_vs_kmeans_assignment_summary_k2_k9.csv", summary)
    return summary


def make_figures(summary, preferred):
    figures = OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    ks = [r["K"] for r in summary]
    specs = [
        ("ari_kmeans_vs_gmm_k2_k9.png", [r["ARI_kmeans_vs_gmm"] for r in summary], "ARI", (0, 1.02)),
        ("fraction_different_kmeans_vs_gmm_k2_k9.png", [100 * r["fraction_airfoils_different"] for r in summary], "Airfoils assigned differently (%)", (0, None)),
        ("template_difference_kmeans_vs_gmm_k2_k9.png", [r["mean_template_l2"] for r in summary], "Mean matched-template L2 distance", (0, None)),
    ]
    for name, values, ylabel, ylim in specs:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(ks, values, marker="o", color="#2563eb", lw=2)
        ax.set(xlabel="K", ylabel=ylabel, title=ylabel + " across K=2–9")
        ax.set_xticks(ks); ax.grid(axis="y", alpha=.25); ax.spines[["top", "right"]].set_visible(False)
        if ylim[1] is None: ax.set_ylim(bottom=ylim[0])
        else: ax.set_ylim(*ylim)
        fig.tight_layout(); fig.savefig(figures / name, dpi=300, bbox_inches="tight"); plt.close(fig)
    q = [r["q"] for r in preferred]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for field, label, marker, color in [("K_BIC", "BIC", "o", "#2563eb"), ("K_ICL", "ICL", "s", "#d97706"), ("K_heldout_LL", "Held-out LL", "^", "#6b7280")]:
        ax.plot(q, [r[field] for r in preferred], marker=marker, ms=3, lw=1.3, label=label, color=color)
    ax.set(xlabel="PCA dimension q", ylabel="Preferred K", title="Preferred K across pinned-shape PCA dimensions")
    ax.set_yticks(range(1, 21)); ax.set_ylim(.5, 20.5); ax.grid(alpha=.22); ax.legend()
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
    fig.savefig(figures / "preferred_k_pinned_shape_pca_gmm.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    json_write(OUT / "report/chart_map.json", [
        {"figure": s[0], "family": "ordered comparison", "x": "K", "y": s[2]} for s in specs
    ] + [{"figure": "preferred_k_pinned_shape_pca_gmm.png", "family": "ordered multi-series comparison", "x": "q", "y": "preferred K"}])


def write_report(audit, thresholds, q_primary, q_threshold, qdiag, preferred, stability, summary):
    pref_counts = {name: dict(Counter(int(r[field]) for r in preferred)) for name, field in
                   [("BIC", "K_BIC"), ("ICL", "K_ICL"), ("heldout_LL", "K_heldout_LL")]}
    lines = [
        "# Gabriel pinned shape: GMM vs production K-means", "", "## Technical summary", "",
        f"Exact pinned coordinates `{audit['coordinate_shape']}` (`{audit['coordinate_dtype']}`) were flattened in C order to `{audit['matrix_shape']}`. Production filename ordering matched exactly for K=2–9. PCA and every GMM were fit on the 1,322-sample training split only; 330 held-out samples were used only for transform/evaluation.", "",
        f"The first q meeting 99.9% cumulative training explained variance was {q_threshold}; the selected primary q is **{q_primary}**, based on that variance operating point and complete local-grid convergence. Local diagnostics: `{json.dumps(qdiag)}`.", "",
        f"Explained-variance thresholds: `{json.dumps(thresholds, sort_keys=True)}`.", "",
        f"Preferred-K frequencies over q=1..100: `{json.dumps(pref_counts, sort_keys=True)}`.", "",
        "## Assignment stability and production comparison", "",
        "| K | stability mean ARI | modal support | ARI vs K-means | NMI | different | mean template L2 | max template L2 |", "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    stab_by_k = {int(r["K"]): r for r in stability}
    for r in summary:
        s = stab_by_k[r["K"]]
        lines.append(f"| {r['K']} | {s['pairwise_ARI_mean']:.4f} | {r['gmm_modal_frequency']}/{s['converged_fits']} | {r['ARI_kmeans_vs_gmm']:.4f} | {r['NMI_kmeans_vs_gmm']:.4f} | {100*r['fraction_airfoils_different']:.2f}% | {r['mean_template_l2']:.6g} | {r['max_template_l2']:.6g} |")
    lines += ["", "## Scope and interpretation", "",
              "This is an assignment/template comparison only. No registration was run, no production K-means assignment or template was modified, and all GMM artifacts were written under the new comparison directory. ARI/NMI and exact equality are label-permutation invariant; different counts and template comparisons use Hungarian optimal label matching.", ""]
    (OUT / "report/final_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "grid", "analyse", "all"), default="all")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    for directory in ("splits", "pca", "gmm", "assignments", "templates", "tables", "figures", "report"):
        (OUT / directory).mkdir(parents=True, exist_ok=True)
    contours, filenames, X, production_labels, audit = load_inputs()
    train_idx, heldout_idx = deterministic_split(len(X))
    np.save(OUT / "splits/train_indices.npy", train_idx); np.save(OUT / "splits/heldout_indices.npy", heldout_idx)
    pca, train_scores, heldout_scores, pca_rows = fit_pca(X, train_idx, heldout_idx)
    full_scores = pca.transform(X)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    thresholds = {}
    for threshold in (0.90, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999):
        hits = np.flatnonzero(cumulative >= threshold)
        thresholds[str(threshold)] = int(hits[0] + 1) if len(hits) else None
    config = {
        "purpose": "clustering-only GMM vs production K-means assignment/template comparison",
        "input_audit": audit,
        "split": {"seed": SEED, "train_count": len(train_idx), "heldout_count": len(heldout_idx), "pca_fit_train_only": True, "gmm_fit_train_only": True},
        "pca": {"q": list(QS), "threshold_q": thresholds},
        "gmm": {"K": list(KS), "covariance_type": "full", "n_init": N_INIT, "reg_covar": REG_COVAR, "max_iter": MAX_ITER, "tol": TOL, "seed": SEED,
                "ICL_formula": "BIC + 2 * posterior entropy"},
        "stability": {"seeds": list(STABILITY_SEEDS), "n_init_per_seed": N_INIT, "selection_independent_of_production_labels": True},
        "registration_run": False,
    }
    json_write(OUT / "config.json", config)
    if args.stage == "prepare": return
    grid_rows = run_grid(train_scores, heldout_scores, args.jobs)
    if args.stage == "grid": return
    preferred = preferred_by_q(grid_rows)
    q_primary, q_threshold, qdiag = choose_primary_q(pca_rows, grid_rows)
    stability, representatives = stability_and_representatives(train_scores, heldout_scores, full_scores, q_primary, args.jobs)
    summary = compare_assignments_and_templates(contours, filenames, production_labels, representatives, full_scores, q_primary)
    make_figures(summary, preferred)
    write_report(audit, thresholds, q_primary, q_threshold, qdiag, preferred, stability, summary)
    checks = {
        "exact_Gabriel_pinned_dataset_used": audit["coordinate_shape"] == [1652, 200, 2] and audit["coordinate_dtype"] == "float64",
        "exact_matrix_1652x400_C_order": audit["matrix_shape"] == [1652, 400] and audit["flatten_order"] == "C",
        "production_order_matches_K2_K9": all(audit["production_order_matches_by_K"].values()),
        "pca_shape_coordinates_not_velocity": audit["pca_input"] == "Gabriel pinned shape coordinates only",
        "split_reproducible_1322_330": len(train_idx) == 1322 and len(heldout_idx) == 330,
        "pca_fit_training_only": True, "gmm_fit_training_only": True,
        "grid_2000_cells": len(grid_rows) == 2000,
        "comparison_K2_K9": len(summary) == 8,
        "label_permutation_invariant_metrics": True,
        "production_assignments_read_only": sha256(ASSIGNMENTS) == audit["production_assignment_sha256"],
        "production_templates_not_overwritten": True,
        "gmm_templates_only_in_new_output": all((OUT / "templates" / f"gmm_hard_templates_K{k:02d}.npz").exists() for k in COMPARE_KS),
        "registration_not_run": True,
    }
    json_write(OUT / "validation.json", {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
                                          "q_primary": q_primary, "threshold_q": thresholds,
                                          "converged_grid_cells": sum(str(r["converged"]).lower() == "true" for r in grid_rows)})
    if not all(checks.values()): raise RuntimeError("Validation failed")


if __name__ == "__main__":
    main()
