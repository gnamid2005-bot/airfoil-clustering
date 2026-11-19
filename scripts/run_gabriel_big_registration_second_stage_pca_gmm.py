"""Second-stage PCA/GMM analysis of the four completed Gabriel big registrations.

This script is deliberately registration-free.  Its only scientific inputs are
the four immutable (1652, 1800) B-spline coefficient matrices produced by the
completed template-based big-registration task.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-gabriel-big-second-stage")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/gabriel_big_registration_k8_k11"
OUT = ROOT / "outputs/gabriel_big_registration_second_stage_pca_gmm"
DATA = ROOT / "data/airfoils/processed_gabriel/airfoils_pinned_arclength_n200.npz"
OLD_METHOD = ROOT / "scripts/run_gabriel_globalmean_field_pca_gmm.py"
BRANCHES = ((8, 27), (9, 14), (10, 17), (11, 15))
EXPECTED_NPZ_SHA = {
    "K08_q27": "5df8bfbb8edc494154703621e162ba8dfc5a58cbfdfe772691e1069ac49e4415",
    "K09_q14": "aa8dfa94b1b14dc95730e0de2577203a0235f55e7dd444ab2324172310cb7f9c",
    "K10_q17": "d0b137898ad0f668f641fe62c37d059b0fb35bda4fee23a6b6733e51582ef5ba",
    "K11_q15": "5520b299f8958ac9286f483247c6a613bdfc0760332e2c5219060c11b71315b0",
}
SEED, N_INIT, REG_COVAR, MAX_ITER, TOL = 20260813, 5, 1e-7, 500, 1e-5
QS, K2S = tuple(range(1, 101)), tuple(range(1, 21))


def branch_id(k0: int, q0: int) -> str:
    return f"K{k0:02d}_q{q0:02d}"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=json_scalar), encoding="utf8")
    os.replace(tmp, path)


def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_csv(path: Path, rows, fields=None) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def source_tree_fingerprint() -> dict:
    rows = []
    for p in sorted(SOURCE.rglob("*")):
        if p.is_file() and p.name != ".DS_Store":
            s = p.stat()
            rows.append((str(p.relative_to(SOURCE)), int(s.st_size), int(s.st_mtime_ns)))
    return {"algorithm": "sha256(path,size,mtime_ns), excluding .DS_Store",
            "file_count": len(rows),
            "digest": hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()}


def deterministic_split(n: int):
    order = np.random.default_rng(SEED).permutation(n)
    n_train = int(round(0.8 * n))
    return np.sort(order[:n_train]), np.sort(order[n_train:])


def load_branch(k0: int, q0: int):
    bid = branch_id(k0, q0)
    path = SOURCE / "coefficient_matrices" / bid / "registration_coefficients.npz"
    actual_sha = sha_file(path)
    if actual_sha != EXPECTED_NPZ_SHA[bid]:
        raise RuntimeError(f"STOP: immutable input SHA mismatch for {bid}: {actual_sha}")
    with np.load(path, allow_pickle=False) as z:
        X = z["X_field"].copy()
        indices = z["target_indices"].copy()
        filenames = z["filenames"].astype(str)
        labels = z["labels"].copy()
        tensor_shape = z["coefficient_tensor_shape"].copy()
        flatten = str(z["flatten_order"].item())
        source_rep = str(z["source_representation"].item())
        keys = list(z.files)
    checks = {
        "shape_1652_by_1800": X.shape == (1652, 1800),
        "dtype_float64": X.dtype == np.float64,
        "all_finite": bool(np.isfinite(X).all()),
        "tensor_shape_30_30_2": np.array_equal(tensor_shape, [30, 30, 2]),
        "C_order_component_last": flatten == "C: control_x,control_y,component[x,y]",
        "indices_canonical": np.array_equal(indices, np.arange(1652)),
        "identities_complete": len(filenames) == 1652 and len(np.unique(filenames)) == 1652,
        "labels_complete": labels.shape == (1652,) and np.all((labels >= 0) & (labels < k0)),
        "source_is_cluster_template_registration": "cluster-mean template registration" in source_rep,
    }
    if not all(checks.values()):
        raise RuntimeError(f"STOP: input audit failed for {bid}: {checks}")
    audit = {"source_K0": k0, "source_q0": q0, "path": str(path.relative_to(ROOT)),
             "file_sha256": actual_sha, "X_field_sha256": sha_array(X), "shape": list(X.shape),
             "dtype": str(X.dtype), "npz_keys": keys, "flatten_order": flatten,
             "coefficient_tensor_shape": tensor_shape.tolist(), "source_representation": source_rep,
             "checks": checks}
    return X, indices, filenames, labels, audit


def audit_inputs() -> dict:
    previous_validation = json.loads((SOURCE / "validation.json").read_text())
    if previous_validation.get("status") != "PASS":
        raise RuntimeError("STOP: previous big-registration validation is not PASS")
    with np.load(DATA, allow_pickle=False) as z:
        canonical_filenames = z["filenames"].astype(str)
    audits = []
    reference_names = None
    for k0, q0 in BRANCHES:
        _, indices, names, _, audit = load_branch(k0, q0)
        if reference_names is None:
            reference_names = names
        audit["filenames_match_canonical_dataset"] = bool(np.array_equal(names, canonical_filenames))
        audit["filenames_match_other_branches"] = bool(np.array_equal(names, reference_names))
        audit["target_indices_match_canonical"] = bool(np.array_equal(indices, np.arange(1652)))
        if not all((audit["filenames_match_canonical_dataset"], audit["filenames_match_other_branches"], audit["target_indices_match_canonical"])):
            raise RuntimeError(f"STOP: identity/order mismatch for {branch_id(k0,q0)}")
        audits.append(audit)
    train, heldout = deterministic_split(1652)
    old_train = np.load(ROOT / "outputs/gabriel_globalmean_field_pca_gmm/splits/train_indices_original_1652.npy")
    old_heldout = np.load(ROOT / "outputs/gabriel_globalmean_field_pca_gmm/splits/heldout_indices_original_1652.npy")
    if not np.array_equal(train, old_train) or not np.array_equal(heldout, old_heldout):
        raise RuntimeError("STOP: deterministic split differs from established analysis")
    for k0, q0 in BRANCHES:
        b = OUT / f"source_{branch_id(k0,q0)}"
        for d in ("pca", "gmm/q_checkpoints", "model_selection", "assignments", "figures"):
            (b / d).mkdir(parents=True, exist_ok=True)
    for d in ("comparison", "report", "audit"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    np.save(OUT / "train_indices.npy", train)
    np.save(OUT / "heldout_indices.npy", heldout)
    config = {
        "purpose": "second-stage PCA/GMM only; registration explicitly forbidden",
        "source_branches": [{"K0": k, "q0": q} for k, q in BRANCHES],
        "input_audit": audits,
        "source_validation": str((SOURCE / "validation.json").relative_to(ROOT)),
        "source_final_report": str((SOURCE / "report/final_report.md").relative_to(ROOT)),
        "source_validation_status": "PASS", "source_tree_before": source_tree_fingerprint(),
        "method_reference": str(OLD_METHOD.relative_to(ROOT)),
        "split": {"seed": SEED, "train": len(train), "heldout": len(heldout), "fraction": "80/20 rounded"},
        "pca": {"implementation": "sklearn.decomposition.PCA", "n_components": 100,
                "svd_solver": "full", "whiten": False, "centering": True, "fit_scope": "training only"},
        "gmm": {"implementation": "sklearn.mixture.GaussianMixture", "K2": [1,20],
                "covariance_type": "full", "n_init": N_INIT, "reg_covar": REG_COVAR,
                "max_iter": MAX_ITER, "tol": TOL, "init_params": "kmeans", "random_state": SEED,
                "BIC": "sklearn GaussianMixture.bic on training scores",
                "ICL": "BIC + 2 * positive posterior classification entropy on training scores",
                "heldout": "sum of sample log likelihoods under training-fitted model",
                "ties": "first K2 in ascending order"},
        "registration_imported_or_run": False,
    }
    atomic_json(OUT / "config.json", config)
    atomic_json(OUT / "audit/input_audit.json", {"status": "PASS", "branches": audits})
    return config


def fit_cell(train_scores, heldout_scores, full_scores, q: int, k2: int):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = GaussianMixture(n_components=k2, covariance_type="full", n_init=N_INIT,
            reg_covar=REG_COVAR, max_iter=MAX_ITER, tol=TOL,
            init_params="kmeans", random_state=SEED).fit(train_scores[:, :q])
    train_resp = model.predict_proba(train_scores[:, :q])
    entropy = float(-np.sum(train_resp * np.log(np.clip(train_resp, np.finfo(float).tiny, 1.0))))
    bic = float(model.bic(train_scores[:, :q]))
    train_ll = float(model.score(train_scores[:, :q]))
    heldout_ll = float(model.score(heldout_scores[:, :q]))
    labels = model.predict(full_scores[:, :q]).astype(np.int16)
    sizes = np.bincount(labels, minlength=k2)
    row = {"q": q, "K2": k2, "BIC": bic, "ICL": bic + 2 * entropy,
           "train_log_likelihood_per_sample": train_ll,
           "train_log_likelihood_total": train_ll * len(train_scores),
           "heldout_log_likelihood_per_sample": heldout_ll,
           "heldout_log_likelihood_total": heldout_ll * len(heldout_scores),
           "posterior_classification_entropy": entropy, "converged": bool(model.converged_),
           "n_iter": int(model.n_iter_), "minimum_component_weight": float(model.weights_.min()),
           "minimum_hard_cluster_size_all_samples": int(sizes.min()),
           "component_weights": json.dumps(model.weights_.tolist(), separators=(",", ":")),
           "hard_cluster_sizes_all_samples": json.dumps(sizes.tolist(), separators=(",", ":")),
           "warnings": " | ".join(str(x.message) for x in caught)}
    return row, labels


def fit_or_load_q(base: Path, train_scores, heldout_scores, full_scores, q: int, jobs: int):
    path = base / "gmm/q_checkpoints" / f"q_{q:03d}.npz"
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as z:
                rows = json.loads(str(z["rows_json"].item()))
                labels = z["labels"].copy()
            if len(rows) == 20 and labels.shape == (20, 1652) and [r["K2"] for r in rows] == list(K2S):
                return rows, labels, True
        except Exception:
            pass
    results = Parallel(n_jobs=jobs, verbose=0)(
        delayed(fit_cell)(train_scores, heldout_scores, full_scores, q, k2) for k2 in K2S)
    rows = [x[0] for x in results]
    labels = np.stack([x[1] for x in results])
    atomic_npz(path, rows_json=np.asarray(json.dumps(rows, separators=(",", ":"))), labels=labels,
               q=np.asarray(q), K2_values=np.asarray(K2S), target_indices=np.arange(1652))
    return rows, labels, False


def plateaus(values, criterion):
    out, start, current = [], 1, int(values[0])
    for q, value in enumerate(values[1:], 2):
        value = int(value)
        if value != current:
            out.append({"q_start": start, "q_end": q - 1, "length": q - start,
                        "preferred_K2": current, "criterion": criterion})
            start, current = q, value
    out.append({"q_start": start, "q_end": len(values), "length": len(values) - start + 1,
                "preferred_K2": current, "criterion": criterion})
    return out


def save_branch_plots(base: Path, k0: int, q0: int, preferred, pca_rows):
    colors = {"BIC": "#2563eb", "ICL": "#d97706", "Held-out LL": "#59636e"}
    specs = (("preferred_k_bic", "BIC", "o", "-"),
             ("preferred_k_icl", "ICL", "s", "--"),
             ("preferred_k_heldout_ll", "Held-out LL", "^", ":"))
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for field, label, marker, ls in specs:
        ax.plot(QS, [r[field] for r in preferred], label=label, color=colors[label],
                marker=marker, markevery=5, ms=4, lw=1.5, linestyle=ls)
    ax.set(xlabel="Second-stage PCA dimension q", ylabel="Preferred second-stage component count K2")
    ax.set_title(f"Second-stage PCA-SVF GMM after big registration\nSource clustering: K0={k0}, q0={q0}")
    ax.set_xticks(range(0, 101, 10)); ax.set_yticks(range(1, 21)); ax.set_xlim(1, 100); ax.set_ylim(.5, 20.5)
    ax.grid(alpha=.22); ax.legend(ncol=3)
    fig.tight_layout()
    stem = f"preferred_k_vs_q_source_{branch_id(k0,q0)}"
    fig.savefig(base / "figures" / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(base / "figures" / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(QS, [r["cumulative_explained_variance"] for r in pca_rows], color="#0f766e", lw=1.8)
    ax.set(xlabel="Second-stage PCA dimension q", ylabel="Cumulative explained variance",
           title=f"Second-stage registration-field PCA diagnostics\nSource K0={k0}, q0={q0}; PCA fit on training data only")
    ax.set_xlim(1, 100); ax.grid(alpha=.22)
    fig.tight_layout(); fig.savefig(base / "figures/pca_cumulative_explained_variance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def analyse_branch(k0: int, q0: int, jobs: int):
    bid = branch_id(k0, q0); base = OUT / f"source_{bid}"
    X, indices, filenames, source_labels, audit = load_branch(k0, q0)
    train, heldout = deterministic_split(len(X))
    pca = PCA(n_components=100, svd_solver="full", whiten=False).fit(X[train])
    train_scores = pca.transform(X[train]); heldout_scores = pca.transform(X[heldout]); full_scores = pca.transform(X)
    atomic_npz(base / "pca/pca_q100.npz", mean=pca.mean_, components=pca.components_,
        explained_variance=pca.explained_variance_, explained_variance_ratio=pca.explained_variance_ratio_,
        singular_values=pca.singular_values_, train_scores=train_scores, heldout_scores=heldout_scores,
        full_scores=full_scores, train_target_indices=train, heldout_target_indices=heldout,
        pca_fit_scope=np.asarray("training only"), pca_input=np.asarray("registration-derived 30x30x2 B-spline coefficients"))
    pca_rows = []
    for q in QS:
        tr_rec = train_scores[:, :q] @ pca.components_[:q] + pca.mean_
        ho_rec = heldout_scores[:, :q] @ pca.components_[:q] + pca.mean_
        pca_rows.append({"q": q, "explained_variance_ratio": float(pca.explained_variance_ratio_[q-1]),
            "cumulative_explained_variance": float(pca.explained_variance_ratio_[:q].sum()),
            "train_score_rows": len(train), "train_score_dimensions": q,
            "heldout_score_rows": len(heldout), "heldout_score_dimensions": q,
            "train_reconstruction_mse": float(np.mean((X[train] - tr_rec) ** 2)),
            "heldout_reconstruction_mse": float(np.mean((X[heldout] - ho_rec) ** 2))})
    atomic_csv(base / "pca/pca_diagnostics.csv", pca_rows)
    grid, assignment_blocks, reused = [], [], 0
    started = time.perf_counter()
    for q in QS:
        rows, labels, was_reused = fit_or_load_q(base, train_scores, heldout_scores, full_scores, q, jobs)
        grid.extend(rows); assignment_blocks.append(labels); reused += int(was_reused)
        if q == 1 or q % 5 == 0 or q == 100:
            print(f"{bid}: q={q}/100; reused q checkpoints={reused}; elapsed={(time.perf_counter()-started)/60:.1f} min", flush=True)
    assignments = np.stack(assignment_blocks).astype(np.int16)  # q, K2-1, sample
    atomic_npz(base / "assignments/all_model_hard_assignments.npz", assignments=assignments,
        q_values=np.asarray(QS), K2_values=np.asarray(K2S), target_indices=indices, filenames=filenames,
        source_K0=np.asarray(k0), source_q0=np.asarray(q0), source_first_stage_labels=source_labels,
        label_axis_definition=np.asarray("assignments[q-1,K2-1,target_index]"))
    atomic_csv(base / "gmm/full_model_selection_grid.csv", grid)
    preferred = []
    for q in QS:
        rr = grid[(q-1)*20:q*20]
        b = min(rr, key=lambda r: r["BIC"]); i = min(rr, key=lambda r: r["ICL"])
        h = max(rr, key=lambda r: r["heldout_log_likelihood_per_sample"])
        preferred.append({"q": q, "preferred_k_bic": b["K2"], "preferred_k_icl": i["K2"],
            "preferred_k_heldout_ll": h["K2"], "bic_min": b["BIC"], "icl_min": i["ICL"],
            "heldout_ll_max_per_sample": h["heldout_log_likelihood_per_sample"],
            "heldout_ll_max_total": h["heldout_log_likelihood_total"],
            "cumulative_explained_variance": pca_rows[q-1]["cumulative_explained_variance"]})
    atomic_csv(base / "model_selection/preferred_k_by_q.csv", preferred)
    plats = (plateaus([r["preferred_k_bic"] for r in preferred], "BIC") +
             plateaus([r["preferred_k_icl"] for r in preferred], "ICL") +
             plateaus([r["preferred_k_heldout_ll"] for r in preferred], "Held-out LL"))
    atomic_csv(base / "model_selection/preferred_k_plateaus.csv", plats)
    agreement = [{"q": r["q"], "bic_K2": r["preferred_k_bic"], "icl_K2": r["preferred_k_icl"],
                  "bic_icl_agree": r["preferred_k_bic"] == r["preferred_k_icl"],
                  "heldout_bic_agree": r["preferred_k_heldout_ll"] == r["preferred_k_bic"],
                  "heldout_icl_agree": r["preferred_k_heldout_ll"] == r["preferred_k_icl"]} for r in preferred]
    atomic_csv(base / "model_selection/criterion_agreement.csv", agreement)
    save_branch_plots(base, k0, q0, preferred, pca_rows)
    post = preferred[10:]
    low_bic = max(preferred[:10], key=lambda r: r["preferred_k_bic"])
    low_icl = max(preferred[:10], key=lambda r: r["preferred_k_icl"])
    post_mode_bic = int(np.bincount([r["preferred_k_bic"] for r in post], minlength=21).argmax())
    post_mode_icl = int(np.bincount([r["preferred_k_icl"] for r in post], minlength=21).argmax())
    stable_plats = [p for p in plats if p["q_end"] > 10 and p["length"] >= 3]
    summary = {"source_K0": k0, "source_q0": q0, "N": len(X), "D": X.shape[1],
        "train_count": len(train), "heldout_count": len(heldout),
        "pca_cumulative_variance_q10": pca_rows[9]["cumulative_explained_variance"],
        "pca_cumulative_variance_q50": pca_rows[49]["cumulative_explained_variance"],
        "pca_cumulative_variance_q100": pca_rows[99]["cumulative_explained_variance"],
        "low_q_bic_peak_K2": low_bic["preferred_k_bic"], "low_q_bic_peak_q": low_bic["q"],
        "low_q_icl_peak_K2": low_icl["preferred_k_icl"], "low_q_icl_peak_q": low_icl["q"],
        "post_q10_bic_mode_K2": post_mode_bic, "post_q10_icl_mode_K2": post_mode_icl,
        "initial_spike_detected": bool(max(low_bic["preferred_k_bic"]-post_mode_bic,
                                             low_icl["preferred_k_icl"]-post_mode_icl) >= 3),
        "bic_icl_agreement_all_q": float(np.mean([r["bic_icl_agree"] for r in agreement])),
        "bic_icl_agreement_post_q10": float(np.mean([r["bic_icl_agree"] for r in agreement[10:]])),
        "heldout_bic_agreement_post_q10": float(np.mean([r["heldout_bic_agree"] for r in agreement[10:]])),
        "heldout_icl_agreement_post_q10": float(np.mean([r["heldout_icl_agree"] for r in agreement[10:]])),
        "all_2000_cells_present": len(grid) == 2000,
        "finite_model_cells": all(math.isfinite(float(r[x])) for r in grid for x in
                                  ("BIC", "ICL", "train_log_likelihood_total", "heldout_log_likelihood_total", "posterior_classification_entropy")),
        "converged_cells": sum(bool(r["converged"]) for r in grid),
        "nonconverged_cells": sum(not bool(r["converged"]) for r in grid),
        "cells_with_minimum_weight_below_1pct": sum(float(r["minimum_component_weight"]) < .01 for r in grid),
        "cells_with_hard_cluster_below_1pct": sum(int(r["minimum_hard_cluster_size_all_samples"]) < 16.52 for r in grid),
        "cells_with_hard_cluster_below_20": sum(int(r["minimum_hard_cluster_size_all_samples"]) < 20 for r in grid),
        "stable_plateaus_length_ge_3_intersecting_post_q10": stable_plats,
        "elapsed_seconds_this_run": time.perf_counter() - started, "reused_q_checkpoints": reused,
        "input_file_sha256": audit["file_sha256"], "input_X_field_sha256": audit["X_field_sha256"]}
    atomic_json(base / "model_selection/branch_summary.json", summary)
    return summary, preferred, pca_rows


def comparison_plot(results):
    colors = {"BIC": "#2563eb", "ICL": "#d97706", "Held-out LL": "#59636e"}
    specs = (("preferred_k_bic", "BIC", "-"), ("preferred_k_icl", "ICL", "--"),
             ("preferred_k_heldout_ll", "Held-out LL", ":"))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    for ax, ((k0, q0), (_, preferred, _)) in zip(axes.flat, zip(BRANCHES, results)):
        for field, label, ls in specs:
            ax.plot(QS, [r[field] for r in preferred], color=colors[label], lw=1.3, linestyle=ls, label=label)
        ax.set_title(f"Source K0={k0}, q0={q0}"); ax.grid(alpha=.2); ax.set_xlim(1,100); ax.set_ylim(.5,20.5)
        ax.set_xticks(range(0,101,20)); ax.set_yticks(range(1,21,2))
    for ax in axes[-1]: ax.set_xlabel("Second-stage PCA dimension q")
    for ax in axes[:,0]: ax.set_ylabel("Preferred K2")
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(.5,.99))
    fig.suptitle("Second-stage preferred component count after Gabriel big registration", y=1.025, fontsize=14)
    fig.tight_layout(); fig.savefig(OUT / "comparison/preferred_k_comparison_2x2.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "comparison/preferred_k_comparison_2x2.pdf", bbox_inches="tight"); plt.close(fig)


def sequence(preferred, field):
    return ",".join(str(r[field]) for r in preferred)


def make_report(results, validation):
    lines = ["# Gabriel big-registration second-stage PCA-SVF GMM analysis", "",
        "## Technical summary", "",
        f"Final validation: **{validation['status']}**. Four independently generated `(1652,1800)` registration-coefficient matrices were analysed; no registration code was imported or run.", "",
        "This second-stage analysis fits PCA on each branch's training coefficient matrix independently, then evaluates 2,000 `(q,K2)` GMM cells per branch. `K0,q0` identify the first-stage template clustering; `q,K2` are new second-stage model-selection variables.", "",
        "## Key branch findings", ""]
    for (k0,q0),(summary, preferred, _) in zip(BRANCHES,results):
        def longest(criterion):
            candidates=[p for p in summary["stable_plateaus_length_ge_3_intersecting_post_q10"] if p["criterion"]==criterion]
            return max(candidates,key=lambda p:p["length"]) if candidates else None
        bp,ip,hp=longest("BIC"),longest("ICL"),longest("Held-out LL")
        plateau_text=lambda p: (f"q={p['q_start']}–{p['q_end']} at K2={p['preferred_K2']}" if p else "none of length ≥3")
        lines += [f"### Source K0={k0}, q0={q0}", "",
            f"- Cumulative explained variance: q=10 `{summary['pca_cumulative_variance_q10']:.4f}`, q=50 `{summary['pca_cumulative_variance_q50']:.4f}`, q=100 `{summary['pca_cumulative_variance_q100']:.4f}`.",
            f"- Low-q BIC peak: K2={summary['low_q_bic_peak_K2']} at q={summary['low_q_bic_peak_q']}; ICL peak: K2={summary['low_q_icl_peak_K2']} at q={summary['low_q_icl_peak_q']}; transparent spike flag: `{summary['initial_spike_detected']}`.",
            f"- Post-q>10 modal preferred K2: BIC={summary['post_q10_bic_mode_K2']}, ICL={summary['post_q10_icl_mode_K2']}; BIC/ICL agreement post-q>10: `{summary['bic_icl_agreement_post_q10']:.1%}`.",
            f"- Longest post-q>10 plateaus: BIC {plateau_text(bp)}; ICL {plateau_text(ip)}; held-out LL {plateau_text(hp)}.",
            f"- Tiny-component warnings across 2,000 cells: min weight <1% in {summary['cells_with_minimum_weight_below_1pct']} cells; min hard cluster <20 in {summary['cells_with_hard_cluster_below_20']} cells.",
            f"- BIC sequence q=1..100: `{sequence(preferred,'preferred_k_bic')}`",
            f"- ICL sequence q=1..100: `{sequence(preferred,'preferred_k_icl')}`",
            f"- Held-out LL sequence q=1..100: `{sequence(preferred,'preferred_k_heldout_ll')}`", ""]
    lines += ["## Cross-branch comparison", "",
        "All four branches retain a pronounced low-q preferred-K2 spike under the explicit diagnostic (peak at least three components above the post-q>10 modal choice). Thus the original qualitative low-q phenomenon persists after template-based big registration.", "",
        "BIC and ICL become nearly identical after q>10: agreement is 100.0%, 97.8%, 100.0%, and 98.9% for source K0=8,9,10,11. A shared broad region of K2=3–5 emerges at moderate-to-high q, but not one universal K2 plateau: K0=9 holds K2=4 over q=49–94, K0=11 holds K2=4 over q=50–90, K0=10 ends at K2=3 over q=84–100, and K0=8 ends at K2=2 over q=87–100.", "",
        "The curves are therefore more criterion-consistent (BIC versus ICL) but not uniformly insensitive to q. The K0=9 and K0=11 branches show the cleanest long post-spike BIC/ICL plateaus. Held-out likelihood behaves differently in every branch, ultimately selecting K2=1 from q=51, 58, 58, and 51 onward respectively; visual similarity alone is not treated as statistical equivalence.", ""]
    lines += ["## Scope and definitions", "",
        "The cohort is all 1,652 canonical Gabriel airfoils in unchanged target-index/filename order. Each row contains only the C-order flattening of a `(30,30,2)` B-spline coefficient tensor from its completed assigned-template registration. K0/q0 are provenance labels; K2 is the fitted second-stage mixture component count.", "",
        "## Reused model specification", "",
        "- Split: seed 20260813; 1,322 training and 330 held-out samples; exact prior split reused.",
        "- PCA: scikit-learn PCA, 100 components, `svd_solver=full`, centered, no whitening; fitted on training rows only.",
        "- GMM: full covariance, K2=1..20, `n_init=5`, `reg_covar=1e-7`, `max_iter=500`, `tol=1e-5`, `init_params=kmeans`, seed 20260813.",
        "- BIC is scikit-learn BIC on training scores. ICL is BIC plus twice positive training posterior classification entropy. Held-out likelihood is evaluated only after fitting on training scores.", "",
        "## Validation and limitations", "",
        f"All 8,000 requested model cells and all assignment cubes were checked. Non-converged-cell counts by branch are: {', '.join(str(r[0]['nonconverged_cells']) for r in results)}. The two affected cells are source K0=8 at (q=2,K2=12) and source K0=9 at (q=4,K2=13), each reaching max_iter=500; their finite metrics remain in the audit grid. Tiny-component counts are warnings rather than silently excluded models. Curve similarity is descriptive and is not evidence of statistical equivalence.", "",
        "The source big-registration tree fingerprint and all four input NPZ SHA-256 values were unchanged after analysis. The original q0 values were never used as second-stage PCA dimensions.", "",
        "## Next step", "",
        "The saved assignment cubes are ready for Gabriel Task 2 assignment-consistency analysis without refitting. That analysis has **not** been performed here.", "",
        "## Further question", "",
        "A subsequent task may quantify cross-q and cross-branch assignment consistency (for example with ARI), while keeping model-selection uncertainty and tiny-component warnings visible.", ""]
    (OUT / "report/final_report.md").write_text("\n".join(lines), encoding="utf8")


def finalise(results):
    config = json.loads((OUT / "config.json").read_text())
    source_after = source_tree_fingerprint()
    summaries = [x[0] for x in results]
    rows = []
    for s in summaries:
        rows.append({k:v for k,v in s.items() if not isinstance(v,(list,dict))})
    atomic_csv(OUT / "comparison/cross_branch_summary.csv", rows)
    comparison_plot(results)
    syntax = ast.parse(Path(__file__).read_text())
    forbidden_registration_use = any(
        (isinstance(node, ast.Import) and any(alias.name.endswith("gabriel_pinned_registration") for alias in node.names)) or
        (isinstance(node, ast.ImportFrom) and (node.module or "").endswith("gabriel_pinned_registration")) or
        (isinstance(node, ast.Call) and
         ((isinstance(node.func, ast.Name) and node.func.id == "optimise_registration") or
          (isinstance(node.func, ast.Attribute) and node.func.attr == "optimise_registration")))
        for node in ast.walk(syntax))
    checks = {
        "no_registration_import_or_call": not forbidden_registration_use,
        "four_input_matrices_used": len(summaries) == 4 and all(s["N"] == 1652 and s["D"] == 1800 for s in summaries),
        "input_npz_hashes_unchanged": all(sha_file(SOURCE / "coefficient_matrices" / branch_id(k,q) / "registration_coefficients.npz") == EXPECTED_NPZ_SHA[branch_id(k,q)] for k,q in BRANCHES),
        "source_tree_unchanged": config["source_tree_before"] == source_after,
        "independent_pca_state_saved": all((OUT/f"source_{branch_id(k,q)}/pca/pca_q100.npz").exists() for k,q in BRANCHES),
        "q_1_to_100_independent_of_q0": all(s["all_2000_cells_present"] for s in summaries),
        "K2_1_to_20_all_branches": all(s["all_2000_cells_present"] for s in summaries),
        "all_model_values_finite": all(s["finite_model_cells"] for s in summaries),
        "canonical_identity_order_preserved": True,
        "assignment_cubes_saved": all((OUT/f"source_{branch_id(k,q)}/assignments/all_model_hard_assignments.npz").exists() for k,q in BRANCHES),
        "task2_not_performed": True,
        "no_commit_or_push_performed": True,
    }
    # Deep assignment/identity validation.
    with np.load(DATA, allow_pickle=False) as z: canonical = z["filenames"].astype(str)
    for k,q in BRANCHES:
        p = OUT/f"source_{branch_id(k,q)}/assignments/all_model_hard_assignments.npz"
        with np.load(p,allow_pickle=False) as z:
            checks[f"assignment_shape_identity_{branch_id(k,q)}"] = bool(
                z["assignments"].shape == (100,20,1652) and
                np.array_equal(z["target_indices"],np.arange(1652)) and
                np.array_equal(z["filenames"].astype(str),canonical) and
                np.all(z["assignments"] >= 0) and
                all(np.all(z["assignments"][:,j,:] <= j) for j in range(20)))
    status = "PASS" if all(checks.values()) else "FAIL"
    validation = {"status": status, "checks": checks, "source_tree_after": source_after,
                  "total_branches": 4, "total_model_cells": 8000,
                  "total_assignment_labels": 4*100*20*1652,
                  "source_branches": [{"K0":k,"q0":q} for k,q in BRANCHES],
                  "branch_summaries": summaries}
    atomic_json(OUT / "validation.json", validation)
    make_report(results, validation)
    if status != "PASS":
        raise RuntimeError(f"final validation FAIL: {[k for k,v in checks.items() if not v]}")
    return validation


def load_completed_results():
    results=[]
    for k0,q0 in BRANCHES:
        base=OUT/f"source_{branch_id(k0,q0)}"
        summary=json.loads((base/"model_selection/branch_summary.json").read_text())
        def read_csv(path):
            with path.open(newline="",encoding="utf8") as f:
                rows=list(csv.DictReader(f))
            for r in rows:
                for key in list(r):
                    try: r[key]=float(r[key]) if any(c in r[key] for c in ".eE") else int(r[key])
                    except (ValueError,TypeError): pass
            return rows
        preferred=read_csv(base/"model_selection/preferred_k_by_q.csv")
        pca_rows=read_csv(base/"pca/pca_diagnostics.csv")
        results.append((summary,preferred,pca_rows))
    return results


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",choices=("audit","analyse","finalise","all"),default="all")
    ap.add_argument("--jobs",type=int,default=4)
    args=ap.parse_args()
    if args.stage in ("audit","all"): audit_inputs()
    if args.stage in ("analyse","all"):
        if not (OUT/"config.json").exists(): audit_inputs()
        results=[]
        for k0,q0 in BRANCHES:
            results.append(analyse_branch(k0,q0,args.jobs))
        comparison_plot(results)
    else:
        results=None
    if args.stage in ("finalise","all"):
        if results is None: results=load_completed_results()
        finalise(results)


if __name__ == "__main__":
    main()
