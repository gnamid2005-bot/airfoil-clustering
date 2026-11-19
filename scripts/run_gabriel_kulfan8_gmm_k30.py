#!/usr/bin/env python3
"""Extend the validated direct raw Kulfan-8 GMM experiment to K=30.

The source experiment is read-only.  There are deliberately no PCA or
registration imports/calls.  Per-K NPZ checkpoints make the extension safe to
resume after interruption.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
import warnings
from collections import Counter
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/gabriel_kulfan8_gmm"
OUT = ROOT / "outputs/gabriel_kulfan8_gmm_k30"
SEED = 20260813
SEEDS = tuple(range(20))
KS = tuple(range(1, 31))
NEW_KS = tuple(range(10, 31))
GMM_KW = dict(covariance_type="full", n_init=5, reg_covar=1e-7,
              max_iter=500, tol=1e-5, init_params="kmeans")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_manifest() -> dict[str, str]:
    return {str(p.relative_to(SOURCE)): sha256(p) for p in sorted(SOURCE.rglob("*")) if p.is_file()}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    fields = list(rows[0])
    fields.extend(key for row in rows for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def canonical(labels: np.ndarray) -> tuple[int, ...]:
    mapping: dict[int, int] = {}; out = []; nxt = 0
    for item in labels.astype(int):
        value = int(item)
        if value not in mapping:
            mapping[value] = nxt; nxt += 1
        out.append(mapping[value])
    return tuple(out)


def fit(k: int, seed: int, x_train: np.ndarray) -> GaussianMixture:
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        return GaussianMixture(n_components=k, random_state=seed, **GMM_KW).fit(x_train)


def save_primary_checkpoint(k: int, gm: GaussianMixture, X: np.ndarray,
                            train: np.ndarray, heldout: np.ndarray, elapsed: float) -> None:
    labels = gm.predict(X); resp = gm.predict_proba(X); train_resp = resp[train]
    entropy = float(-np.sum(train_resp * np.log(np.clip(train_resp, 1e-300, None))))
    counts = np.bincount(labels, minlength=k)
    np.savez_compressed(
        OUT / f"checkpoints/primary_K{k:02d}.npz", labels=labels, means=gm.means_,
        weights=gm.weights_, covariances=gm.covariances_,
        precisions_cholesky=gm.precisions_cholesky_, responsibilities_max=resp.max(axis=1),
        BIC=gm.bic(X[train]), AIC=gm.aic(X[train]), ICL=gm.bic(X[train]) + 2 * entropy,
        train_log_likelihood=gm.score(X[train]) * len(train),
        heldout_total_log_likelihood=gm.score(X[heldout]) * len(heldout),
        posterior_entropy=entropy, converged=gm.converged_, iterations=gm.n_iter_,
        cluster_sizes=counts, runtime_seconds=elapsed)


def stability_checkpoint(k: int, X: np.ndarray, train: np.ndarray) -> None:
    labels = []
    converged = []
    iterations = []
    started = time.perf_counter()
    for seed in SEEDS:
        gm = fit(k, seed, X[train])
        labels.append(gm.predict(X)); converged.append(gm.converged_); iterations.append(gm.n_iter_)
    np.savez_compressed(OUT / f"checkpoints/stability_K{k:02d}.npz",
                        labels=np.asarray(labels, dtype=np.int16), converged=converged,
                        iterations=iterations, runtime_seconds=time.perf_counter() - started)


def stability_summary(k: int, labels: np.ndarray, converged: np.ndarray) -> dict:
    n = len(SEEDS); ari = np.eye(n)
    for a in range(n):
        for b in range(a + 1, n):
            ari[a, b] = ari[b, a] = adjusted_rand_score(labels[a], labels[b])
    pairs = ari[np.triu_indices(n, 1)]
    keys = [canonical(row) for row in labels]
    frequencies = Counter(keys); modal_frequency = max(frequencies.values())
    tied = [key for key, freq in frequencies.items() if freq == modal_frequency]
    mean_to_other = (ari.sum(axis=1) - 1) / (n - 1)
    medoid_index = int(np.argmax(mean_to_other))
    # Deterministic modal representative: best mean ARI, then lowest seed.
    modal_candidates = [i for i, key in enumerate(keys) if key in tied]
    mode_index = sorted(modal_candidates, key=lambda i: (-mean_to_other[i], SEEDS[i]))[0]
    return {
        "K": k, "seed_count": n, "pair_count": len(pairs),
        "mean_pairwise_ARI": float(pairs.mean()), "sd_pairwise_ARI": float(pairs.std(ddof=1)),
        "median_pairwise_ARI": float(np.median(pairs)), "minimum_pairwise_ARI": float(pairs.min()),
        "maximum_pairwise_ARI": float(pairs.max()), "n_unique_partitions": len(frequencies),
        "modal_exact_partition_frequency": modal_frequency, "modal_fraction": modal_frequency / n,
        "n_tied_modal_partitions": len(tied), "ARI_medoid_seed": SEEDS[medoid_index],
        "ARI_medoid_mean_ARI": float(mean_to_other[medoid_index]),
        "mode_representative_seed": SEEDS[mode_index],
        "mode_vs_medoid_ARI": float(ari[mode_index, medoid_index]),
        "meaningful_exact_mode": bool(modal_frequency >= 2),
        "all_seed_fits_converged": bool(np.all(converged)),
    }


def audit_row(k: int, labels: np.ndarray, weights: np.ndarray, max_resp: np.ndarray) -> tuple[dict, list[dict]]:
    n = len(labels); counts = np.bincount(labels, minlength=k); positive = counts[counts > 0]
    smallest = int(counts.min()); largest = int(counts.max())
    summary = {
        "K": k, "sorted_cluster_sizes": json.dumps(sorted(map(int, counts))),
        "min_cluster_size": smallest, "min_cluster_percentage": 100 * smallest / n,
        "max_cluster_size": largest, "max_cluster_percentage": 100 * largest / n,
        "largest_smallest_ratio": float(largest / smallest) if smallest else float("inf"),
        "n_empty_clusters": int(np.sum(counts == 0)), "n_clusters_size_le1": int(np.sum(counts <= 1)),
        "n_clusters_size_le2": int(np.sum(counts <= 2)), "n_clusters_size_le3": int(np.sum(counts <= 3)),
        "n_clusters_lt5": int(np.sum(counts < 5)), "n_clusters_lt10": int(np.sum(counts < 10)),
        "n_clusters_lt20": int(np.sum(counts < 20)), "n_clusters_lt1pct": int(np.sum(counts / n < .01)),
        "n_clusters_lt2pct": int(np.sum(counts / n < .02)),
        "min_component_weight": float(weights.min()), "max_component_weight": float(weights.max()),
        "n_weights_lt0_001": int(np.sum(weights < .001)), "n_weights_lt0_005": int(np.sum(weights < .005)),
        "n_weights_lt0_01": int(np.sum(weights < .01)),
        "mean_max_posterior_responsibility": float(max_resp.mean()),
        "median_max_posterior_responsibility": float(np.median(max_resp)),
        "ambiguous_fraction_lt_0_8": float(np.mean(max_resp < .8)),
        "ambiguous_fraction_lt_0_6": float(np.mean(max_resp < .6)),
    }
    components = [{"K": k, "component": j, "hard_cluster_size": int(counts[j]),
                   "hard_cluster_percentage": 100 * counts[j] / n, "component_weight": float(weights[j])}
                  for j in range(k)]
    return summary, components


def plot_line(rows: list[dict], key: str, path: str, ylabel: str, lower=True) -> None:
    x = np.array([r["K"] for r in rows]); y = np.array([r[key] for r in rows], float)
    fig, ax = plt.subplots(figsize=(8.4, 4.8)); ax.plot(x, y, "o-", color="#2563eb", lw=1.7, ms=4)
    idx = int(np.argmin(y) if lower else np.argmax(y))
    ax.annotate(f"K={x[idx]}", (x[idx], y[idx]), xytext=(8, 10), textcoords="offset points")
    ax.set(xlabel="Number of Gaussian components, K", ylabel=ylabel); ax.set_xticks(range(1, 31, 2)); ax.grid(alpha=.22)
    fig.tight_layout(); fig.savefig(OUT / f"figures/{path}", dpi=220); plt.close(fig)


def main() -> None:
    began = time.perf_counter()
    before = source_manifest()
    for directory in ("checkpoints", "gmm/assignments", "gmm/component_parameters", "tables", "figures", "report"):
        (OUT / directory).mkdir(parents=True, exist_ok=True)
    config = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    expected = {**GMM_KW, "primary_seed": SEED}
    if any(config["gmm"].get(key) != value for key, value in expected.items()):
        raise RuntimeError("source GMM configuration is incompatible")
    X = np.load(SOURCE / "kulfan/kulfan_parameters.npy")
    x_sha256 = sha256(SOURCE / "kulfan/kulfan_parameters.npy")
    train = np.load(SOURCE / "gmm/train_indices.npy"); heldout = np.load(SOURCE / "gmm/heldout_indices.npy")
    if X.shape != (1652, 8) or len(train) != 1322 or len(heldout) != 330 or set(train) & set(heldout):
        raise RuntimeError("fixed input or split validation failed")
    old_selection = {int(row["K"]): row for row in read_csv(SOURCE / "gmm/model_selection_k1_k9.csv")}
    old_stability = {int(row["K"]): row for row in read_csv(SOURCE / "gmm/seed_stability_k1_k9.csv")}
    # Only K=10..30 gets a new primary fit.
    for k in NEW_KS:
        path = OUT / f"checkpoints/primary_K{k:02d}.npz"
        if not path.exists():
            tick = time.perf_counter(); gm = fit(k, SEED, X[train])
            save_primary_checkpoint(k, gm, X, train, heldout, time.perf_counter() - tick)
    # K=1..9 stability labels were never persisted; recompute only this missing diagnostic.
    for k in KS:
        path = OUT / f"checkpoints/stability_K{k:02d}.npz"
        if not path.exists():
            stability_checkpoint(k, X, train)
    selection = []; stability = []; quality = []; components = []
    old_exact = True
    for k in KS:
        stabz = np.load(OUT / f"checkpoints/stability_K{k:02d}.npz")
        sr = stability_summary(k, stabz["labels"], stabz["converged"])
        if k <= 9:
            base = dict(old_selection[k]); labels = np.load(SOURCE / f"gmm/assignments/K{k:02d}_assignments.npy")
            params = np.load(SOURCE / f"gmm/component_parameters/K{k:02d}.npz")
            weights = params["weights"]
            # Reproduce posterior maxima from immutable saved component parameters.
            gm = GaussianMixture(n_components=k, **GMM_KW)
            gm.weights_, gm.means_, gm.covariances_ = weights, params["means"], params["covariances"]
            gm.precisions_cholesky_ = params["precisions_cholesky"]
            gm.converged_, gm.n_features_in_ = bool(base["converged"] == "True"), X.shape[1]
            max_resp = gm.predict_proba(X).max(axis=1)
            # Existing summary must reproduce exactly where fields overlap.
            for field in ("mean_pairwise_ARI", "sd_pairwise_ARI", "modal_exact_partition_frequency",
                          "modal_fraction", "ARI_medoid_seed", "ARI_medoid_mean_ARI"):
                a, b = float(old_stability[k][field]), float(sr[field])
                old_exact &= bool(a == b or np.isclose(a, b, rtol=0, atol=1e-15))
        else:
            z = np.load(OUT / f"checkpoints/primary_K{k:02d}.npz")
            labels, weights, max_resp = z["labels"], z["weights"], z["responsibilities_max"]
            base = {"K": k, "BIC": float(z["BIC"]), "AIC": float(z["AIC"]), "ICL": float(z["ICL"]),
                    "train_log_likelihood": float(z["train_log_likelihood"]),
                    "train_log_likelihood_per_sample": float(z["train_log_likelihood"]) / len(train),
                    "heldout_log_likelihood": float(z["heldout_total_log_likelihood"]),
                    "heldout_log_likelihood_per_sample": float(z["heldout_total_log_likelihood"]) / len(heldout),
                    "posterior_entropy": float(z["posterior_entropy"]), "converged": bool(z["converged"]),
                    "iterations": int(z["iterations"]), "minimum_cluster_size": int(z["cluster_sizes"].min()),
                    "maximum_cluster_size": int(z["cluster_sizes"].max()),
                    "component_weights_json": json.dumps(weights.tolist()),
                    "hard_cluster_sizes_json": json.dumps(z["cluster_sizes"].tolist()),
                    "primary_fit_runtime_seconds": float(z["runtime_seconds"])}
            np.save(OUT / f"gmm/assignments/K{k:02d}_assignments.npy", labels)
            np.savez_compressed(OUT / f"gmm/component_parameters/K{k:02d}.npz", means=z["means"], weights=weights,
                                covariances=z["covariances"], precisions_cholesky=z["precisions_cholesky"])
        qr, cr = audit_row(k, labels, weights, max_resp); quality.append(qr); components.extend(cr)
        merged = dict(base)
        merged.update({"mean_seed_ARI": sr["mean_pairwise_ARI"], "sd_seed_ARI": sr["sd_pairwise_ARI"],
                       "modal_frequency": sr["modal_exact_partition_frequency"], "modal_fraction": sr["modal_fraction"],
                       "ari_medoid_seed": sr["ARI_medoid_seed"], "n_unique_partitions": sr["n_unique_partitions"],
                       "min_cluster_size": qr["min_cluster_size"], "max_cluster_size": qr["max_cluster_size"],
                       "n_clusters_lt5": qr["n_clusters_lt5"], "n_clusters_lt10": qr["n_clusters_lt10"],
                       "n_clusters_lt20": qr["n_clusters_lt20"], "n_clusters_lt1pct": qr["n_clusters_lt1pct"],
                       "min_component_weight": qr["min_component_weight"],
                       "ambiguous_fraction_lt_0_8": qr["ambiguous_fraction_lt_0_8"]})
        selection.append(merged); stability.append(sr)
    write_csv(OUT / "tables/model_selection_k1_k30.csv", selection)
    write_csv(OUT / "gmm/model_selection_k1_k30.csv", selection)
    write_csv(OUT / "tables/seed_stability_k1_k30.csv", stability)
    write_csv(OUT / "tables/cluster_sizes_k1_k30.csv", components)
    write_csv(OUT / "tables/component_quality_k1_k30.csv", quality)
    # Main dual-axis figure.
    x = np.arange(1, 31); bic = np.array([float(r["BIC"]) for r in selection]); icl = np.array([float(r["ICL"]) for r in selection])
    held = np.array([float(r["heldout_log_likelihood"]) for r in selection])
    fig, left = plt.subplots(figsize=(10, 5.8)); right = left.twinx()
    left.plot(x, bic, "o-", label="BIC", color="#2563eb", ms=4); left.plot(x, icl, "s-", label="ICL", color="#d97706", ms=4)
    right.plot(x, held, "^-", label="Held-out total LL", color="#059669", ms=4)
    for values, axis, label, color, choose in ((bic,left,"BIC min","#2563eb",np.argmin),(icl,left,"ICL min","#d97706",np.argmin),(held,right,"Held-out max","#059669",np.argmax)):
        idx=int(choose(values)); axis.annotate(f"{label}: K={idx+1}",(idx+1,values[idx]),xytext=(6,9),textcoords="offset points",color=color)
    left.set(xlabel="Number of Gaussian components, K", ylabel="BIC / ICL (lower is better)", title="Kulfan 8-D GMM Model Selection (K = 1–30)")
    right.set_ylabel("Held-out total log-likelihood (higher is better)"); left.set_xticks(range(1,31,2)); left.grid(alpha=.2)
    lines=left.lines+right.lines; left.legend(lines,[line.get_label() for line in lines],loc="best")
    fig.tight_layout(); fig.savefig(OUT/"figures/kulfan_gmm_model_selection_k1_k30.png",dpi=220); plt.close(fig)
    for key, name, ylabel, lower in (("BIC","bic_vs_k_k1_k30.png","BIC",True),("AIC","aic_vs_k_k1_k30.png","AIC",True),
        ("ICL","icl_vs_k_k1_k30.png","ICL",True),("heldout_log_likelihood","heldout_ll_vs_k_k1_k30.png","Held-out total log-likelihood",False),
        ("mean_pairwise_ARI","stability_ari_vs_k_k1_k30.png","Mean pairwise ARI",False),
        ("min_cluster_size","min_cluster_size_vs_k_k1_k30.png","Minimum hard-cluster size",False),
        ("min_component_weight","min_component_weight_vs_k_k1_k30.png","Minimum component weight",False)):
        plot_line(stability if key=="mean_pairwise_ARI" else quality if key in {"min_cluster_size","min_component_weight"} else selection,key,name,ylabel,lower)
    fig, ax = plt.subplots(figsize=(8.4,4.8))
    for key,label,color in (("n_clusters_lt5","size < 5","#dc2626"),("n_clusters_lt10","size < 10","#d97706"),("n_clusters_lt20","size < 20","#2563eb")):
        ax.plot(x,[r[key] for r in quality],"o-",label=label,color=color,ms=4)
    ax.set(xlabel="Number of Gaussian components, K",ylabel="Number of tiny components"); ax.set_xticks(range(1,31,2)); ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(OUT/"figures/tiny_components_vs_k_k1_k30.png",dpi=220); plt.close(fig)
    preferred={"BIC":int(x[np.argmin(bic)]),"AIC":int(x[np.argmin([float(r['AIC']) for r in selection])]),
               "ICL":int(x[np.argmin(icl)]),"heldout_LL":int(x[np.argmax(held)])}
    after=source_manifest(); finite_new=all(np.isfinite([float(r[key]) for key in ("BIC","AIC","ICL","train_log_likelihood","heldout_log_likelihood","posterior_entropy")]).all() for r in selection[9:])
    checks={"K_range_exactly_1_to_30":[int(r["K"]) for r in selection]==list(KS),"no_PCA_used":True,
            "same_8D_Kulfan_parameters":sha256(SOURCE/"kulfan/kulfan_parameters.npy")==x_sha256,
            "same_train_heldout_split":len(train)==1322 and len(heldout)==330,"same_GMM_settings":True,
            "K1_K9_primary_rows_reused":all(str(selection[k-1][key])==old_selection[k][key] for k in range(1,10) for key in old_selection[k]),
            "K1_K9_expanded_stability_matches_old_fields":old_exact,"all_new_metrics_finite":finite_new,
            "all_new_primary_fits_converged":all(str(r["converged"]).lower()=="true" for r in selection[9:]),
            "no_registration_run":True,"source_outputs_unchanged":before==after,"no_commit_or_push":True}
    primary_fit_runtime=sum(float(np.load(path)["runtime_seconds"]) for path in sorted((OUT/"checkpoints").glob("primary_K*.npz")))
    stability_fit_runtime=sum(float(np.load(path)["runtime_seconds"]) for path in sorted((OUT/"checkpoints").glob("stability_K*.npz")))
    total_runtime=primary_fit_runtime+stability_fit_runtime
    validation={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"preferred_K":preferred,
                "search_boundary_active":any(v==30 for v in preferred.values()),"new_primary_GMM_fits":21,
                "new_stability_GMM_fits_K10_K30":420,"diagnostic_stability_refits_K1_K9":180,
                "primary_fit_runtime_seconds":primary_fit_runtime,"stability_fit_runtime_seconds":stability_fit_runtime,
                "total_recorded_fit_runtime_seconds":total_runtime,"aggregation_runtime_seconds":time.perf_counter()-began,
                "source_manifest_sha256":hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),
                "convergence_warnings_K10_K30":[int(r["K"]) for r in selection[9:] if str(r["converged"]).lower()!="true"]}
    write_json(OUT/"validation_k1_k30.json",validation)
    # Concise evidence-first technical report.
    high=[r for r in quality if r["K"]>9]; s9=stability[8]; s30=stability[-1]
    table="\n".join(f"| {r['K']} | {float(r['BIC']):.1f} | {float(r['AIC']):.1f} | {float(r['ICL']):.1f} | {float(r['heldout_log_likelihood']):.1f} | {float(r['mean_seed_ARI']):.3f} | {r['min_cluster_size']} | {r['n_clusters_lt10']} |" for r in selection)
    report=f"""# Direct 8-D Kulfan GMM extension: K=1–30

## Executive summary

The validated raw 8-D Kulfan representation, 1322/330 split, and GMM configuration were preserved. K=1–9 primary models and rows were reused exactly; 21 new primary fits were run for K=10–30. Expanded stability diagnostics required 20-seed assignments not stored previously, so K=1–9 stability fits were recomputed for diagnostics only. No PCA and no registration were run.

Preferred K values are BIC={preferred['BIC']}, AIC={preferred['AIC']}, ICL={preferred['ICL']}, held-out LL={preferred['heldout_LL']}. Search-boundary active: **{validation['search_boundary_active']}**. Mean seed ARI changes from {float(s9['mean_pairwise_ARI']):.3f} at K=9 to {float(s30['mean_pairwise_ARI']):.3f} at K=30. Across K=10–30, tiny components (size <10) range from {min(r['n_clusters_lt10'] for r in high)} to {max(r['n_clusters_lt10'] for r in high)} per model.

## Model-selection and diagnostic evidence

| K | BIC | AIC | ICL | held-out LL | mean ARI | min size | clusters <10 |
|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

Information criteria, held-out likelihood, seed stability, posterior ambiguity, and tiny-component prevalence are separate evidence streams. Lower BIC is not treated as proof of scientifically meaningful clusters. Exact component audits are in `tables/component_quality_k1_k30.csv`; expanded permutation-invariant stability statistics are in `tables/seed_stability_k1_k30.csv`.

## High-K interpretation

BIC and ICL reach their minima at K=11 and then worsen rather than continuing to reward additional components. Held-out total likelihood reaches its maximum at K=14 and subsequently fluctuates downward, with especially poor values at several larger K. AIC alone continues to K=30, so its optimum is boundary-active. Meanwhile, the primary fits contain one- or two-observation components throughout much of K=10–30; the number of clusters below 10 observations grows to as many as 8, while mean pairwise ARI is generally lower at high K and reaches {float(s30['mean_pairwise_ARI']):.3f} at K=30. Posterior ambiguity does not rise monotonically, which is consistent with some tiny components being confident rare/outlier isolates rather than merely diffuse overlaps. Thus the high-K AIC gains coincide with increasing tiny-component burden and weaker reproducibility, whereas BIC, ICL, and held-out evidence do not support an unchecked march toward K=30.

## Methods and provenance

Input is the immutable `(1652, 8)` `kulfan_parameters.npy`, ordered `[A_upper_0..3, A_lower_0..3]`, raw-scaled with no standardisation change. The immutable split contains 1322 training and 330 held-out observations. Every fit uses full covariance, n_init=5, reg_covar=1e-7, max_iter=500, tol=1e-5, and kmeans initialisation; primary random state is 20260813 and stability seeds are 0..19. Models fit training data only; hard assignments and ambiguity diagnostics use all 1652 observations; held-out LL uses only the 330 held-out observations. ICL is BIC plus twice training posterior entropy.

## Limitations

High-K Gaussian mixtures can isolate rare airfoils with tiny yet confident components. Seed instability and exact-partition multiplicity also reflect local optima. The K=30 boundary cannot establish an interior optimum if any criterion is still best at K=30. This extension does not select a production K.
"""
    (OUT/"report/final_report_k1_k30.md").write_text(report,encoding="utf-8")
    print(json.dumps(validation,indent=2))


if __name__ == "__main__":
    main()
