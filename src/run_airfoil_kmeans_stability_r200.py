"""Exact R=200 extension of the original direct-airfoil K-means experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import time
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score

INPUT = Path("data/airfoils/processed/airfoils_n100_seed0.npz")
OLD = Path("outputs/airfoil_kmeans_stability")
OUT = Path("outputs/airfoil_kmeans_stability_r200")
KS = list(range(2, 21))
SEEDS = list(range(200))
SUBSETS = [20, 50, 100, 200]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def summary(v: np.ndarray) -> dict:
    return {
        "mean_pairwise_ari": float(v.mean()), "std_pairwise_ari": float(v.std(ddof=0)),
        "median_pairwise_ari": float(np.median(v)), "min_pairwise_ari": float(v.min()),
        "max_pairwise_ari": float(v.max()), "p05_pairwise_ari": float(np.percentile(v, 5)),
        "p25_pairwise_ari": float(np.percentile(v, 25)), "p75_pairwise_ari": float(np.percentile(v, 75)),
        "p95_pairwise_ari": float(np.percentile(v, 95)),
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title); ax.set_xlabel("Number of clusters K"); ax.set_ylabel(ylabel)
    ax.set_xticks(KS); ax.grid(True, alpha=.22, color="#9CA3AF")
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    for d in [OUT / "all_runs", OUT / "tables", OUT / "figures", OUT / "report"]:
        d.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with np.load(INPUT, allow_pickle=False) as z:
        shapes = np.asarray(z["shapes"], dtype=np.float64)
    x = np.ascontiguousarray(shapes.reshape(shapes.shape[0], -1))
    input_hash = digest(INPUT)
    with (OLD / "kmeans_stability_summary.csv").open(newline="", encoding="utf-8") as f:
        old_summary = {int(r["k"]): r for r in csv.DictReader(f)}
    old_labels = np.load(OLD / "kmeans_labels.npz", allow_pickle=False)["labels"]
    runs, full_pairs, summaries, convergence = [], [], [], []
    labels_by_k = {}

    fit_started = time.perf_counter()
    for ki, k in enumerate(KS):
        labels_matrix = np.empty((200, shapes.shape[0]), dtype=np.int32)
        for seed in SEEDS:
            model = KMeans(n_clusters=k, init="k-means++", n_init=1, max_iter=1000,
                           tol=1e-6, random_state=seed, algorithm="lloyd")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                labels = model.fit_predict(x)
            cw = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
            counts = np.bincount(labels, minlength=k)
            labels_matrix[seed] = labels
            convergence.extend([{"k": k, "seed": seed, "warning": w} for w in cw])
            runs.append({"k": k, "seed": seed, "inertia": float(model.inertia_),
                         "n_iterations": int(model.n_iter_), "converged": not cw and model.n_iter_ < model.max_iter,
                         "n_nonempty_clusters": int(np.count_nonzero(counts)),
                         "cluster_size_sum": int(counts.sum()), "minimum_cluster_size": int(counts.min()),
                         "maximum_cluster_size": int(counts.max()), "cluster_sizes": json.dumps(counts.tolist())})
        labels_by_k[k] = labels_matrix
        np.savez_compressed(OUT / "all_runs" / f"labels_k{k:02d}.npz", labels=labels_matrix,
                            seeds=np.asarray(SEEDS), k=np.asarray(k))
    fit_seconds = time.perf_counter() - fit_started

    convergence_rows = []
    for k in KS:
        lm = labels_by_k[k]
        vals = []
        for a, b in combinations(range(200), 2):
            ari = float(adjusted_rand_score(lm[a], lm[b]))
            vals.append(ari); full_pairs.append({"k": k, "seed_a": a, "seed_b": b, "ari": ari})
        av = np.asarray(vals)
        summaries.append({"k": k, "n_runs": 200, "n_pairwise_comparisons": len(av), **summary(av)})
        for r in SUBSETS:
            sub = np.asarray([adjusted_rand_score(lm[a], lm[b]) for a, b in combinations(range(r), 2)])
            convergence_rows.append({"k": k, "n_runs": r, "n_pairwise_comparisons": len(sub),
                                     "mean_pairwise_ari": float(sub.mean()),
                                     "std_pairwise_ari": float(sub.std(ddof=0))})
    pair_seconds = time.perf_counter() - fit_started - fit_seconds

    comparison = []
    for row in summaries:
        k = row["k"]
        comparison.append({"k": k, "mean_ari_r20": float(old_summary[k]["mean_pairwise_ari"]),
                           "sd_ari_r20": float(old_summary[k]["std_pairwise_ari"]),
                           "mean_ari_r200": row["mean_pairwise_ari"], "sd_ari_r200": row["std_pairwise_ari"],
                           "change_mean_r200_minus_r20": row["mean_pairwise_ari"] - float(old_summary[k]["mean_pairwise_ari"]),
                           "change_sd_r200_minus_r20": row["std_pairwise_ari"] - float(old_summary[k]["std_pairwise_ari"])})
    write_rows(OUT / "tables" / "kmeans_stability_r200.csv", runs)
    write_rows(OUT / "tables" / "pairwise_ari_summary_r200.csv", summaries)
    write_rows(OUT / "tables" / "pairwise_ari_r200.csv", full_pairs)
    write_rows(OUT / "tables" / "r20_vs_r200_comparison.csv", comparison)
    write_rows(OUT / "tables" / "stability_convergence_r20_r50_r100_r200.csv", convergence_rows)

    k = np.asarray(KS); mean = np.asarray([r["mean_pairwise_ari"] for r in summaries]); sd = np.asarray([r["std_pairwise_ari"] for r in summaries])
    fig, ax = plt.subplots(figsize=(12, 7)); ax.fill_between(k, np.clip(mean-sd,-1,1), np.clip(mean+sd,-1,1), color="#93C5FD", alpha=.42, label="mean ± 1 SD"); ax.plot(k, mean, "o-", color="#1D4ED8", lw=2, label="mean pairwise ARI"); style(ax, "K-means Stability vs Number of Clusters\n200 independent runs per K; 19,900 pairwise ARIs per K", "Adjusted Rand Index (ARI)"); ax.set_ylim(-.05,1.05); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT/"figures"/"kmeans_stability_vs_k_r200.png", dpi=300); plt.close(fig)
    old_mean=np.asarray([float(old_summary[int(v)]["mean_pairwise_ari"]) for v in k]); old_sd=np.asarray([float(old_summary[int(v)]["std_pairwise_ari"]) for v in k])
    fig, ax = plt.subplots(figsize=(12,7)); ax.plot(k, old_mean, "o--", color="#C2410C", label="R=20"); ax.plot(k, mean, "o-", color="#1D4ED8", label="R=200"); style(ax,"Mean ARI: 20 vs 200 Independent Runs","Mean pairwise ARI"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT/"figures"/"kmeans_stability_r20_vs_r200.png",dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12,7)); ax.plot(k, old_sd, "o--", color="#C2410C", label="R=20"); ax.plot(k, sd, "o-", color="#1D4ED8", label="R=200"); style(ax,"Pairwise ARI Standard Deviation: 20 vs 200 Runs","ARI standard deviation (ddof=0)"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT/"figures"/"ari_sd_r20_vs_r200.png",dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12,7)); colors={20:"#C2410C",50:"#B45309",100:"#0F766E",200:"#1D4ED8"};
    for r in SUBSETS:
        d=[v for v in convergence_rows if v["n_runs"]==r]; ax.plot([v["k"] for v in d],[v["mean_pairwise_ari"] for v in d],"o-",lw=1.7,color=colors[r],label=f"R={r}")
    style(ax,"Convergence of Mean Stability Estimate","Mean pairwise ARI"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT/"figures"/"stability_convergence_vs_runs.png",dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12,7));
    for r in SUBSETS:
        d=[v for v in convergence_rows if v["n_runs"]==r]; ax.plot([v["k"] for v in d],[v["std_pairwise_ari"] for v in d],"o-",lw=1.7,color=colors[r],label=f"R={r}")
    style(ax,"Convergence of Pairwise ARI Standard Deviation","ARI standard deviation (ddof=0)"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT/"figures"/"ari_sd_convergence_vs_runs.png",dpi=300); plt.close(fig)

    exact_old_subset = all(np.array_equal(labels_by_k[k][:20], old_labels[i]) for i,k in enumerate(KS))
    runtime = time.perf_counter()-started
    validation = {"input_shape": list(shapes.shape), "flattened_shape": list(x.shape), "input_sha256": input_hash,
                  "direct_shape_input": True, "pca_used": False, "velocity_fields_used": False,
                  "k_values": KS, "exactly_200_runs_each_k": all(sum(v["k"]==kk for v in runs)==200 for kk in KS),
                  "seeds_distinct_reproducible": SEEDS == list(range(200)), "n_init_inside_each_fit": 1,
                  "exactly_19900_pairwise_each_k": all(sum(v["k"]==kk for v in full_pairs)==19900 for kk in KS),
                  "all_ari_finite": bool(np.isfinite([v["ari"] for v in full_pairs]).all()), "all_inertia_finite": bool(np.isfinite([v["inertia"] for v in runs]).all()),
                  "all_cluster_sizes_sum_n": all(v["cluster_size_sum"]==1654 for v in runs),
                  "all_requested_clusters_nonempty": all(v["n_nonempty_clusters"]==v["k"] for v in runs),
                  "all_runs_converged": all(v["converged"] for v in runs), "convergence_warning_count": len(convergence),
                  "old_seed_0_19_labels_exactly_reproduced": exact_old_subset,
                  "total_fits": len(runs), "total_pairwise_ari": len(full_pairs), "fit_runtime_seconds": fit_seconds,
                  "pairwise_runtime_seconds": pair_seconds, "total_runtime_seconds": runtime}
    (OUT/"validation.json").write_text(json.dumps(validation,indent=2,sort_keys=True),encoding="utf-8")
    config={"input":str(INPUT),"input_sha256":input_hash,"representation":"direct C-order flattened (1654,200,2) -> (1654,400)","k_values":KS,"seeds":SEEDS,"independent_runs_per_k":200,"kmeans":{"implementation":"sklearn.cluster.KMeans","init":"k-means++","n_init":1,"max_iter":1000,"tol":1e-6,"algorithm":"lloyd"},"stability":{"metric":"Adjusted Rand Index","definition":"mean of all pairwise ARIs","pairs_per_k":19900,"std":"population SD of pairwise ARIs, ddof=0"}}
    (OUT/"experiment_config.json").write_text(json.dumps(config,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(validation,indent=2))


if __name__ == "__main__":
    main()
