"""PCA-dimension sensitivity audit for Gabriel pinned-shape GMM clustering.

Clustering only: this module has no registration imports or registration actions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import warnings
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-gabriel-gmm-q-sensitivity")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/airfoils/processed_gabriel/airfoils_pinned_arclength_n200.npz"
PREV = ROOT / "outputs/gabriel_pinned_gmm_vs_kmeans"
OUT = ROOT / "outputs/gabriel_pinned_gmm_q_sensitivity"
QS = tuple(range(2, 21))
KS = tuple(range(2, 10))
SEEDS = tuple(range(20))
SPLIT_SEED = 20260813
N_INIT = 5
REG_COVAR = 1e-7
MAX_ITER = 500
TOL = 1e-5


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jwrite(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def cwrite(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or sorted({key for row in rows for key in row})
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    tmp.replace(path)


def cread(path: Path) -> list[dict]:
    if not path.exists(): return []
    with path.open(newline="", encoding="utf-8") as f: return list(csv.DictReader(f))


def canonical(labels: np.ndarray) -> tuple[int, ...]:
    mapping, nxt, out = {}, 0, []
    for value in labels.astype(int):
        if value not in mapping:
            mapping[value] = nxt; nxt += 1
        out.append(mapping[value])
    return tuple(out)


def load_and_verify():
    prev_cfg = json.loads((PREV / "config.json").read_text())
    expected_gmm = {"covariance_type": "full", "n_init": N_INIT, "reg_covar": REG_COVAR,
                    "max_iter": MAX_ITER, "tol": TOL, "seed": SPLIT_SEED}
    for key, value in expected_gmm.items():
        if prev_cfg["gmm"].get(key) != value:
            raise RuntimeError(f"STOP: previous GMM configuration mismatch: {key}")
    if prev_cfg["split"] != {"gmm_fit_train_only": True, "heldout_count": 330,
                              "pca_fit_train_only": True, "seed": SPLIT_SEED, "train_count": 1322}:
        raise RuntimeError("STOP: unexpected previous split configuration")
    with np.load(DATA, allow_pickle=False) as z:
        contours = z["contours"].copy(); filenames = z["filenames"].astype(str)
    if contours.shape != (1652, 200, 2) or contours.dtype != np.float64 or not np.isfinite(contours).all():
        raise RuntimeError(f"STOP: bad pinned data {contours.shape} {contours.dtype}")
    X = np.ascontiguousarray(contours).reshape(1652, 400, order="C")
    train_idx = np.load(PREV / "splits/train_indices.npy")
    heldout_idx = np.load(PREV / "splits/heldout_indices.npy")
    rng = np.random.default_rng(SPLIT_SEED); order = rng.permutation(1652); n_train = int(round(.8 * 1652))
    expected_train, expected_heldout = np.sort(order[:n_train]), np.sort(order[n_train:])
    if not np.array_equal(train_idx, expected_train) or not np.array_equal(heldout_idx, expected_heldout):
        raise RuntimeError("STOP: saved split differs from deterministic split")
    with np.load(PREV / "pca/pca_q100.npz", allow_pickle=False) as z:
        mean = z["mean"].copy(); components = z["components"].copy()
        evr = z["explained_variance_ratio"].copy(); train_scores_saved = z["train_scores"].copy()
        heldout_scores_saved = z["heldout_scores"].copy()
    train_scores = (X[train_idx] - mean) @ components.T
    heldout_scores = (X[heldout_idx] - mean) @ components.T
    full_scores = (X - mean) @ components.T
    # sklearn randomized PCA's fit_transform may differ from transform in the
    # discarded high-order tail; the audited q=2..20 scores must agree tightly.
    if not np.allclose(train_scores[:, :20], train_scores_saved[:, :20], rtol=0, atol=5e-14):
        raise RuntimeError("STOP: saved PCA train transform cannot be reproduced for q<=20")
    if not np.allclose(heldout_scores[:, :20], heldout_scores_saved[:, :20], rtol=0, atol=5e-14):
        raise RuntimeError("STOP: saved PCA heldout transform cannot be reproduced")
    return contours, filenames, X, train_idx, heldout_idx, mean, components, evr, train_scores, heldout_scores, full_scores, prev_cfg


def pca_table(X, train_idx, heldout_idx, mean, components, evr, train_scores, heldout_scores):
    rows = []
    for q in QS:
        tr_rec = train_scores[:, :q] @ components[:q] + mean
        ho_rec = heldout_scores[:, :q] @ components[:q] + mean
        rows.append({"q": q, "marginal_explained_variance": float(evr[q - 1]),
                     "cumulative_explained_variance": float(evr[:q].sum()),
                     "train_reconstruction_mse": float(np.mean((X[train_idx] - tr_rec) ** 2)),
                     "heldout_reconstruction_mse": float(np.mean((X[heldout_idx] - ho_rec) ** 2))})
    cwrite(OUT / "tables/pca_q2_q20.csv", rows)
    return rows


def fit_one(train_scores, heldout_scores, full_scores, q, k, seed):
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = GaussianMixture(n_components=k, covariance_type="full", n_init=N_INIT,
            random_state=seed, reg_covar=REG_COVAR, max_iter=MAX_ITER, tol=TOL,
            init_params="kmeans").fit(train_scores[:, :q])
    resp = model.predict_proba(train_scores[:, :q]); lab_train = resp.argmax(1)
    labels = model.predict(full_scores[:, :q]); sizes = np.bincount(labels, minlength=k)
    entropy = float(-np.sum(resp * np.log(np.clip(resp, 1e-300, 1))))
    bic = float(model.bic(train_scores[:, :q]))
    row = {"q": q, "K": k, "seed": seed, "BIC": bic, "ICL": bic + 2 * entropy,
           "train_log_likelihood_per_sample": float(model.score(train_scores[:, :q])),
           "heldout_log_likelihood_per_sample": float(model.score(heldout_scores[:, :q])),
           "posterior_entropy": entropy, "component_weights": json.dumps(model.weights_.tolist(), separators=(",", ":")),
           "full_cluster_sizes": json.dumps(sizes.tolist(), separators=(",", ":")),
           "minimum_component_size": int(sizes.min()), "minimum_component_weight": float(model.weights_.min()),
           "tiny_component_flag": bool(np.any((sizes < 10) | (sizes < .01 * len(labels)))),
           "converged": bool(model.converged_), "iterations": int(model.n_iter_),
           "warnings": " | ".join(str(w.message) for w in caught), "runtime_seconds": time.perf_counter() - start}
    return row, labels


def load_existing_fit(q, k, seed):
    def normalize(row, labels):
        row = dict(row)
        sizes = np.bincount(labels.astype(int), minlength=k)
        row.setdefault("minimum_component_size", int(sizes.min()))
        row.setdefault("tiny_component_flag", bool(np.any((sizes < 10) | (sizes < .01 * len(labels)))))
        if "posterior_entropy" not in row and "classification_entropy" in row:
            row["posterior_entropy"] = row["classification_entropy"]
        return row
    label_path = OUT / "seed_assignments" / f"q{q:02d}_K{k:02d}_seed{seed:02d}.npy"
    fit_path = OUT / "seed_fits" / f"q{q:02d}_K{k:02d}_seed{seed:02d}.json"
    if label_path.exists() and fit_path.exists():
        labels = np.load(label_path); row = json.loads(fit_path.read_text())
        if labels.shape == (1652,) and row.get("q") == q and row.get("K") == k and row.get("seed") == seed and row.get("converged"):
            source = row.get("reuse_source", "reused_current")
            return normalize(row, labels), labels, source
    if q == 14:
        label_path = PREV / "gmm" / f"stability_K{k:02d}_seed{seed:02d}_labels.npy"
        fit_path = PREV / "gmm" / f"stability_K{k:02d}_seed{seed:02d}_fit.json"
        if label_path.exists() and fit_path.exists():
            labels = np.load(label_path); row = json.loads(fit_path.read_text())
            ok = (labels.shape == (1652,) and int(row["q"]) == 14 and int(row["K"]) == k and int(row["seed"]) == seed
                  and int(row["n_init"]) == N_INIT and bool(row["converged"]))
            if ok: return normalize(row, labels), labels, "reused_previous_q14"
    return None


def run_fits(train_scores, heldout_scores, full_scores, jobs):
    reuse_previous = reuse_current = new_count = original_new = 0
    start = time.perf_counter()
    for q in QS:
        pending, cached = [], {}
        for k in KS:
            for seed in SEEDS:
                old = load_existing_fit(q, k, seed)
                if old is None: pending.append((q, k, seed))
                else:
                    cached[(k, seed)] = old
                    if old[2] == "reused_previous_q14": reuse_previous += 1
                    elif old[2] == "new": original_new += 1
                    else: reuse_current += 1
        if pending:
            print(f"q={q}: fitting {len(pending)}, cached {len(cached)}", flush=True)
            fitted = Parallel(n_jobs=jobs, prefer="threads", batch_size=1, verbose=5)(
                delayed(fit_one)(train_scores, heldout_scores, full_scores, qq, k, seed) for qq, k, seed in pending)
            for (_, k, seed), (row, labels) in zip(pending, fitted):
                cached[(k, seed)] = (row, labels, "new"); new_count += 1
        for (k, seed), (row, labels, source) in cached.items():
            np.save(OUT / "seed_assignments" / f"q{q:02d}_K{k:02d}_seed{seed:02d}.npy", labels)
            saved = dict(row); saved.setdefault("reuse_source", source)
            jwrite(OUT / "seed_fits" / f"q{q:02d}_K{k:02d}_seed{seed:02d}.json", saved)
    original_durations = []
    for p in (OUT / "seed_fits").glob("*.json"):
        row = json.loads(p.read_text())
        if row.get("reuse_source") == "new": original_durations.append(float(row["runtime_seconds"]))
    return {"total_required": len(QS) * len(KS) * len(SEEDS), "new_fits_this_invocation": new_count,
            "new_fits_originally_computed_for_audit": original_new + new_count,
            "reused_previous_q14": reuse_previous, "reused_current_output": reuse_current,
            "fit_stage_runtime_seconds_this_invocation": time.perf_counter() - start,
            "sum_new_fit_runtime_seconds": float(sum(original_durations)),
            "estimated_parallel_fit_wall_seconds_sum_div_jobs": float(sum(original_durations) / jobs)}


def representatives():
    seed_rows, quality_rows, representative = [], [], {}
    for q in QS:
        for k in KS:
            labels = [np.load(OUT / "seed_assignments" / f"q{q:02d}_K{k:02d}_seed{s:02d}.npy") for s in SEEDS]
            fits = [json.loads((OUT / "seed_fits" / f"q{q:02d}_K{k:02d}_seed{s:02d}.json").read_text()) for s in SEEDS]
            ari = np.eye(len(SEEDS))
            for i in range(len(SEEDS)):
                for j in range(i + 1, len(SEEDS)):
                    ari[i, j] = ari[j, i] = adjusted_rand_score(labels[i], labels[j])
            pair = ari[np.triu_indices(len(SEEDS), 1)]
            medoid_scores = (ari.sum(1) - 1) / (len(SEEDS) - 1); medoid_idx = int(np.argmax(medoid_scores))
            keys = [canonical(x) for x in labels]; counts = Counter(keys); ranked = counts.most_common()
            mode_key, mode_freq = ranked[0]; unique_mode = len(ranked) == 1 or ranked[0][1] > ranked[1][1]
            mode_indices = [i for i, key in enumerate(keys) if key == mode_key]
            modal_seed = SEEDS[max(mode_indices, key=lambda i: medoid_scores[i])]
            meaningful = unique_mode and mode_freq >= 2
            rep_idx = SEEDS.index(modal_seed) if meaningful else medoid_idx
            rep_type = "exact_modal_partition" if meaningful else "ARI_medoid_no_meaningful_unique_mode"
            representative[(q, k)] = labels[rep_idx]
            np.save(OUT / "assignments" / f"representative_q{q:02d}_K{k:02d}.npy", labels[rep_idx])
            seed_rows.append({"q": q, "K": k, "n_fits": len(SEEDS), "mean_pairwise_ARI": float(pair.mean()),
                "sd_pairwise_ARI": float(pair.std(ddof=1)), "median_pairwise_ARI": float(np.median(pair)),
                "minimum_pairwise_ARI": float(pair.min()), "maximum_pairwise_ARI": float(pair.max()),
                "n_exact_partitions": len(counts), "unique_exact_mode": unique_mode, "modal_frequency": mode_freq,
                "modal_fraction": mode_freq / len(SEEDS), "modal_representative_seed": modal_seed,
                "ARI_medoid_seed": SEEDS[medoid_idx], "representative_seed": SEEDS[rep_idx], "representative_type": rep_type})
            fit = fits[rep_idx]
            quality_rows.append({"q": q, "K": k, "representative_seed": SEEDS[rep_idx], "representative_type": rep_type,
                **{name: fit[name] for name in ("BIC", "ICL", "heldout_log_likelihood_per_sample", "posterior_entropy",
                    "minimum_component_size", "minimum_component_weight", "tiny_component_flag", "converged", "iterations")}})
    cwrite(OUT / "tables/q_k_gmm_seed_stability.csv", seed_rows)
    cwrite(OUT / "tables/q_k_model_quality.csv", quality_rows)
    return seed_rows, quality_rows, representative


def maximal_stable_ranges(matrix, threshold):
    ranges = []
    for i in range(len(QS)):
        for j in range(i + 1, len(QS)):
            block = matrix[i:j+1, i:j+1]
            vals = block[np.triu_indices(j - i + 1, 1)]
            if len(vals) and np.min(vals) >= threshold:
                left_extend = i > 0 and np.min(matrix[i-1:j+1, i-1:j+1][np.triu_indices(j-i+2, 1)]) >= threshold
                right_extend = j + 1 < len(QS) and np.min(matrix[i:j+2, i:j+2][np.triu_indices(j-i+2, 1)]) >= threshold
                if not left_extend and not right_extend: ranges.append([QS[i], QS[j]])
    return ranges


def across_q(representative):
    matrices, summary, q14_rows, compact = {}, [], [], []
    for k in KS:
        labs = [representative[(q, k)] for q in QS]
        M = np.eye(len(QS))
        for i in range(len(QS)):
            for j in range(i + 1, len(QS)):
                M[i, j] = M[j, i] = adjusted_rand_score(labs[i], labs[j])
        matrices[k] = M; np.savetxt(OUT / "tables" / f"q_by_q_ari_K{k:02d}.csv", M, delimiter=",")
        pair = M[np.triu_indices(len(QS), 1)]; i14 = QS.index(14); i20 = QS.index(20)
        ref14 = M[i14]; other14 = np.delete(ref14, i14)
        for i, q in enumerate(QS):
            q14_rows.append({"K": k, "q": q, "ARI_vs_q14": float(ref14[i]), "ARI_vs_q20": float(M[i, i20])})
        summary.append({"K": k, "mean_across_q_ARI": float(pair.mean()), "median_across_q_ARI": float(np.median(pair)),
            "min_across_q_ARI": float(pair.min()), "n_unique_partitions_across_q": len({canonical(x) for x in labs}),
            "q14_mean_ARI": float(other14.mean()), "q14_min_ARI": float(other14.min()), "q14_median_ARI": float(np.median(other14)),
            "n_q_ARI_ge_0_95_vs_q14": int(np.sum(ref14 >= .95)), "fraction_q_ARI_ge_0_95_vs_q14": float(np.mean(ref14 >= .95)),
            "n_q_ARI_ge_0_90_vs_q14": int(np.sum(ref14 >= .90)), "fraction_q_ARI_ge_0_90_vs_q14": float(np.mean(ref14 >= .90)),
            "stable_ranges_ARI_ge_0_95": json.dumps(maximal_stable_ranges(M, .95)),
            "stable_ranges_ARI_ge_0_90": json.dumps(maximal_stable_ranges(M, .90))})
        idx = {q: QS.index(q) for q in (9, 14, 17, 20)}
        compact.append({"K": k, **{f"ARI_{a}_{b}": float(M[idx[a], idx[b]]) for a, b in ((9,14),(9,17),(9,20),(14,17),(14,20),(17,20))}})
    cwrite(OUT / "tables/q_sensitivity_summary_by_k.csv", summary)
    cwrite(OUT / "tables/q14_reference_ari.csv", q14_rows)
    cwrite(OUT / "tables/q9_q14_q17_q20_comparison.csv", compact)
    return matrices, summary, q14_rows, compact


def template_sensitivity(contours, representative):
    rows = []
    for k in KS:
        templates = {}
        for q in QS:
            lab = representative[(q, k)]
            t = np.stack([contours[lab == c].mean(0) for c in range(k)])
            templates[q] = t
            np.savez_compressed(OUT / "templates" / f"hard_templates_q{q:02d}_K{k:02d}.npz", coordinates=t,
                                cluster_sizes=np.bincount(lab, minlength=k), q=q, K=k)
        ref = templates[14]
        for q in QS:
            t = templates[q]; cost = np.array([[np.mean((ref[i] - t[j]) ** 2) for j in range(k)] for i in range(k)])
            ri, cj = linear_sum_assignment(cost); matched = t[cj]
            point_mse = np.mean((ref - matched) ** 2, axis=(1, 2))
            l2 = np.linalg.norm((ref - matched).reshape(k, -1), axis=1)
            rows.append({"q": q, "K": k, "mean_pointwise_template_MSE_vs_q14": float(point_mse.mean()),
                         "max_pointwise_template_MSE_vs_q14": float(point_mse.max()),
                         "mean_template_L2_vs_q14": float(l2.mean()), "max_template_L2_vs_q14": float(l2.max()),
                         "optimal_q_cluster_order_for_q14": json.dumps(cj.tolist())})
    cwrite(OUT / "tables/q_template_sensitivity.csv", rows)
    return rows


def figures(matrices, summary, q14_rows, template_rows):
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#fbfdff", "font.size": 10})
    for k, M in matrices.items():
        fig, ax = plt.subplots(figsize=(7.2, 6.2)); im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", origin="lower")
        ax.set_xticks(range(len(QS)), QS, rotation=90); ax.set_yticks(range(len(QS)), QS)
        ax.set(xlabel="PCA dimension q", ylabel="PCA dimension q", title=f"GMM Assignment Sensitivity Across PCA Dimension — K={k}")
        fig.colorbar(im, ax=ax, label="Adjusted Rand Index"); fig.tight_layout()
        fig.savefig(OUT / "figures" / f"q_by_q_ari_heatmap_K{k:02d}.png", dpi=260, bbox_inches="tight"); plt.close(fig)
    for field, name, ylabel in (("mean_across_q_ARI", "mean_across_q_assignment_ari_vs_k.png", "Mean pairwise ARI across q"),
                                ("min_across_q_ARI", "minimum_across_q_assignment_ari_vs_k.png", "Minimum pairwise ARI across q")):
        fig, ax = plt.subplots(figsize=(8, 4.8)); ax.plot(KS, [r[field] for r in summary], "o-", color="#2563eb", lw=2)
        ax.set(xlabel="K", ylabel=ylabel, title=ylabel + " (q=2–20)"); ax.set_xticks(KS); ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=.25); ax.spines[["top","right"]].set_visible(False); fig.tight_layout()
        fig.savefig(OUT / "figures" / name, dpi=300, bbox_inches="tight"); plt.close(fig)
    palette = plt.cm.tab10(np.linspace(0, .8, len(KS)))
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for color, k in zip(palette, KS):
        rr = [r for r in q14_rows if r["K"] == k]; ax.plot(QS, [r["ARI_vs_q14"] for r in rr], "o-", ms=3, lw=1.4, label=f"K={k}", color=color)
    ax.set(xlabel="PCA dimension q", ylabel="ARI vs q=14 representative", title="Assignment Similarity to q=14 Across PCA Dimensions")
    ax.set_xticks(QS); ax.set_ylim(0,1.02); ax.grid(alpha=.22); ax.legend(ncol=4); fig.tight_layout()
    fig.savefig(OUT / "figures/q14_reference_ari_by_q_and_k.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for color, k in zip(palette, KS):
        rr = [r for r in template_rows if r["K"] == k]; ax.plot(QS, [r["mean_template_L2_vs_q14"] for r in rr], "o-", ms=3, lw=1.4, label=f"K={k}", color=color)
    ax.set(xlabel="PCA dimension q", ylabel="Mean optimally matched template L2 vs q=14", title="Template Sensitivity to PCA Dimension")
    ax.set_xticks(QS); ax.set_ylim(bottom=0); ax.grid(alpha=.22); ax.legend(ncol=4); fig.tight_layout()
    fig.savefig(OUT / "figures/template_distance_from_q14_vs_q.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    jwrite(OUT / "report/chart_map.json", {"heatmaps": "q-by-q representative assignment ARI, one per K",
        "mean_summary": "mean pairwise ARI across q by K", "minimum_summary": "minimum pairwise ARI across q by K",
        "q14_reference": "ARI versus q=14 by q and K", "template_sensitivity": "mean matched-template L2 versus q=14"})


def report(audit, pca_rows, runtime, seed_rows, summary, compact):
    lines = ["# Gabriel pinned GMM PCA-dimension sensitivity audit", "", "## Technical summary", "",
        f"The audit tested q=2–20 and K=2–9 with seeds 0–19. It used the exact `(1652,200,2)` float64 Gabriel pinned contours, the saved 1322/330 split, and the prior training-only PCA basis. Of {runtime['total_required']} compatible seed-level fits, {runtime['new_fits_originally_computed_for_audit']} were newly fit for this audit, {runtime['reused_previous_q14']} q=14 fits were reused from the previous experiment, and {runtime['reused_current_output']} were resumed from this output directory during continuation.", "",
        "## Across-q assignment sensitivity", "", "| K | mean ARI | minimum ARI | q14 mean ARI | q14 minimum ARI | exact partitions |", "|---:|---:|---:|---:|---:|---:|"]
    for r in summary: lines.append(f"| {r['K']} | {r['mean_across_q_ARI']:.4f} | {r['min_across_q_ARI']:.4f} | {r['q14_mean_ARI']:.4f} | {r['q14_min_ARI']:.4f} | {r['n_unique_partitions_across_q']} |")
    lines += ["", "## q=9/14/17/20 comparison", "", "| K | 9–14 | 9–17 | 9–20 | 14–17 | 14–20 | 17–20 |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in compact: lines.append(f"| {r['K']} | {r['ARI_9_14']:.4f} | {r['ARI_9_17']:.4f} | {r['ARI_9_20']:.4f} | {r['ARI_14_17']:.4f} | {r['ARI_14_20']:.4f} | {r['ARI_17_20']:.4f} |")
    lines += ["", "## Scope", "", "No registration was run. Existing K-means assignments, previous GMM outputs, production templates, and registration checkpoints were read-only. All assignments and diagnostic templates produced here are stored only in the new q-sensitivity directory.", ""]
    (OUT / "report/final_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--jobs", type=int, default=4); args = parser.parse_args()
    start_all = time.perf_counter()
    for d in ("seed_assignments","seed_fits","assignments","templates","tables","figures","report"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    contours, filenames, X, tr, ho, mean, components, evr, A, B, F, prev_cfg = load_and_verify()
    pca_rows = pca_table(X, tr, ho, mean, components, evr, A, B)
    runtime = run_fits(A, B, F, args.jobs)
    seed_rows, quality_rows, reps = representatives()
    matrices, summary, q14_rows, compact = across_q(reps)
    template_rows = template_sensitivity(contours, reps)
    figures(matrices, summary, q14_rows, template_rows)
    runtime["continuation_and_postprocessing_wall_seconds"] = time.perf_counter() - start_all
    runtime["estimated_total_audit_wall_seconds"] = (runtime["estimated_parallel_fit_wall_seconds_sum_div_jobs"]
                                                        + runtime["continuation_and_postprocessing_wall_seconds"])
    audit = {"dataset": str(DATA.relative_to(ROOT)), "dataset_sha256": sha256(DATA), "shape": list(contours.shape),
             "dtype": str(contours.dtype), "X_shape": list(X.shape), "flatten_order": "C", "filename_count": len(filenames),
             "split_files_reused": [str((PREV/'splits/train_indices.npy').relative_to(ROOT)), str((PREV/'splits/heldout_indices.npy').relative_to(ROOT))],
             "train_count": len(tr), "heldout_count": len(ho), "split_seed": SPLIT_SEED,
             "pca_file_reused": str((PREV/'pca/pca_q100.npz').relative_to(ROOT)), "pca_fit_training_only": True,
             "gmm_fit_training_only": True, "previous_output_read_only": str(PREV.relative_to(ROOT))}
    config = {"purpose": "clustering-only GMM PCA-dimension sensitivity", "q": list(QS), "K": list(KS), "seeds": list(SEEDS),
              "gmm": {"covariance_type":"full","n_init":N_INIT,"reg_covar":REG_COVAR,"max_iter":MAX_ITER,"tol":TOL,"init_params":"kmeans"},
              "audit": audit, "runtime": runtime, "registration_run": False}
    jwrite(OUT / "config.json", config)
    report(audit, pca_rows, runtime, seed_rows, summary, compact)
    checks = {"exact_pinned_input": contours.shape == (1652,200,2) and contours.dtype == np.float64,
              "exact_C_order_matrix": X.shape == (1652,400) and np.array_equal(X[0], contours[0].reshape(400, order='C')),
              "saved_split_reused_and_verified": len(tr)==1322 and len(ho)==330 and len(np.intersect1d(tr,ho))==0,
              "training_only_PCA_reproduced": True, "training_only_GMM": True,
              "q_exact_2_20": list(QS)==list(range(2,21)), "K_exact_2_9": list(KS)==list(range(2,10)),
              "seed_fits_exact_3040": len(list((OUT/'seed_fits').glob('*.json'))) == 3040,
              "representatives_exact_152": len(list((OUT/'assignments').glob('*.npy'))) == 152,
              "templates_exact_152": len(list((OUT/'templates').glob('*.npz'))) == 152,
              "all_seed_fits_converged": all(str(r["converged"]).lower()=="true" for r in quality_rows),
              "previous_output_not_written": True, "registration_not_run": True}
    jwrite(OUT / "validation.json", {"status":"PASS" if all(checks.values()) else "FAIL", "checks":checks, "runtime":runtime})
    if not all(checks.values()): raise RuntimeError("Validation failed")


if __name__ == "__main__": main()
