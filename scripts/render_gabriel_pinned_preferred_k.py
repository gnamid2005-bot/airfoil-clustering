"""Render preferred-K curves from the complete existing pinned GMM grid.

No models are fit and no registration code is imported.
"""
from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-gabriel-preferred-k")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/gabriel_pinned_gmm_vs_kmeans"
GRID = OUT / "tables/pca_gmm_model_selection.csv"
PCA = OUT / "pca/pca_study_q1_q100.csv"
OLD = ROOT / "outputs/airfoil_shape_pca_gmm_selection/gmm/preferred_k_by_q.csv"


def read(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(path, rows, fields):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    tmp.replace(path)


def runs(values):
    result = []
    start = 1
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[i - 1]:
            result.append({"q_start": start, "q_end": i, "K": int(values[i - 1]), "length": i - start + 1})
            start = i + 1
    return result


def main():
    grid, pca = read(GRID), read(PCA)
    pairs = [(int(r["q"]), int(r["K"])) for r in grid]
    valid = (len(grid) == 2000 and len(set(pairs)) == 2000 and set(pairs) == {(q, k) for q in range(1,101) for k in range(1,21)}
             and all(r["converged"].lower() == "true" for r in grid)
             and all(math.isfinite(float(r[x])) for r in grid for x in ("BIC","ICL","heldout_log_likelihood_per_sample")))
    if not valid: raise RuntimeError("STOP: existing pinned q1..100 x K1..20 grid is incomplete or invalid")
    if len(pca) != 100 or [int(r["q"]) for r in pca] != list(range(1,101)):
        raise RuntimeError("STOP: pinned PCA table is incomplete")
    preferred = []
    for q in range(1,101):
        rr = [r for r in grid if int(r["q"]) == q]
        preferred.append({"q":q, "cumulative_explained_variance":float(pca[q-1]["cumulative_explained_variance"]),
                          "K_BIC":int(min(rr,key=lambda r:float(r["BIC"]))["K"]),
                          "K_ICL":int(min(rr,key=lambda r:float(r["ICL"]))["K"]),
                          "K_Heldout_LL":int(max(rr,key=lambda r:float(r["heldout_log_likelihood_per_sample"]))["K"])})
    write(OUT/"tables/preferred_k_by_q_gabriel_pinned.csv", preferred,
          ["q","cumulative_explained_variance","K_BIC","K_ICL","K_Heldout_LL"])
    left = []
    for k in range(2,10):
        hits = [r for r in preferred if r["K_BIC"] == k or r["K_ICL"] == k]
        if hits:
            r = hits[0]; b = r["K_BIC"] == k; i = r["K_ICL"] == k
            criterion = "both" if b and i else ("BIC" if b else "ICL")
            left.append({"K":k,"q_leftmost":r["q"],"first_criterion":criterion,"BIC_selected":b,"ICL_selected":i})
        else:
            left.append({"K":k,"q_leftmost":"","first_criterion":"never selected","BIC_selected":False,"ICL_selected":False})
    write(OUT/"tables/leftmost_q_by_k_gabriel_pinned.csv", left,
          ["K","q_leftmost","first_criterion","BIC_selected","ICL_selected"])

    colors = {"BIC":"#2563eb", "ICL":"#d97706", "Held-out LL":"#6b7280"}
    markers = {"BIC":"o", "ICL":"s", "Held-out LL":"^"}
    fields = {"BIC":"K_BIC", "ICL":"K_ICL", "Held-out LL":"K_Heldout_LL"}
    q = np.arange(1,101)
    fig, ax = plt.subplots(figsize=(12,5.5))
    for label in fields:
        ax.plot(q,[r[fields[label]] for r in preferred],marker=markers[label],ms=3,lw=1.3,label=label,color=colors[label])
    ax.set(xlabel="Shape-PCA dimension q",ylabel="Preferred GMM component count K",
           title="Preferred GMM Component Count Across Gabriel-Pinned SHAPE-SPACE PCA Dimensions")
    ax.set_ylim(.5,20.5); ax.set_yticks(range(1,21)); ax.grid(alpha=.24); ax.legend(); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT/"figures/preferred_k_gabriel_pinned_q1_q100.png",dpi=300,bbox_inches="tight"); plt.close(fig)

    old = read(OLD)
    if len(old) != 100 or [int(r["q"]) for r in old] != list(range(1,101)):
        raise RuntimeError("STOP: old direct-shape sequence incomplete")
    old_fields = {"BIC":"K_BIC", "ICL":"K_ICL", "Held-out LL":"K_heldout_LL"}
    fig, axes = plt.subplots(2,1,figsize=(12,9),sharex=True,sharey=True)
    for ax,title,rows,fmap in [(axes[0],"OLD direct-shape representation (1654 samples)",old,old_fields),
                               (axes[1],"Gabriel pinned representation (1652 samples)",preferred,fields)]:
        for label in fmap:
            ax.plot(q,[int(r[fmap[label]]) for r in rows],marker=markers[label],ms=2.7,lw=1.2,label=label,color=colors[label])
        ax.set_title(title); ax.set_ylabel("Preferred K"); ax.set_ylim(.5,20.5); ax.set_yticks(range(1,21)); ax.grid(alpha=.22)
        ax.spines[["top","right"]].set_visible(False); ax.legend(ncol=3)
    axes[1].set_xlabel("Shape-PCA dimension q")
    fig.suptitle("Preferred GMM Component Count: Direct Shape vs Gabriel-Pinned Shape",fontsize=15,y=.995)
    fig.tight_layout(); fig.savefig(OUT/"figures/preferred_k_direct_vs_gabriel_pinned.png",dpi=300,bbox_inches="tight"); plt.close(fig)

    sequences = {key:[r[field] for r in preferred] for key,field in fields.items()}
    old_sequences = {key:[int(r[field]) for r in old] for key,field in old_fields.items()}
    diagnostics = {"status":"PASS","grid_cells_reused":2000,"grid_cells_refit":0,
        "pinned_sequences":sequences,"old_direct_sequences":old_sequences,
        "pinned_frequencies":{k:dict(Counter(v)) for k,v in sequences.items()},
        "old_direct_frequencies":{k:dict(Counter(v)) for k,v in old_sequences.items()},
        "pinned_runs":{k:runs(v) for k,v in sequences.items()},
        "bic_icl_agree_q":[r["q"] for r in preferred if r["K_BIC"]==r["K_ICL"]],
        "heldout_disagrees_with_both_q":[r["q"] for r in preferred if r["K_Heldout_LL"] != r["K_BIC"] and r["K_Heldout_LL"] != r["K_ICL"]],
        "leftmost":left,"registration_run":False}
    (OUT/"tables/preferred_k_gabriel_pinned_validation.json").write_text(json.dumps(diagnostics,indent=2,sort_keys=True),encoding="utf-8")


if __name__ == "__main__": main()
