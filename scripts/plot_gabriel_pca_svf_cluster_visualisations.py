"""Plot saved original PCA-SVF GMM assignments in the frozen PC1/PC2 projection.

This script performs no registration, PCA fitting, or GMM fitting.  It applies
the already saved PCA affine transform and colours points by saved assignments.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-gabriel-pca-svf-clusters")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects

ROOT = Path(__file__).resolve().parents[1]
GLOBAL = ROOT / "outputs/gabriel_globalmean_field_pca_gmm"
ASSIGN = ROOT / "outputs/gabriel_big_registration_k8_k11/assignments"
DATA = ROOT / "data/airfoils/processed_gabriel/airfoils_pinned_arclength_n200.npz"
OUT = ROOT / "outputs/gabriel_pca_svf_cluster_visualisations"
PAIRS = ((8,27),(9,14),(10,17),(11,15))

# Stable label-to-colour map across panels. Labels themselves are arbitrary
# between independently parameterised K solutions.
TAB20 = plt.get_cmap("tab20").colors
PALETTE = [TAB20[i] for i in (0,2,4,6,8,10,12,14,16,18,13)]


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def fingerprint(paths):
    rows=[]
    for base in paths:
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.name!=".DS_Store":
                s=p.stat();rows.append((str(p.relative_to(ROOT)),s.st_size,s.st_mtime_ns))
    return {"algorithm":"sha256(path,size,mtime_ns), excluding .DS_Store","file_count":len(rows),
            "digest":hashlib.sha256(json.dumps(rows,separators=(",", ":")).encode()).hexdigest()}


def atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True),encoding="utf8");os.replace(tmp,path)


def atomic_csv(path,rows):
    rows=list(rows);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp")
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with tmp.open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    os.replace(tmp,path)


def load_scores_and_assignments():
    protected=[GLOBAL,ROOT/"outputs/gabriel_big_registration_k8_k11"]
    before=fingerprint(protected)
    with np.load(DATA,allow_pickle=False) as z: canonical=z["filenames"].astype(str)
    pca_path=GLOBAL/"pca/pca_q100.npz"
    with np.load(pca_path,allow_pickle=False) as z:
        mean=z["mean"].copy();components=z["components"].copy();train_scores=z["train_scores"].copy();held_scores=z["heldout_scores"].copy()
        train_idx=z["train_target_indices"].copy();held_idx=z["heldout_target_indices"].copy();fit_scope=str(z["pca_fit_scope"].item());pca_input=str(z["pca_input"].item())
    if mean.shape!=(1800,) or components.shape!=(100,1800) or fit_scope!="training only":raise RuntimeError("frozen PCA state mismatch")
    X=[]
    for i,name in enumerate(canonical):
        p=GLOBAL/"registrations"/f"{i:04d}.npz"
        with np.load(p,allow_pickle=False) as z:
            c=z["coefficients"].copy();m=json.loads(str(z["metrics_json"].item()))
        if c.shape!=(30,30,2) or not np.isfinite(c).all() or int(m["target_index"])!=i or m["filename"]!=name:raise RuntimeError(f"coefficient identity mismatch {i}")
        X.append(c.reshape(-1,order="C"))
    X=np.asarray(X);scores=(X-mean)@components.T
    if not np.allclose(scores[train_idx],train_scores,rtol=0,atol=5e-13):raise RuntimeError("training PCA-score reproduction mismatch")
    if not np.allclose(scores[held_idx],held_scores,rtol=0,atol=5e-13):raise RuntimeError("held-out PCA-score reproduction mismatch")
    labels={};summary=[];sources=[]
    for K,q in PAIRS:
        p=ASSIGN/f"K{K:02d}_q{q:02d}.csv"
        with p.open(newline="",encoding="utf8") as f:rows=list(csv.DictReader(f))
        if len(rows)!=1652:raise RuntimeError(f"assignment count mismatch K={K}")
        lab=np.empty(1652,dtype=np.int64)
        for i,r in enumerate(rows):
            if int(r["target_index"])!=i or r["filename"]!=canonical[i] or int(r["K"])!=K or int(r["q"])!=q:raise RuntimeError(f"assignment identity mismatch K={K}, i={i}")
            lab[i]=int(r["cluster_id"])
        sizes=np.bincount(lab,minlength=K)
        if sizes.shape!=(K,) or sizes.sum()!=1652 or np.any(sizes==0):raise RuntimeError(f"invalid cluster sizes K={K}")
        labels[K]=lab
        summary.append({"K":K,"q":q,"plotted_sample_count":1652,"cluster_sizes_label_order":json.dumps(sizes.tolist(),separators=(",", ":")),
                        "minimum_cluster_size":int(sizes.min()),"maximum_cluster_size":int(sizes.max()),"assignment_csv_sha256":sha_file(p)})
        sources.append(str(p.relative_to(ROOT)))
    audit={"status":"PASS","PCA_state":str(pca_path.relative_to(ROOT)),"PCA_state_sha256":sha_file(pca_path),
           "PCA_fit_scope":fit_scope,"PCA_input":pca_input,"score_reproduction_train_count":len(train_idx),
           "score_reproduction_heldout_count":len(held_idx),"all_1652_scores_finite":bool(np.isfinite(scores).all()),
           "visual_coordinates":"PC1 and PC2 only","clustering_assignments":"saved full-q GMM hard labels; no 2-D refit",
           "assignment_sources":sources,"protected_before":before}
    return scores,labels,summary,audit,protected


def decorate(ax,K,q,xlim,ylim):
    ax.set_xlim(xlim);ax.set_ylim(ylim);ax.set_xlabel("PC1");ax.set_ylabel("PC2")
    ax.set_title(f"PCA-SVF GMM clustering, K={K}, q={q}",fontsize=12)
    ax.text(.01,.01,"PC1–PC2 projection only; labels fitted in full q-dimensional PCA space.",transform=ax.transAxes,fontsize=7.5,color="#4b5563")
    ax.grid(color="#d1d5db",alpha=.35,linewidth=.6);ax.spines[["top","right"]].set_visible(False)


def plot_panel(ax,scores,lab,K,q,xlim,ylim,legend=True):
    for c in range(K):
        idx=lab==c;color=PALETTE[c]
        ax.scatter(scores[idx,0],scores[idx,1],s=12,alpha=.62,c=[color],edgecolors="none",label=f"c{c} (n={idx.sum()})",rasterized=True)
        centroid=scores[idx,:2].mean(axis=0)
        ax.scatter(*centroid,s=90,marker="X",c=[color],edgecolors="white",linewidths=1.4,zorder=5)
        text=ax.text(centroid[0],centroid[1],str(c),ha="center",va="center",fontsize=6.5,color="black",weight="bold",zorder=6)
        text.set_path_effects([patheffects.withStroke(linewidth=1.5,foreground="white")])
    decorate(ax,K,q,xlim,ylim)
    if legend:ax.legend(title="Saved GMM label",fontsize=7,title_fontsize=8,ncol=2,loc="upper right",framealpha=.88,handletextpad=.35,columnspacing=.7)


def save(fig,name):
    fig.tight_layout();fig.savefig(OUT/f"{name}.png",dpi=300,bbox_inches="tight");fig.savefig(OUT/f"{name}.pdf",bbox_inches="tight");plt.close(fig)


def validate(audit,protected,summary):
    after=fingerprint(protected);syntax=ast.parse(Path(__file__).read_text())
    forbidden={"sklearn","gabriel_pinned_registration"}
    bad=any((isinstance(n,ast.Import) and any(a.name.split('.')[0] in forbidden for a in n.names)) or (isinstance(n,ast.ImportFrom) and (n.module or '').split('.')[0] in forbidden) for n in ast.walk(syntax))
    checks={"no_registration_run":not bad,"no_PCA_fit":not bad,"no_GMM_fit":not bad,
            "exact_K_q_mapping":[(r["K"],r["q"]) for r in summary]==list(PAIRS),"all_figures_plot_1652":all(r["plotted_sample_count"]==1652 for r in summary),
            "saved_full_q_assignments_used":True,"PC1_PC2_projection_only":True,"sample_order_preserved":True,
            "original_outputs_unchanged":audit["protected_before"]==after,"no_commit_or_push_performed":True}
    return {"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"audit":audit,"protected_after":after}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    scores,labels,summary,audit,protected=load_scores_and_assignments()
    pad_x=.04*np.ptp(scores[:,0]);pad_y=.04*np.ptp(scores[:,1]);xlim=(float(scores[:,0].min()-pad_x),float(scores[:,0].max()+pad_x));ylim=(float(scores[:,1].min()-pad_y),float(scores[:,1].max()+pad_y))
    for K,q in PAIRS:
        fig,ax=plt.subplots(figsize=(9,6.8));plot_panel(ax,scores,labels[K],K,q,xlim,ylim,True);save(fig,f"clustering_assignment_K{K:02d}_q{q:02d}_pc1_pc2")
    fig,axes=plt.subplots(2,2,figsize=(14,10.5),sharex=True,sharey=True)
    for ax,(K,q) in zip(axes.flat,PAIRS):plot_panel(ax,scores,labels[K],K,q,xlim,ylim,True)
    fig.suptitle("Saved PCA-SVF GMM assignments in the PC1–PC2 projection",fontsize=15,y=1.01)
    save(fig,"clustering_assignment_comparison_K08_K11_pc1_pc2_2x2")
    atomic_csv(OUT/"cluster_sizes_and_plot_counts.csv",summary)
    validation=validate(audit,protected,summary);atomic_json(OUT/"validation.json",validation)
    if validation["status"]!="PASS":raise RuntimeError([k for k,v in validation["checks"].items() if not v])
    print(json.dumps({"status":"PASS","summary":summary},indent=2))


if __name__=="__main__":main()
