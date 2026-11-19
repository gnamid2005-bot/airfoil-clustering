#!/usr/bin/env python3
"""Cross-K, same-q analysis using existing Task-2 PCA-SVF assignments only."""
from __future__ import annotations
import csv, json, os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-gabriel-cross-k")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"outputs/gabriel_task2_assignment_consistency_refit"
OUT=ROOT/"outputs/gabriel_cross_k_assignment_stability"
KS=(8,10,11); PAIRS=((8,10),(8,11),(10,11)); QS=range(30,101)
COLORS={(8,10):"#2563EB",(8,11):"#D97706",(10,11):"#7C3AED"}
STYLES={(8,10):"-",(8,11):"--",(10,11):"-."}

def write_csv(path, rows):
    rows=list(rows);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def write_json(path,x):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(x,indent=2,sort_keys=True),encoding="utf8")
def load():
    source_validation=json.loads((SRC/"validation.json").read_text())
    if source_validation["status"]!="PASS" or source_validation["GMM_cells_completed"]!=400:raise RuntimeError("Task-2 source validation failed")
    out={}; identity=None
    for K in KS:
        p=SRC/f"assignments/pca_svf_assignments_K{K:02d}.npz"
        with np.load(p,allow_pickle=False) as z:
            q=z["q"].copy();labels=z["assignments"].copy();idx=z["target_indices"].copy();names=z["filenames"].astype(str);storedK=int(z["K"])
        if storedK!=K or labels.shape!=(100,1651) or not np.array_equal(q,np.arange(1,101)):raise RuntimeError(f"K/q/shape mismatch K={K}")
        current=(idx,names)
        if identity is None:identity=current
        elif not np.array_equal(idx,identity[0]) or not np.array_equal(names,identity[1]):raise RuntimeError("sample identity/order mismatch")
        out[K]=labels
    if len(np.unique(identity[0]))!=1651 or len(np.unique(identity[1]))!=1651:raise RuntimeError("duplicate sample identity")
    return out,identity
def contingency(a,b,ka,kb):
    C=np.zeros((ka,kb),int)
    for x,y in zip(a,b):C[int(x),int(y)]+=1
    return C
def main():
    for d in (OUT/"tables",OUT/"figures",OUT/"report"):d.mkdir(parents=True,exist_ok=True)
    labs,(idx,names)=load();rows=[]
    for q in QS:
        row={"q":q}
        for a,b in PAIRS:
            A,B=labs[a][q-1],labs[b][q-1]
            row[f"ari_k{a}_k{b}"]=adjusted_rand_score(A,B)
            row[f"ami_k{a}_k{b}"]=adjusted_mutual_info_score(A,B)
        rows.append(row)
    write_csv(OUT/"tables/cross_k_ari_q30_q100.csv",rows)
    summary=[]
    for a,b in PAIRS:
        v=np.array([r[f"ari_k{a}_k{b}"] for r in rows]);
        summary.append({"pair":f"K={a} vs K={b}","K_left":a,"K_right":b,"q_start":30,"q_end":100,"n_q":len(v),"mean_ARI":v.mean(),"median_ARI":np.median(v),"minimum_ARI":v.min(),"q_at_minimum":int(rows[int(v.argmin())]["q"]),"maximum_ARI":v.max(),"q_at_maximum":int(rows[int(v.argmax())]["q"]),"std_ARI_population":v.std(ddof=0),"range_ARI":v.max()-v.min(),"mean_absolute_adjacent_change":np.mean(np.abs(np.diff(v))),"maximum_absolute_adjacent_change":np.max(np.abs(np.diff(v)))})
    write_csv(OUT/"tables/cross_k_ari_summary.csv",summary)
    snaps=[]
    for q in (30,40,50,75,100):
        r=rows[q-30];snaps.append({"q":q,**{f"ari_k{a}_k{b}":r[f"ari_k{a}_k{b}"] for a,b in PAIRS}})
    write_csv(OUT/"tables/fixed_q_snapshots.csv",snaps)
    details=[]
    for q in (30,50):
        for a,b in PAIRS:
            C=contingency(labs[a][q-1],labs[b][q-1],a,b);rr,cc=linear_sum_assignment(-C)
            # Each row records one source cluster and its distribution across the higher-K partition.
            for i in range(a):
                top=np.argsort(C[i])[::-1]; total=C[i].sum(); probs=C[i]/total
                details.append({"q":q,"pair":f"K={a} vs K={b}","lower_K_cluster":i,"lower_cluster_size":int(total),"largest_overlapping_higher_cluster":int(top[0]),"largest_overlap_count":int(C[i,top[0]]),"largest_overlap_fraction":float(probs[top[0]]),"second_largest_overlap_fraction":float(probs[top[1]]) if b>1 else 0.0,"n_higher_clusters_with_at_least_10pct":int(np.sum(probs>=.1))})
            np.savetxt(OUT/f"tables/contingency_q{q:03d}_K{a:02d}_K{b:02d}.csv",C,delimiter=",",fmt="%d")
    write_csv(OUT/"tables/cluster_overlap_detail_q30_q50.csv",details)
    # Main line plot: focused y scale is explicitly stated in subtitle.
    fig,ax=plt.subplots(figsize=(10.5,6.2));
    for a,b in PAIRS:
        v=[r[f"ari_k{a}_k{b}"] for r in rows];ax.plot(list(QS),v,color=COLORS[a,b],ls=STYLES[a,b],lw=2.2,label=f"K={a} vs K={b}")
    ax.axvline(30,color="#374151",lw=1.4,ls=":");ax.text(30.8,.985,"approx. post-stability region begins",va="top",color="#374151",fontsize=9)
    ax.set(title="Cross-K PCA-SVF Assignment Similarity After Stability",xlabel="PCA dimension q",ylabel="Adjusted Rand Index (ARI)",xlim=(29.5,100.5),ylim=(0,1));ax.grid(axis="y",color="#D1D5DB",alpha=.7);ax.legend(frameon=False,ncol=3,loc="lower right");ax.text(0,-.17,"q=30 is an empirical approximate boundary; N=1,651; same-q comparisons only.",transform=ax.transAxes,fontsize=9,color="#4B5563");fig.tight_layout()
    fig.savefig(OUT/"figures/cross_k_ari_post_stability.png",dpi=300,bbox_inches="tight");fig.savefig(OUT/"figures/cross_k_ari_post_stability.pdf",bbox_inches="tight");plt.close(fig)
    M=np.eye(3)
    for a,b in PAIRS:
        val=next(x["mean_ARI"] for x in summary if x["K_left"]==a and x["K_right"]==b);i=KS.index(a);j=KS.index(b);M[i,j]=M[j,i]=val
    fig,ax=plt.subplots(figsize=(6.4,5.5));im=ax.imshow(M,vmin=0,vmax=1,cmap="Blues");ax.set_xticks(range(3),[f"K={k}" for k in KS]);ax.set_yticks(range(3),[f"K={k}" for k in KS]);ax.set_title("Mean Cross-K ARI, q=30–100",pad=14)
    for i in range(3):
        for j in range(3):ax.text(j,i,f"{M[i,j]:.3f}",ha="center",va="center",color="white" if M[i,j]>.55 else "#111827",fontweight="bold")
    fig.colorbar(im,ax=ax,label="Mean ARI");fig.tight_layout();fig.savefig(OUT/"figures/mean_cross_k_ari_heatmap.png",dpi=300,bbox_inches="tight");plt.close(fig)
    validation={"status":"PASS","source":str(SRC.relative_to(ROOT)),"source_assignments_already_available":True,"K_values":list(KS),"q_min":30,"q_max":100,"q_count":71,"same_q_comparisons_only":True,"sample_count":1651,"target_indices_identical":True,"filenames_identical_in_exact_order":True,"all_labels_in_expected_ranges":all(np.all((labs[k]>=0)&(labs[k]<k)) for k in KS),"ARI":"sklearn.metrics.adjusted_rand_score (label-permutation invariant)","registration_run":False,"template_registration_run":False,"GMM_fit_run":False,"PCA_fit_run":False,"model_selection_changed":False,"existing_scientific_output_overwritten":False,"commit_or_push":False}
    write_json(OUT/"validation.json",validation)
    config={"analysis":"same-q cross-K PCA-SVF assignment similarity","source":str(SRC.relative_to(ROOT)),"K_values":list(KS),"pairs":[list(x) for x in PAIRS],"q_range":[30,100],"q30_interpretation":"approximate empirical start of post-stability region, not a theoretical threshold","primary_metric":"ARI","secondary_metric":"AMI","sample_count":1651,"sample_identity":"target_indices and filenames verified identical across all assignment files"};write_json(OUT/"config.json",config)
    means={x["pair"]:x["mean_ARI"] for x in summary};best=max(means,key=means.get);worst=min(means,key=means.get)
    split_stats={"mean_rows_with_multiple_ge10pct":float(np.mean([r["n_higher_clusters_with_at_least_10pct"]>=2 for r in details])),"median_largest_overlap_fraction":float(np.median([r["largest_overlap_fraction"] for r in details]))}
    report=["# Gabriel：稳定区后的跨 K assignment 比较","","## 结论","",f"q=30–100 内，最高平均相似度为 **{best}**（ARI={means[best]:.4f}），最低为 **{worst}**（ARI={means[worst]:.4f}）。这与 within-K adjacent-q 稳定性是不同问题：固定 K 随 q 稳定，并不意味着不同 K 产生相同分区。","","q=30 是依据既有 adjacent-q 图选择的经验性近似边界，不是理论阈值。全部比较使用同一 q、同一顺序的 1,651 个样本。","","## 汇总","","| Pair | Mean | Median | Min (q) | Max (q) | SD | Mean absolute adjacent-q change |","|---|---:|---:|---:|---:|---:|---:|"]
    for x in summary:report.append(f"| {x['pair']} | {x['mean_ARI']:.4f} | {x['median_ARI']:.4f} | {x['minimum_ARI']:.4f} ({x['q_at_minimum']}) | {x['maximum_ARI']:.4f} ({x['q_at_maximum']}) | {x['std_ARI_population']:.4f} | {x['mean_absolute_adjacent_change']:.4f} |")
    report += ["","## 分区关系解释","",f"q=30 与 q=50 的低 K→高 K contingency 显示，{100*split_stats['mean_rows_with_multiple_ge10pct']:.1f}% 的低-K簇在至少两个高-K簇中各有≥10%的成员；单个最大重叠簇的中位覆盖率为 {100*split_stats['median_largest_overlap_fraction']:.1f}%。因此存在明显的簇细分/拆分，但并非纯粹层级嵌套：中等 ARI 与多簇重叠也表明发生了实质性的成员重新分配。ARI 本身不能证明层级关系。","","## 安全与验证","","未运行 registration、template registration、PCA 或 GMM；未改变模型选择规则；未覆盖任何既有科学输出；未 commit/push。"]
    (OUT/"report/cross_k_assignment_stability_report.md").write_text("\n".join(report)+"\n",encoding="utf8")
    print(json.dumps({"summary":summary,"split_stats":split_stats,"validation":validation},indent=2))
if __name__=="__main__":main()
