"""Add all saved cluster members to the frozen K=1..11 poster overview."""
from __future__ import annotations
import csv, hashlib, json, os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR","/tmp/mpl-gabriel-k1-k11-members")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/airfoils/processed_gabriel/airfoils_pinned_arclength_n200.npz"
LOW=ROOT/"outputs/gabriel_big_registration_k2_k7_full_leftmost"
HIGH=ROOT/"outputs/gabriel_big_registration_k8_k11"
OUT=ROOT/"outputs/gabriel_airfoil_cluster_shapes_k1_k11"
OLD=OUT/"figures/overview/airfoil_cluster_means_K01_K11_poster_overview.png"
PREVIOUS=OUT/"figures/overview/airfoil_cluster_means_K01_K11_with_members.png"
Q={1:None,2:1,3:89,4:5,5:3,6:2,7:5,8:27,9:14,10:17,11:15};N=1652
MEMBER_COLOR="#D62728";MEMBER_ALPHA=.040;MEMBER_LW=.30;MEAN_COLOR="#1565C0";MEAN_LW=2.05

def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def main():
 old_hash=sha(OLD);previous_hash=sha(PREVIOUS)
 with np.load(DATA,allow_pickle=False) as z:contours=z["contours"].copy();names=z["filenames"].astype(str)
 labels={1:np.zeros(N,dtype=int)};source_files=[DATA]
 for k in range(2,12):
  q=Q[k];base=LOW if k<=7 else HIGH;bid=f"K{k:02d}_q{q:02d}";cp=base/"assignments"/f"{bid}.csv";lp=base/"assignments"/f"{bid}_labels.npy";tp=base/"templates"/bid/"all_templates.npz"
  rows=list(csv.DictReader(cp.open()));lab=np.load(lp).astype(int)
  if len(rows)!=N or not np.array_equal(np.array([int(r["target_index"]) for r in rows]),np.arange(N)) or not np.array_equal(np.array([r["filename"] for r in rows]),names) or not np.array_equal(np.array([int(r["cluster_id"]) for r in rows]),lab):raise RuntimeError(f"identity mismatch K={k}")
  labels[k]=lab;source_files += [cp,lp,tp]
 before={str(p.relative_to(ROOT)):sha(p) for p in source_files};means={};sizes={};orders={}
 for k in range(1,12):
  sizes[k]=np.bincount(labels[k],minlength=k);orders[k]=sorted(range(k),key=lambda c:(-sizes[k][c],c));means[k]=np.stack([contours[labels[k]==c].mean(0) for c in range(k)])
  if k>=2:
   q=Q[k];base=LOW if k<=7 else HIGH
   with np.load(base/"templates"/f"K{k:02d}_q{q:02d}"/"all_templates.npz",allow_pickle=False) as z:saved=z["coordinates"].copy()
   if not np.allclose(means[k],saved,rtol=0,atol=1e-14):raise RuntimeError(f"mean drift K={k}")
 # Reuse exact frozen axis limits and size order from the prior validation.
 prior=json.loads((OUT/"validation.json").read_text());xlim=tuple(prior["axis_limits"]["x"]);ylim=tuple(prior["axis_limits"]["y"])
 if any(orders[k]!=prior["size_order_raw_labels"][str(k)] for k in orders):raise RuntimeError("cluster ordering drift")
 fig=plt.figure(figsize=(28,22));gs=fig.add_gridspec(11,11,hspace=.52,wspace=.16,left=.055,right=.995,top=.96,bottom=.025)
 display=[]
 for row,k in enumerate(range(1,12)):
  fig.text(.008,1-(row+.5)/11,f"K={k}\nq={'—' if Q[k] is None else Q[k]}",ha="left",va="center",fontsize=11,weight="bold")
  for col,c in enumerate(orders[k]):
   ax=fig.add_subplot(gs[row,col]);idx=np.flatnonzero(labels[k]==c)
   for z in contours[idx]:ax.plot(z[:,0],z[:,1],color=MEMBER_COLOR,alpha=MEMBER_ALPHA,lw=MEMBER_LW,rasterized=True,zorder=1)
   mu=means[k][c];ax.plot(mu[:,0],mu[:,1],color=MEAN_COLOR,alpha=1,lw=MEAN_LW,zorder=4)
   ax.set_xlim(*xlim);ax.set_ylim(*ylim);ax.set_aspect("equal",adjustable="box");ax.set_xticks([]);ax.set_yticks([]);ax.set_title(f"C{c} · n={len(idx)}",fontsize=6.5,pad=1.5)
   for s in ax.spines.values():s.set_color("#D4DAE3");s.set_linewidth(.55)
   display.append({"K":k,"q":"" if Q[k] is None else Q[k],"raw_cluster_label":c,"cluster_size":len(idx),"displayed_n":len(idx),"display_mode":"all members","member_alpha":MEMBER_ALPHA,"member_linewidth":MEMBER_LW,"mean_linewidth":MEAN_LW})
 fig.suptitle("Airfoil Cluster Means Across K=1–11\nClusters ordered by descending size; identical geometry scale in every panel",fontsize=20,weight="bold")
 legend=[Line2D([0],[0],color=MEAN_COLOR,lw=MEAN_LW,label="Cluster mean"),Line2D([0],[0],color=MEMBER_COLOR,lw=.8,alpha=.55,label="Individual airfoils")]
 fig.legend(handles=legend,loc="lower center",bbox_to_anchor=(.5,.002),ncol=2,frameon=False,fontsize=10,handlelength=3)
 stem=OUT/"figures/overview/airfoil_cluster_means_K01_K11_blue_mean_red_members"
 fig.savefig(stem.with_suffix(".png"),dpi=220,facecolor="white");fig.savefig(stem.with_suffix(".pdf"),facecolor="white");fig.savefig(stem.with_suffix(".svg"),facecolor="white");plt.close(fig)
 # New supporting manifest; no prior table is overwritten.
 p=OUT/"tables/k1_k11_blue_mean_red_members_display_manifest.csv";fields=list(display[0]);tmp=p.with_suffix(".csv.tmp")
 with tmp.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(display)
 os.replace(tmp,p)
 after={str(p.relative_to(ROOT)):sha(p) for p in source_files};checks={"assignments_unchanged":before==after,"cluster_sizes_unchanged":all(sizes[k].tolist()==prior["cluster_sizes"][str(k)] for k in sizes),"cluster_order_unchanged":all(orders[k]==prior["size_order_raw_labels"][str(k)] for k in orders),"means_numerically_identical":True,"means_use_all_members":True,"all_members_displayed":all(r["displayed_n"]==r["cluster_size"] for r in display),"identical_axes":True,"K_q_labels_correct":Q=={int(k):(None if v is None else int(v)) for k,v in prior["q_by_K"].items()},"only_visual_style_and_legend_changed":True,"no_scientific_recomputation":True,"original_overview_not_overwritten":sha(OLD)==old_hash,"previous_with_members_not_overwritten":sha(PREVIOUS)==previous_hash}
 audit={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"style":{"member_color":MEMBER_COLOR,"member_alpha":MEMBER_ALPHA,"member_linewidth":MEMBER_LW,"mean_color":MEAN_COLOR,"mean_alpha":1.0,"mean_linewidth":MEAN_LW},"axis_limits":{"x":xlim,"y":ylim},"all_members_displayed":True,"outputs":[str(stem.with_suffix(x)) for x in (".png",".pdf",".svg")]}
 (OUT/"audit/blue_mean_red_members_validation.json").write_text(json.dumps(audit,indent=2)+"\n")
 print(json.dumps(audit,indent=2))
if __name__=="__main__":main()
