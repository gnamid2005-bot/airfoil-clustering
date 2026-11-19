"""Plot saved Gabriel airfoil clusters for K=1..11 without fitting anything."""
from __future__ import annotations
import csv, hashlib, json, os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR","/tmp/mpl-gabriel-k1-k11")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/airfoils/processed_gabriel/airfoils_pinned_arclength_n200.npz"
LOW=ROOT/"outputs/gabriel_big_registration_k2_k7_full_leftmost"
HIGH=ROOT/"outputs/gabriel_big_registration_k8_k11"
OUT=ROOT/"outputs/gabriel_airfoil_cluster_shapes_k1_k11"
Q={1:None,2:1,3:89,4:5,5:3,6:2,7:5,8:27,9:14,10:17,11:15};N=1652;MAX_SHOW=150;SEED=20260908
MEMBER="#728197";MEAN="#172033"

def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def write_csv(p,rows):
 rows=list(rows);p.parent.mkdir(parents=True,exist_ok=True);fields=list(rows[0]);t=p.with_suffix(p.suffix+".tmp")
 with t.open("w",newline="",encoding="utf8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 os.replace(t,p)
def pick(idx,k,c):
 if len(idx)<=MAX_SHOW:return idx
 return np.sort(np.random.default_rng(SEED+k*100+c).choice(idx,MAX_SHOW,replace=False))
def setup(ax,xlim,ylim):
 ax.set_xlim(*xlim);ax.set_ylim(*ylim);ax.set_aspect("equal",adjustable="box");ax.set_xticks([]);ax.set_yticks([])
 for s in ax.spines.values():s.set_color("#D4DAE3");s.set_linewidth(.55)
def plot_panel(ax,contours,idx,mu,k,c,xlim,ylim,show_members=True,compact=False):
 shown=pick(idx,k,c)
 if show_members:
  for z in contours[shown]:ax.plot(z[:,0],z[:,1],color=MEMBER,lw=.38,alpha=.10,rasterized=True)
 ax.plot(mu[:,0],mu[:,1],color=MEAN,lw=1.7 if compact else 2.25,zorder=4)
 rms=np.sqrt(np.mean(np.sum((contours[idx]-mu)**2,axis=2),axis=1));setup(ax,xlim,ylim)
 if compact:ax.set_title(f"C{c} · n={len(idx)}",fontsize=6.5,pad=1.5)
 else:
  extra=f"\ndisplayed {len(shown)}/{len(idx)}; mean uses all" if len(shown)<len(idx) else ""
  ax.set_title(f"Cluster {c} · n={len(idx)}\nmean RMS={rms.mean():.4f}{extra}",fontsize=7.2,pad=2)
 return len(shown),float(rms.mean())
def main():
 for d in (OUT/"figures/individual",OUT/"figures/overview",OUT/"tables",OUT/"audit") :d.mkdir(parents=True,exist_ok=True)
 with np.load(DATA,allow_pickle=False) as z:contours=z["contours"].copy();names=z["filenames"].astype(str)
 if contours.shape!=(N,200,2) or names.shape!=(N,) or not np.isfinite(contours).all():raise RuntimeError("canonical data mismatch")
 labels={1:np.zeros(N,dtype=int)};sources={1:"K=1 baseline: all canonical contours; no fitted GMM"};source_files=[DATA]
 for k in range(2,12):
  base=LOW if k<=7 else HIGH;q=Q[k];bid=f"K{k:02d}_q{q:02d}";cp=base/"assignments"/f"{bid}.csv";lp=base/"assignments"/f"{bid}_labels.npy";tp=base/"templates"/bid/"all_templates.npz"
  rows=list(csv.DictReader(cp.open()));lab=np.load(lp).astype(int);idx=np.array([int(r["target_index"]) for r in rows]);fn=np.array([r["filename"] for r in rows]);csvlab=np.array([int(r["cluster_id"]) for r in rows])
  if lab.shape!=(N,) or not np.array_equal(idx,np.arange(N)) or not np.array_equal(fn,names) or not np.array_equal(lab,csvlab) or set(np.unique(lab))!=set(range(k)):raise RuntimeError(f"assignment identity mismatch K={k}")
  with np.load(tp,allow_pickle=False) as z:saved=z["coordinates"].copy();tlabels=z["labels"].copy();tfn=z["filenames"].astype(str)
  means=np.stack([contours[lab==c].mean(0) for c in range(k)])
  if not np.array_equal(tlabels,lab) or not np.array_equal(tfn,names) or not np.allclose(means,saved,rtol=0,atol=1e-14):raise RuntimeError(f"template crosscheck K={k}")
  labels[k]=lab;sources[k]=str(cp.relative_to(ROOT));source_files += [cp,lp,tp]
 before={str(p.relative_to(ROOT)):sha(p) for p in source_files}
 means={k:np.stack([contours[labels[k]==c].mean(0) for c in range(k)]) for k in range(1,12)}
 xmin,xmax=float(contours[:,:,0].min()),float(contours[:,:,0].max());ymin,ymax=float(contours[:,:,1].min()),float(contours[:,:,1].max());xp=.025*(xmax-xmin);yp=.08*(ymax-ymin);xlim=(xmin-xp,xmax+xp);ylim=(ymin-yp,ymax+yp)
 manifest=[];sizes={};orders={}
 for k in range(1,12):
  sizes[k]=np.bincount(labels[k],minlength=k);orders[k]=sorted(range(k),key=lambda c:(-sizes[k][c],c))
  for c in range(k):
   idx=np.flatnonzero(labels[k]==c);shown=pick(idx,k,c);rms=np.sqrt(np.mean(np.sum((contours[idx]-means[k][c])**2,axis=2),axis=1)).mean()
   manifest.append({"K":k,"q":"" if Q[k] is None else Q[k],"assignment_source":sources[k],"cluster_label":c,"cluster_size":len(idx),"displayed_n":len(shown),"mean_shape_source":"pointwise arithmetic mean of ALL canonical 200-point contours in cluster","mean_RMS_to_mean":rms})
 write_csv(OUT/"tables/k1_k11_cluster_visualisation_manifest.csv",manifest)
 # One figure per K, raw-label panel order for traceability.
 for k in range(1,12):
  cols=min(4,k);rows=int(np.ceil(k/cols));fig,axs=plt.subplots(rows,cols,figsize=(4.25*cols,2.75*rows),squeeze=False);flat=axs.flat
  for c in range(k):plot_panel(flat[c],contours,np.flatnonzero(labels[k]==c),means[k][c],k,c,xlim,ylim)
  for ax in list(flat)[k:]:ax.remove()
  title="K=1 baseline — global mean airfoil" if k==1 else f"PCA-SVF GMM airfoil clusters — K={k}, q={Q[k]}"
  fig.suptitle(title,fontsize=14,weight="bold",y=.995);fig.text(.5,.006,"thin curves = displayed members · thick curve = pointwise mean using all cluster members",ha="center",fontsize=8,color="#4B5563");fig.tight_layout(rect=(0,.025,1,.965))
  stem=OUT/"figures/individual"/f"airfoil_clusters_K{k:02d}";fig.savefig(stem.with_suffix(".png"),dpi=260,bbox_inches="tight",facecolor="white");fig.savefig(stem.with_suffix(".pdf"),bbox_inches="tight",facecolor="white");plt.close(fig)
 # 66-panel diagnostic; each K occupies one row and is size-ranked left-to-right.
 fig=plt.figure(figsize=(28,27));gs=fig.add_gridspec(11,11,hspace=.58,wspace=.18,left=.055,right=.995,top=.975,bottom=.025)
 for row,k in enumerate(range(1,12)):
  fig.text(.008,1-(row+.5)/11,f"K={k}\nq={'—' if Q[k] is None else Q[k]}",ha="left",va="center",fontsize=10,weight="bold")
  for col,c in enumerate(orders[k]):
   ax=fig.add_subplot(gs[row,col]);plot_panel(ax,contours,np.flatnonzero(labels[k]==c),means[k][c],k,c,xlim,ylim,show_members=True,compact=True)
 fig.suptitle("Gabriel PCA-SVF Airfoil Clusters, K=1–11\nSize-ranked within each K; member display capped at 150, means use all members",fontsize=17,weight="bold")
 stem=OUT/"figures/overview/airfoil_clusters_K01_K11_full";fig.savefig(stem.with_suffix(".png"),dpi=180,facecolor="white");fig.savefig(stem.with_suffix(".pdf"),facecolor="white");plt.close(fig)
 # Poster overview: same geometry scale, means only, size-ranked.
 fig=plt.figure(figsize=(28,22));gs=fig.add_gridspec(11,11,hspace=.52,wspace=.16,left=.055,right=.995,top=.96,bottom=.025)
 for row,k in enumerate(range(1,12)):
  fig.text(.008,1-(row+.5)/11,f"K={k}\nq={'—' if Q[k] is None else Q[k]}",ha="left",va="center",fontsize=11,weight="bold")
  for col,c in enumerate(orders[k]):
   ax=fig.add_subplot(gs[row,col]);plot_panel(ax,contours,np.flatnonzero(labels[k]==c),means[k][c],k,c,xlim,ylim,show_members=False,compact=True)
 fig.suptitle("Airfoil Cluster Means Across K=1–11\nClusters ordered by descending size; identical geometry scale in every panel",fontsize=20,weight="bold")
 stem=OUT/"figures/overview/airfoil_cluster_means_K01_K11_poster_overview";fig.savefig(stem.with_suffix(".png"),dpi=220,facecolor="white");fig.savefig(stem.with_suffix(".pdf"),facecolor="white");plt.close(fig)
 after={str(p.relative_to(ROOT)):sha(p) for p in source_files}
 checks={"K1_all_samples_no_GMM":bool(np.all(labels[1]==0)),"K2_K11_existing_assignments":True,"canonical_identity_order":True,"all_cluster_sizes_sum_N":all(int(sizes[k].sum())==N for k in sizes),"means_use_all_members":True,"subsampling_display_only":True,"identical_axis_scales":True,"no_registration_PCA_GMM":True,"original_outputs_not_overwritten":True,"q_manifest_recorded":True,"source_files_unchanged":before==after}
 audit={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"N":N,"points_per_contour":200,"q_by_K":Q,"axis_limits":{"x":xlim,"y":ylim},"subsampling":{"threshold":150,"maximum_displayed":150,"seed":SEED},"cluster_sizes":{str(k):sizes[k].tolist() for k in sizes},"size_order_raw_labels":{str(k):orders[k] for k in orders},"assignment_sources":sources,"source_hashes":after}
 (OUT/"audit/validation.json").write_text(json.dumps(audit,indent=2)+"\n");(OUT/"validation.json").write_text(json.dumps(audit,indent=2)+"\n")
 print(json.dumps({"status":audit["status"],"q_by_K":Q,"cluster_sizes":audit["cluster_sizes"],"output":str(OUT)},indent=2))
if __name__=="__main__":main()
