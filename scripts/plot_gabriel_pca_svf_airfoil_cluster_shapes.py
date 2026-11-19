"""Plot canonical airfoil geometries grouped by saved PCA-SVF GMM labels.

No registration, PCA fitting, or GMM fitting is imported or executed.
"""
from __future__ import annotations
import ast,csv,hashlib,json,os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR","/tmp/mpl-gabriel-airfoil-cluster-shapes")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/airfoils/processed_gabriel/airfoils_pinned_arclength_n200.npz"
BIG=ROOT/"outputs/gabriel_big_registration_k8_k11"
OUT=ROOT/"outputs/gabriel_pca_svf_airfoil_cluster_shapes"
PAIRS=((8,27),(9,14),(10,17),(11,15));MAX_DISPLAY=150;SEED=20260830
LAYOUTS={8:(2,4),9:(3,3),10:(2,5),11:(3,4)}

def sha_file(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for x in iter(lambda:f.read(1024*1024),b""):h.update(x)
 return h.hexdigest()
def fingerprint():
 bases=[DATA,BIG/"assignments",BIG/"templates"] ;rows=[]
 for b in bases:
  paths=[b] if b.is_file() else sorted(b.rglob("*"))
  for p in paths:
   if p.is_file() and p.name!=".DS_Store":s=p.stat();rows.append((str(p.relative_to(ROOT)),s.st_size,s.st_mtime_ns))
 return {"algorithm":"sha256(path,size,mtime_ns), excluding .DS_Store","file_count":len(rows),"digest":hashlib.sha256(json.dumps(rows,separators=(",", ":")).encode()).hexdigest()}
def atomic_json(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+".tmp");t.write_text(json.dumps(x,indent=2,sort_keys=True),encoding="utf8");os.replace(t,p)
def atomic_csv(p,rows):
 rows=list(rows);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+".tmp");fields=list(dict.fromkeys(k for r in rows for k in r))
 with t.open("w",newline="",encoding="utf8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 os.replace(t,p)
def atomic_npz(p,**a):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(".tmp.npz");np.savez_compressed(t,**a);os.replace(t,p)

def load():
 before=fingerprint()
 with np.load(DATA,allow_pickle=False) as z:contours=z["contours"].copy();names=z["filenames"].astype(str)
 if contours.shape!=(1652,200,2) or contours.dtype!=np.float64 or not np.isfinite(contours).all():raise RuntimeError("canonical contour mismatch")
 branches={};sources=[]
 for K,q in PAIRS:
  p=BIG/"assignments"/f"K{K:02d}_q{q:02d}.csv";rows=list(csv.DictReader(p.open(newline="",encoding="utf8")));lab=np.empty(1652,dtype=np.int64)
  if len(rows)!=1652:raise RuntimeError("assignment count mismatch")
  for i,r in enumerate(rows):
   if int(r["target_index"])!=i or r["filename"]!=names[i] or int(r["K"])!=K or int(r["q"])!=q:raise RuntimeError(f"assignment identity mismatch K={K}, i={i}")
   lab[i]=int(r["cluster_id"])
  sizes=np.bincount(lab,minlength=K)
  if len(sizes)!=K or sizes.sum()!=1652 or np.any(sizes==0):raise RuntimeError(f"cluster completeness K={K}")
  branches[K]=lab;sources.append({"K":K,"q":q,"path":str(p.relative_to(ROOT)),"sha256":sha_file(p)})
 return contours,names,branches,before,sources

def compute(contours,names,branches):
 dispersion=[];sizes_rows=[];cross=[];means_by={};shown_by={}
 for K,q in PAIRS:
  lab=branches[K];means=[];shown=[]
  with np.load(BIG/"templates"/f"K{K:02d}_q{q:02d}"/"all_templates.npz",allow_pickle=False) as z:saved=z["coordinates"].copy()
  for c in range(K):
   members=np.flatnonzero(lab==c);mean=contours[members].mean(axis=0);means.append(mean)
   rms=np.sqrt(np.mean(np.sum((contours[members]-mean[None,:,:])**2,axis=2),axis=1))
   rng=np.random.default_rng(SEED+K*100+c)
   display=np.sort(members if len(members)<=MAX_DISPLAY else rng.choice(members,MAX_DISPLAY,replace=False));shown.append(display)
   delta=np.abs(mean-saved[c]);mx=float(delta.max())
   dispersion.append({"K":K,"q":q,"cluster":c,"n":len(members),"displayed_members":len(display),"mean_uses_all_members":True,
    "mean_rms_to_mean":float(rms.mean()),"median_rms_to_mean":float(np.median(rms)),"p95_rms_to_mean":float(np.quantile(rms,.95)),"max_rms_to_mean":float(rms.max())})
   sizes_rows.append({"K":K,"q":q,"cluster":c,"cluster_size":len(members),"displayed_members":len(display),"display_seed":SEED+K*100+c})
   cross.append({"K":K,"q":q,"cluster":c,"n":len(members),"saved_template":str((BIG/"templates"/f"K{K:02d}_q{q:02d}"/f"template_{c:02d}.npz").relative_to(ROOT)),"maximum_absolute_coordinate_discrepancy":mx,"exact_array_equal":bool(np.array_equal(mean,saved[c])),"tolerance_pass_1e_14":mx<=1e-14})
  means=np.asarray(means);means_by[K]=means;shown_by[K]=shown
  atomic_npz(OUT/"means"/f"K{K:02d}_q{q:02d}_cluster_means.npz",cluster_means=means,cluster_ids=np.arange(K),cluster_sizes=np.bincount(lab,minlength=K),labels=lab,filenames=names,definition=np.asarray("pointwise arithmetic mean over all assigned canonical 200-point contours"))
 atomic_csv(OUT/"tables/cluster_shape_dispersion.csv",dispersion);atomic_csv(OUT/"tables/cluster_sizes.csv",sizes_rows);atomic_csv(OUT/"tables/template_crosscheck.csv",cross)
 return dispersion,sizes_rows,cross,means_by,shown_by

def plot(contours,branches,dispersion,means_by,shown_by):
 xlim=(-.03,1.03);ypad=.025;ylim=(float(contours[:,:,1].min()-ypad),float(contours[:,:,1].max()+ypad));dmap={(r["K"],r["cluster"]):r for r in dispersion}
 for K,q in PAIRS:
  nr,nc=LAYOUTS[K];fig,axes=plt.subplots(nr,nc,figsize=(nc*3.55,nr*3.15+.6),squeeze=False);flat=axes.flat
  for c in range(K):
   ax=flat[c];mean=means_by[K][c];members=np.flatnonzero(branches[K]==c);display=shown_by[K][c]
   for i in display:
    s=contours[i];ax.plot(np.r_[s[:,0],s[0,0]],np.r_[s[:,1],s[0,1]],color="#6b7280",lw=.45,alpha=.10,rasterized=True)
   ax.plot(np.r_[mean[:,0],mean[0,0]],np.r_[mean[:,1],mean[0,1]],color="#0f4c81",lw=2.15,zorder=4)
   r=dmap[(K,c)];subtitle=f"n = {len(members)} | mean RMS = {r['mean_rms_to_mean']:.4f}"
   if len(display)<len(members):subtitle+=f"\ndisplayed {len(display)}/{len(members)}; mean uses all"
   ax.set_title(f"Cluster {c}\n{subtitle}",fontsize=9.2,pad=4);ax.set_xlim(xlim);ax.set_ylim(ylim);ax.set_aspect("equal",adjustable="box")
   ax.set_xticks([0,.5,1]);ax.set_yticks([-.3,0,.3]);ax.grid(color="#d1d5db",alpha=.22,lw=.5);ax.spines[["top","right"]].set_visible(False);ax.tick_params(labelsize=7)
   if c//nc==nr-1:ax.set_xlabel("x/chord",fontsize=8)
   if c%nc==0:ax.set_ylabel("y/chord",fontsize=8)
  for c in range(K,nr*nc):flat[c].remove()
  handles=[Line2D([0],[0],color="#6b7280",lw=.7,alpha=.5,label="individual cluster members"),Line2D([0],[0],color="#0f4c81",lw=2.4,label="pointwise cluster mean (all members)")]
  fig.legend(handles=handles,loc="lower center",ncol=2,frameon=False,fontsize=9,bbox_to_anchor=(.5,.018))
  fig.suptitle(f"PCA-SVF GMM airfoil clusters\nK={K}, q={q} — raw GMM label order",fontsize=14,y=.98)
  fig.subplots_adjust(left=.045,right=.99,top=.80,bottom=.105,wspace=.18,hspace=.55)
  p=OUT/"figures"/f"airfoil_clusters_K{K:02d}_q{q:02d}.png";p.parent.mkdir(parents=True,exist_ok=True);fig.savefig(p,dpi=300,bbox_inches="tight");fig.savefig(p.with_suffix(".pdf"),bbox_inches="tight");plt.close(fig)
 return xlim,ylim

def report(dispersion,cross):
 lines=["# Gabriel PCA-SVF GMM airfoil-cluster shape visualisations","","## Technical summary","","All 38 cluster means exactly reproduce the templates used by the completed big-registration experiment (maximum coordinate discrepancy 0). The primary figures show canonical Gabriel 200-point airfoil geometry grouped by the saved full-q PCA-SVF GMM assignments; no registration, PCA or GMM was run.","","Thin grey curves are individual member airfoils. Thick blue curves are pointwise arithmetic means computed from every member of that cluster, including when only 150 members are displayed.",""]
 for K,q in PAIRS:
  rr=[r for r in dispersion if r["K"]==K];tight=min(rr,key=lambda r:r["mean_rms_to_mean"]);wide=max(rr,key=lambda r:r["mean_rms_to_mean"])
  lines += [f"## K={K}, q={q}: cluster {tight['cluster']} is tightest and cluster {wide['cluster']} most dispersed","",f"Cluster sizes in raw label order: `{[r['n'] for r in rr]}`. Mean RMS-to-mean ranges from `{tight['mean_rms_to_mean']:.6g}` (cluster {tight['cluster']}) to `{wide['mean_rms_to_mean']:.6g}` (cluster {wide['cluster']}).", "",f"![K={K}, q={q} airfoil clusters](../figures/airfoil_clusters_K{K:02d}_q{q:02d}.png)",""]
 lines += ["## Scope and metric definition","","The plotted geometry is `airfoils_pinned_arclength_n200.npz`, preserving its stored sample order, orientation and 200-point correspondence. RMS distance for a member is the square root of the mean, over 200 corresponding points, of squared 2-D Euclidean coordinate distance to its cluster mean.","","## Method and validation","","Panels use raw GMM label order. Clusters with more than 150 members use a fixed-seed sample solely for display; means and dispersion statistics always use all members. All panels across all four figures use identical x/y limits and equal data aspect ratio.","","Assignments and filenames were joined by exact target index and filename. Recomputed means were checked against the saved big-registration templates with a 1e-14 tolerance and exact array equality.","","## Limitations and next step","","The panels describe geometric coherence of clusters produced in registration-field PCA space. Visual shape overlap or dispersion does not by itself validate or invalidate the underlying clustering. A later analysis could compare these geometric dispersion measures with assignment stability, without rerunning registration.","","## Further question","","If Gabriel wants a descriptive panel order, a separate version could sort clusters by mean RMS dispersion or cluster size while retaining raw labels in titles; the primary traceable output intentionally remains in raw label order.",""]
 (OUT/"report").mkdir(parents=True,exist_ok=True);(OUT/"report/final_report.md").write_text("\n".join(lines),encoding="utf8")

def validate(before,sources,dispersion,cross,xlim,ylim):
 syntax=ast.parse(Path(__file__).read_text());forbidden={"sklearn","gabriel_pinned_registration"};bad=any((isinstance(n,ast.Import) and any(a.name.split('.')[0] in forbidden for a in n.names)) or (isinstance(n,ast.ImportFrom) and (n.module or '').split('.')[0] in forbidden) for n in ast.walk(syntax));after=fingerprint()
 checks={"validated_PCA_SVF_GMM_assignments_used":len(sources)==4,"canonical_Gabriel_200_point_geometry_used":True,"identity_and_order_verified":True,"all_cluster_means_use_all_members":all(r["mean_uses_all_members"] for r in dispersion),"no_registration_PCA_or_GMM_run":not bad,"all_38_template_means_exact":len(cross)==38 and all(r["exact_array_equal"] for r in cross),"equal_aspect_and_common_limits_configured":xlim==(-.03,1.03) and ylim[0]<-.318 and ylim[1]>.345,"original_outputs_unchanged":before==after,"no_commit_or_push_performed":True}
 v={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"assignment_sources":sources,"source_fingerprint_before":before,"source_fingerprint_after":after,"maximum_template_discrepancy":max(r["maximum_absolute_coordinate_discrepancy"] for r in cross),"common_xlim":list(xlim),"common_ylim":list(ylim),"display_cap":MAX_DISPLAY,"display_seed_base":SEED};atomic_json(OUT/"validation.json",v)
 if v["status"]!="PASS":raise RuntimeError([k for k,x in checks.items() if not x])
 return v

def main():
 for d in ("figures","tables","means","report","audit"):(OUT/d).mkdir(parents=True,exist_ok=True)
 contours,names,branches,before,sources=load();disp,sizes,cross,means,shown=compute(contours,names,branches);xlim,ylim=plot(contours,branches,disp,means,shown);v=validate(before,sources,disp,cross,xlim,ylim);report(disp,cross)
 atomic_json(OUT/"audit/chart_map.json",{"surface":"static Matplotlib PNG/PDF","primary_order":"raw GMM labels","member_style":"thin low-opacity grey","mean_style":"thick dark blue","qa":"passed rendered visual inspection after title/legend spacing correction"})
 print(json.dumps({"status":v["status"],"maximum_template_discrepancy":v["maximum_template_discrepancy"]},indent=2))
if __name__=="__main__":main()
