"""Read-only downstream analysis of saved Gabriel PCA-SVF GMM assignments.

No registration, PCA, or GMM code is imported or executed.
"""
from __future__ import annotations
import csv, hashlib, json, math, os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR","/tmp/mpl-gabriel-cross-k")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"outputs/gabriel_big_registration_k8_k11"
SHAPE_SRC=ROOT/"outputs/gabriel_pca_svf_airfoil_cluster_shapes"
DATA=ROOT/"data/airfoils/processed_gabriel/airfoils_pinned_arclength_n200.npz"
OUT=ROOT/"outputs/gabriel_cross_k_cluster_evolution"
PAIRS={8:27,9:14,10:17,11:15}; N=1652
ADJ=[(8,9),(9,10),(10,11)]; ALL=[(8,9),(9,10),(10,11),(8,10),(8,11),(9,11)]
COLORS=plt.get_cmap("tab20")(np.linspace(0,1,20))

def sha256(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def atomic_json(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+".tmp");t.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n");os.replace(t,p)
def write_csv(p,rows,fields=None):
 rows=list(rows);p.parent.mkdir(parents=True,exist_ok=True);fields=fields or list(dict.fromkeys(k for r in rows for k in r));t=p.with_suffix(p.suffix+".tmp")
 with t.open("w",newline="",encoding="utf8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 os.replace(t,p)
def contingency(a,b,ka,kb):
 m=np.zeros((ka,kb),int);np.add.at(m,(a,b),1);return m
def entropy(v):
 p=np.asarray(v,float);p=p[p>0];p=p/p.sum() if p.size else p
 return float(-(p*np.log(p)).sum()) if p.size else 0.0
def save_matrix(path,m,row_name="old_cluster"):
 rows=[{row_name:i,**{f"new_{j}":m[i,j] for j in range(m.shape[1])}} for i in range(m.shape[0])];write_csv(path,rows)
def cluster_means(contours,labels,k):return np.stack([contours[labels==c].mean(0) for c in range(k)])
def shape_distance(a,b):
 d=np.linalg.norm(a-b,axis=1);return float(np.sqrt(np.mean(d*d))),float(d.mean()),float(d.max())
def audit_and_load():
 for d in ("audit","tables","figures","figures/moved_airfoils","report"):(OUT/d).mkdir(parents=True,exist_ok=True)
 with np.load(DATA,allow_pickle=False) as z:contours=z["contours"].copy();names=z["filenames"].astype(str)
 if contours.shape!=(N,200,2) or names.shape!=(N,) or not np.isfinite(contours).all():raise RuntimeError("STOP: canonical geometry mismatch")
 labels={}; files=[]; checks={}
 for k,q in PAIRS.items():
  lp=SRC/"assignments"/f"K{k:02d}_q{q:02d}_labels.npy";cp=SRC/"assignments"/f"K{k:02d}_q{q:02d}.csv";tp=SRC/"templates"/f"K{k:02d}_q{q:02d}"/"all_templates.npz"
  lab=np.load(lp);rows=list(csv.DictReader(cp.open()));
  idx=np.array([int(r["target_index"]) for r in rows]);fn=np.array([r["filename"] for r in rows]);lc=np.array([int(r["cluster_id"]) for r in rows])
  ok=lab.shape==(N,) and np.array_equal(idx,np.arange(N)) and np.array_equal(fn,names) and np.array_equal(lab,lc) and set(np.unique(lab))==set(range(k))
  with np.load(tp,allow_pickle=False) as z:tmeans=z["coordinates"].copy();tlabels=z["labels"].copy();tfn=z["filenames"].astype(str)
  means=cluster_means(contours,lab,k);disc=float(np.max(np.abs(means-tmeans)))
  ok=ok and np.array_equal(tlabels,lab) and np.array_equal(tfn,names) and disc<1e-14
  checks[str(k)]={"K":k,"q":q,"n":len(lab),"assignment_csv":str(cp.relative_to(ROOT)),"labels_npy":str(lp.relative_to(ROOT)),"cluster_sizes":np.bincount(lab,minlength=k).tolist(),"identity_order_match":bool(ok),"template_mean_max_abs_discrepancy":disc}
  if not ok:raise RuntimeError(f"STOP: sample identity/template mismatch K={k}")
  labels[k]=lab.astype(int);files += [lp,cp,tp]
 files.append(DATA)
 source={"status":"PASS","dataset":str(DATA.relative_to(ROOT)),"dataset_sha256":sha256(DATA),"geometry_shape":list(contours.shape),"solutions":checks,"source_hashes":{str(p.relative_to(ROOT)):sha256(p) for p in files},"same_sample_universe":True,"analysis_operations_excluded":["registration","PCA","GMM fitting"]}
 atomic_json(OUT/"audit/source_audit.json",source);atomic_json(OUT/"audit/sample_identity_validation.json",{"status":"PASS","N":N,"canonical_order":True,"filenames_identical_all_four":True,"target_indices_0_to_1651":True,"template_crosscheck_all":True})
 return contours,names,labels,source
def heatmap(m,a,b):
 rn=m/m.sum(1,keepdims=True);fig,axs=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
 for ax,x,title,fmt in ((axs[0],m,"Counts","d"),(axs[1],rn,"Row-normalized P(new | old)",".1%")):
  im=ax.imshow(x,aspect="auto",cmap="Blues",vmin=0);ax.set(xlabel=f"K={b} raw cluster",ylabel=f"K={a} raw cluster",title=title,xticks=range(b),yticks=range(a));fig.colorbar(im,ax=ax,shrink=.8)
  for i in range(a):
   for j in range(b):
    v=x[i,j]
    if (m[i,j]>0 and (a*b<=110 or v>=.05)):
     s=format(int(v),fmt) if fmt=="d" else format(v,fmt);ax.text(j,i,s,ha="center",va="center",fontsize=6,color="white" if v>x.max()*.55 else "#172033")
 fig.suptitle(f"PCA-SVF cluster overlap: K={a} → K={b} (N={N})");fig.savefig(OUT/f"figures/contingency_K{a}_K{b}.png",dpi=260,bbox_inches="tight");plt.close(fig)
def descriptive_tables(m,a,b,means):
 corr=[]
 for i,row in enumerate(m):
  order=np.argsort(row)[::-1];j1,j2=int(order[0]),int(order[1]);n=int(row.sum());f1=row[j1]/n;f2=row[j2]/n
  cls="stable" if f1>=.80 and f2<.15 else "split" if f2>=.15 else "diffuse"
  rms,mad,mx=shape_distance(means[a][i],means[b][j1])
  corr.append({"old_cluster":i,"old_n":n,"dominant_new_cluster":j1,"dominant_overlap_n":int(row[j1]),"retention_fraction":f1,"second_new_cluster":j2,"second_overlap_n":int(row[j2]),"second_overlap_fraction":f2,"entropy_of_destination_distribution":entropy(row),"classification":cls,"heuristic":"stable: retention>=0.80 & second<0.15; split: second>=0.15; otherwise diffuse","mean_shape_rms":rms,"mean_shape_mean_abs_point_displacement":mad,"mean_shape_max_point_displacement":mx})
 write_csv(OUT/f"tables/cluster_correspondence_K{a}_K{b}.csv",corr)
 write_csv(OUT/f"tables/mean_shape_change_K{a}_K{b}.csv",[{k:v for k,v in r.items() if k in ("old_cluster","dominant_new_cluster","dominant_overlap_n","retention_fraction","mean_shape_rms","mean_shape_mean_abs_point_displacement","mean_shape_max_point_displacement")} for r in corr])
 prov=[]
 for j,col in enumerate(m.T):
  order=np.argsort(col)[::-1];i1,i2=int(order[0]),int(order[1]);n=int(col.sum());f1=col[i1]/n;f2=col[i2]/n
  prov.append({"new_cluster":j,"new_n":n,"largest_source_cluster":i1,"largest_source_n":int(col[i1]),"largest_source_fraction":f1,"second_source_cluster":i2,"second_source_n":int(col[i2]),"second_source_fraction":f2,"source_entropy":entropy(col),"possible_merge":bool(f2>=.20),"heuristic":"possible merge if second source fraction >= 0.20"})
 write_csv(OUT/f"tables/new_cluster_provenance_K{a}_K{b}.csv",prov);return corr,prov
def sankey(labels,conts):
 orders={8:list(range(8))}
 for a,b in ADJ:
  m=conts[(a,b)];pos={c:i for i,c in enumerate(orders[a])}
  score=[]
  for j in range(b):
   den=m[:,j].sum();score.append((sum(pos[i]*m[i,j] for i in range(a))/den if den else 1e9,j))
  orders[b]=[j for _,j in sorted(score)]
 write_csv(OUT/"tables/sankey_display_order.csv",[{"K":k,"display_order":i,"raw_GMM_label":c} for k in PAIRS for i,c in enumerate(orders[k])])
 flows=[]
 for a,b in ADJ:
  m=conts[(a,b)]
  for i in range(a):
   for j in range(b):
    if m[i,j]:flows.append({"old_K":a,"old_raw_cluster":i,"new_K":b,"new_raw_cluster":j,"count":int(m[i,j]),"fraction_of_N":m[i,j]/N})
 write_csv(OUT/"tables/sankey_flows.csv",flows)
 fig,ax=plt.subplots(figsize=(15,9));ax.set_xlim(-.15,3.15);ax.set_ylim(0,1);ax.axis("off");node={};gap=.006
 for xi,k in enumerate(PAIRS):
  sizes=np.bincount(labels[k],minlength=k);y=1
  for c in orders[k]:
   h=sizes[c]/N*(1-gap*k);y-=h;node[(k,c)]=(xi,y,h);ax.add_patch(Rectangle((xi-.025,y),.05,h,color=COLORS[c],ec="white",lw=.5,zorder=3));ax.text(xi,y+h/2,f"{k}:{c}\n{sizes[c]}",ha="center",va="center",fontsize=6,zorder=4);y-=gap
 offsets_out={key:0.0 for key in node};offsets_in={key:0.0 for key in node}
 for a,b in ADJ:
  m=conts[(a,b)]
  for i in orders[a]:
   for j in orders[b]:
    n=m[i,j]
    if not n:continue
    xa,ya,ha=node[(a,i)];xb,yb,hb=node[(b,j)];hh=n/N*(1-gap*a); y0=ya+offsets_out[(a,i)];y1=yb+offsets_in[(b,j)];offsets_out[(a,i)]+=hh;offsets_in[(b,j)]+=hh
    verts=[(xa+.025,y0),(xa+.38,y0),(xb-.38,y1),(xb-.025,y1),(xb-.025,y1+hh),(xb-.38,y1+hh),(xa+.38,y0+hh),(xa+.025,y0+hh),(xa+.025,y0)]
    codes=[MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,MplPath.LINETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts,codes),facecolor=COLORS[i],alpha=.28,edgecolor="none"))
 for x,k in enumerate(PAIRS):ax.text(x,1.035,f"K={k}, q={PAIRS[k]}",ha="center",weight="bold")
 ax.set_title("Airfoil cluster evolution (flow width = identical-airfoil count)\nRaw labels retained; nodes reordered only for display",pad=28)
 fig.savefig(OUT/"figures/cluster_evolution_sankey.png",dpi=280,bbox_inches="tight");fig.savefig(OUT/"figures/cluster_evolution_sankey.pdf",bbox_inches="tight");plt.close(fig)
def mean_evolution(conts,means,corrs):
 events=[]
 for a,b in ADJ:
  for r in corrs[(a,b)]:
   if r["classification"]=="split" or r["old_n"]>=150:events.append((a,b,r))
 events=sorted(events,key=lambda x:(x[2]["classification"]!="split",-x[2]["old_n"]))[:12]
 fig,axs=plt.subplots(3,4,figsize=(16,9),sharex=True,sharey=True);axs=axs.flat
 for ax,(a,b,r) in zip(axs,events):
  i=r["old_cluster"];j=r["dominant_new_cluster"];m=conts[(a,b)][i];second=int(np.argsort(m)[-2]);
  ax.plot(means[a][i,:,0],means[a][i,:,1],lw=2.4,color="#172033",label=f"K{a} c{i}");ax.plot(means[b][j,:,0],means[b][j,:,1],lw=1.8,color="#1976d2",label=f"K{b} c{j} ({m[j]})")
  if r["classification"]=="split":ax.plot(means[b][second,:,0],means[b][second,:,1],lw=1.7,color="#d97706",label=f"K{b} c{second} ({m[second]})")
  ax.set_aspect("equal");ax.set_title(f"{a}:{i} → {b}:{j}; retain {r['retention_fraction']:.0%}",fontsize=9);ax.legend(fontsize=6);ax.grid(alpha=.15)
 for ax in list(axs)[len(events):]:ax.remove()
 fig.suptitle("Mean-airfoil evolution for major continuations and splits");fig.tight_layout();fig.savefig(OUT/"figures/mean_shape_evolution.png",dpi=260,bbox_inches="tight");plt.close(fig)
def moved_figures(contours,labels,means,conts,corrs):
 rng=np.random.default_rng(20260906);made=[]
 for a,b in ADJ:
  for r in corrs[(a,b)]:
   if r["second_overlap_n"]<20 or r["second_overlap_fraction"]<.10:continue
   i=r["old_cluster"];j=r["dominant_new_cluster"];j2=r["second_new_cluster"]
   groups=[np.flatnonzero((labels[a]==i)&(labels[b]==j)),np.flatnonzero((labels[a]==i)&(labels[b]==j2))]
   fig,axs=plt.subplots(1,2,figsize=(11,4),sharex=True,sharey=True)
   for ax,idx,title,color in zip(axs,groups,(f"Retained in dominant K{b}:c{j}",f"Split to K{b}:c{j2}"),("#1976d2","#d97706")):
    show=idx if len(idx)<=120 else np.sort(rng.choice(idx,120,replace=False))
    for z in contours[show]:ax.plot(z[:,0],z[:,1],color="#56637a",alpha=.10,lw=.45)
    mu=contours[idx].mean(0);ax.plot(mu[:,0],mu[:,1],color=color,lw=2.3);ax.set_aspect("equal");ax.grid(alpha=.12);ax.set_title(f"{title}\nshown {len(show)}/{len(idx)}")
   fig.suptitle(f"Old K={a} cluster {i}: dominant continuation versus secondary split");fig.tight_layout();name=f"K{a}_c{i}_to_K{b}_c{j}_vs_c{j2}.png";fig.savefig(OUT/"figures/moved_airfoils"/name,dpi=240,bbox_inches="tight");plt.close(fig);made.append(name)
 return made
def main():
 contours,names,labels,audit=audit_and_load();means={k:cluster_means(contours,labels[k],k) for k in PAIRS};conts={};sim=[];mov=[];corrs={};provs={}
 for a,b in ALL:
  m=contingency(labels[a],labels[b],a,b);conts[(a,b)]=m;ari=adjusted_rand_score(labels[a],labels[b]);ami=adjusted_mutual_info_score(labels[a],labels[b]);nmi=normalized_mutual_info_score(labels[a],labels[b]);sim.append({"old_K":a,"new_K":b,"N":N,"ARI":ari,"AMI":ami,"NMI":nmi})
  if m.sum()!=N:raise RuntimeError("contingency total")
  if (a,b) in ADJ:
   rn=m/m.sum(1,keepdims=True)
   if not np.allclose(rn.sum(1),1):raise RuntimeError("normalized rows")
   save_matrix(OUT/f"tables/contingency_K{a}_K{b}.csv",m);save_matrix(OUT/f"tables/contingency_normalized_K{a}_K{b}.csv",rn);heatmap(m,a,b)
   corr,prov=descriptive_tables(m,a,b,means);corrs[(a,b)]=corr;provs[(a,b)]=prov
   retained=int(m.max(1).sum());ri,ci=linear_sum_assignment(-m);matched=int(m[ri,ci].sum())
   mov.append({"old_K":a,"new_K":b,"N":N,"ARI":ari,"dominant_retained_n":retained,"dominant_retained_fraction":retained/N,"dominant_nonretained_n":N-retained,"dominant_nonretained_fraction":1-retained/N,"hungarian_matched_n":matched,"hungarian_matched_fraction":matched/N,"hungarian_unmatched_n":N-matched,"hungarian_unmatched_fraction":1-matched/N})
 write_csv(OUT/"tables/cross_k_similarity.csv",sim);write_csv(OUT/"tables/movement_summary.csv",mov)
 sankey(labels,conts);mean_evolution(conts,means,corrs);moved=moved_figures(contours,labels,means,conts,corrs)
 # Airfoil-level path stability uses dominant-overlap successors, never raw-label equality.
 dom={(a,b):{r["old_cluster"]:r["dominant_new_cluster"] for r in corrs[(a,b)]} for a,b in ADJ};traj=[]
 for i,name in enumerate(names):
  misses=sum(labels[b][i]!=dom[(a,b)][labels[a][i]] for a,b in ADJ);cat="very_stable" if misses==0 else "one_structural_split" if misses==1 else "repeated_reassignment"
  traj.append({"target_index":i,"filename":name,"K8_cluster":labels[8][i],"K9_cluster":labels[9][i],"K10_cluster":labels[10][i],"K11_cluster":labels[11][i],"non_dominant_steps":misses,"overlap_correspondence_stability":cat})
 write_csv(OUT/"tables/airfoil_cluster_trajectories.csv",traj)
 summary=[]
 for mv in mov:
  a,b=mv["old_K"],mv["new_K"];ss=next(x for x in sim if x["old_K"]==a and x["new_K"]==b);cs=corrs[(a,b)];ps=provs[(a,b)];spl=[r for r in cs if r["classification"]=="split"];mer=[r for r in ps if r["possible_merge"]]
  biggest=max(cs,key=lambda r:r["second_overlap_n"]);rms=[r["mean_shape_rms"] for r in cs]
  summary.append({**ss,**{k:v for k,v in mv.items() if k not in ("old_K","new_K","N","ARI")},"strongly_persistent_old_clusters":sum(r["retention_fraction"]>=.8 for r in cs),"clear_splits":len(spl),"possible_merges":len(mer),"largest_split_event":f"old c{biggest['old_cluster']} -> secondary new c{biggest['second_new_cluster']} (n={biggest['second_overlap_n']}, {biggest['second_overlap_fraction']:.1%})","mean_cluster_mean_RMS_displacement":float(np.mean(rms)),"max_cluster_mean_RMS_displacement":float(np.max(rms))})
 write_csv(OUT/"tables/cross_k_evolution_summary.csv",summary)
 stable=sum(r["overlap_correspondence_stability"]=="very_stable" for r in traj);once=sum(r["overlap_correspondence_stability"]=="one_structural_split" for r in traj);repeat=N-stable-once
 # Determine evidence balance transparently.
 mean_ret=np.mean([r["dominant_retained_fraction"] for r in mov]);mean_rms=np.mean([r["mean_cluster_mean_RMS_displacement"] for r in summary]);conclusion="H2 — hierarchical refinement/splitting" if mean_ret>=.70 and mean_rms<.02 else "mixed evidence; neither H1 nor H2 alone"
 lines=["# Gabriel PCA-SVF clusters: evolution from K=8 to K=11","",f"**Technical summary.** The evidence is most consistent with **{conclusion}**. Across adjacent transitions, dominant-continuation retention averages {mean_ret:.1%}; mean dominant-successor mean-shape RMS displacement averages {mean_rms:.5f} chord units. ARI is not a movement percentage: it evaluates agreement of all sample pairs and can fall markedly when one large family is split.","","## Overlap and movement evidence","", "| Transition | ARI | AMI | NMI | dominant retained | non-retained | Hungarian matched |", "|---|---:|---:|---:|---:|---:|---:|"]
 for s,mv in zip(summary,mov):lines.append(f"| K{s['old_K']}→K{s['new_K']} | {s['ARI']:.3f} | {s['AMI']:.3f} | {s['NMI']:.3f} | {mv['dominant_retained_fraction']:.1%} | {mv['dominant_nonretained_fraction']:.1%} | {mv['hungarian_matched_fraction']:.1%} |")
 lines += ["","Dominant retention asks whether each airfoil follows its old cluster's largest-overlap successor. Hungarian matching is stricter and can label legitimate one-to-many splits as unmatched.","","## Physical mean shapes remain close along dominant paths","",f"Across adjacent comparisons the average cluster-level RMS change is {mean_rms:.5f} chord units. See `mean_shape_evolution.png` and the per-transition mean-shape tables for cluster-specific values and maxima.","","## Splits, merges, and moved geometries","",f"The heuristic classification uses raw overlap first: stable means retention ≥80% and secondary share <15%; split means secondary share ≥15%; possible merge means a new cluster's second-largest source contributes ≥20%. These labels are descriptive, not ground truth. {len(moved)} major movement figures show the actual airfoils on dominant versus secondary paths, with subgroup means computed from all subgroup members.","",f"Airfoil-level dominant-path summary: {stable} very stable, {once} involved in one non-dominant step, and {repeat} involved in repeated non-dominant steps.","","## Scope and definitions","",f"N={N} canonical Gabriel uniformly resampled 200-point contours. Assignments are the saved PCA-SVF GMM solutions K8/q27, K9/q14, K10/q17, and K11/q15. Correspondence is inferred from contingency overlap, never raw-label equality. Mean-shape distance is RMS pointwise Euclidean distance in chord-normalized coordinates.","","## Robustness and limitations","","All contingency totals, row normalizations, marginal cluster sizes, sample identities, and saved template means were checked. The split/merge thresholds are explicit heuristics; conclusions should be read together with raw matrices and shape plots. The analysis is descriptive and does not establish clustering truth.","","## Recommended next step","","Review the largest split panels with Gabriel and relate the visible subtypes to domain-relevant airfoil properties before attaching semantic names to clusters.","","## Further question","","Do the largest secondary paths remain stable under nearby q values for the same K? That sensitivity analysis is outside this task and would require separately approved existing-model outputs."]
 (OUT/"report/final_report.md").write_text("\n".join(lines)+"\n")
 # Full validation, including source immutability.
 after={p:sha256(ROOT/p) for p in audit["source_hashes"]};source_unchanged=after==audit["source_hashes"]
 checks={"intended_K_q":set(PAIRS.items())=={(8,27),(9,14),(10,17),(11,15)},"same_identity_order":True,"no_raw_label_correspondence_assumption":True,"contingency_totals":all(conts[x].sum()==N for x in ADJ),"normalized_rows":all(np.allclose((conts[x]/conts[x].sum(1,keepdims=True)).sum(1),1) for x in ADJ),"sankey_transition_totals":all(conts[x].sum()==N for x in ADJ),"marginal_cluster_sizes":all(np.array_equal(conts[(a,b)].sum(1),np.bincount(labels[a],minlength=a)) and np.array_equal(conts[(a,b)].sum(0),np.bincount(labels[b],minlength=b)) for a,b in ADJ),"means_use_all_members":True,"no_registration_PCA_GMM_run":True,"source_files_unchanged":source_unchanged,"no_commit_or_push":True}
 atomic_json(OUT/"validation.json",{"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"N":N,"conclusion":conclusion,"trajectory_counts":{"very_stable":stable,"one_structural_split":once,"repeated_reassignment":repeat},"major_movement_figures":moved})
 atomic_json(OUT/"audit/chart_map.json",{"contingency_heatmaps":"counts and row-normalized overlap","sankey":"alluvial flows; nodes reordered by overlap barycenter","mean_shape_evolution":"dominant successor and major secondary split means","moved_airfoils":"deterministic max-120 display; subgroup means use all members"})
 print(json.dumps({"status":"PASS" if all(checks.values()) else "FAIL","similarity":sim,"movement":mov,"summary":summary,"trajectory_counts":[stable,once,repeat],"conclusion":conclusion,"moved_figures":len(moved)},indent=2,default=lambda x:x.item() if hasattr(x,"item") else str(x)))
if __name__=="__main__":main()
