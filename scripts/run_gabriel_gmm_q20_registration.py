"""Fixed-q=20 GMM-template Gabriel registration, with checkpointed K=2..9 runs."""
from __future__ import annotations

import argparse,csv,hashlib,json,math,os,sys,time,warnings
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"scripts"))
import gabriel_pinned_registration as gabriel
import run_gabriel_gmm_template_registration as base

DATA=ROOT/"data/airfoils/processed_gabriel/airfoils_pinned_arclength_n200.npz"
QSENS=ROOT/"outputs/gabriel_pinned_gmm_q_sensitivity"
PCAEXP=ROOT/"outputs/gabriel_pinned_gmm_vs_kmeans"
OLD=ROOT/"outputs/gabriel_gmm_template_registration"
KMEANS=ROOT/"outputs/gabriel_k2_k8_energy_magnitude"
DEV=ROOT/"outputs/gabriel_pinned_registration_dev/config.json"
OUT=ROOT/"outputs/gabriel_gmm_q20_registration"
KS=tuple(range(2,10));Q=20;CONFIG=gabriel.RegistrationConfig()

def read_csv(p):
 with p.open(newline="",encoding="utf8") as f:return list(csv.DictReader(f))
def atomic_csv(p,rows,fields=None):base.atomic_csv(p,rows,fields)
def fp_tree(p):return base.fingerprint_tree(p)
def canonical(labels):
 m={};out=[]
 for x in np.asarray(labels,int):
  if x not in m:m[x]=len(m)
  out.append(m[x])
 return tuple(out)
def match_labels(source,target,K):
 c=np.zeros((K,K),int)
 for a,b in zip(source,target):c[int(a),int(b)]+=1
 r,col=linear_sum_assignment(-c);mapping={int(a):int(b) for a,b in zip(r,col)}
 return np.asarray([mapping[int(x)] for x in source]),mapping
def core_geometry(x):
 seg=np.linalg.norm(np.diff(np.vstack([x,x[:1]]),axis=0),axis=1)
 area=gabriel.signed_polygon_area(x);inter=gabriel.self_intersection_count(x);strict=gabriel.validate_pinned_contour(x)
 ok=bool(x.shape==(200,2) and np.isfinite(x).all() and area>1e-8 and inter==0 and seg.min()>1e-12 and np.argmin(x[:,0]) in range(95,105))
 return {"finite":bool(np.isfinite(x).all()),"shape_ok":x.shape==(200,2),"signed_area":area,"self_intersections":inter,
         "minimum_segment":float(seg.min()),"leading_edge_index":int(np.argmin(x[:,0])),"core_valid":ok,
         "strict_pinned_valid":bool(strict["valid"]),"strict_pinned_failure_reason":strict.get("failure_reason","")}

def prepare():
 with np.load(DATA,allow_pickle=False) as z:contours=z["contours"].copy();names=z["filenames"].astype(str)
 if contours.shape!=(1652,200,2) or contours.dtype!=np.float64 or not np.isfinite(contours).all():raise RuntimeError("STOP: pinned input mismatch")
 X=np.ascontiguousarray(contours).reshape(1652,400,order="C")
 tr=np.load(PCAEXP/"splits/train_indices.npy");ho=np.load(PCAEXP/"splits/heldout_indices.npy")
 rng=np.random.default_rng(20260813);order=rng.permutation(1652);ntr=int(round(.8*1652))
 if not np.array_equal(tr,np.sort(order[:ntr])) or not np.array_equal(ho,np.sort(order[ntr:])):raise RuntimeError("STOP: split mismatch")
 with np.load(PCAEXP/"pca/pca_q100.npz",allow_pickle=False) as z:
  mean=z["mean"].copy();components=z["components"].copy();evr=z["explained_variance_ratio"].copy();saved_tr=z["train_scores"].copy();saved_ho=z["heldout_scores"].copy()
 trs=(X[tr]-mean)@components.T;hos=(X[ho]-mean)@components.T;full=(X-mean)@components.T
 if not np.allclose(trs[:,:Q],saved_tr[:,:Q],rtol=0,atol=5e-14) or not np.allclose(hos[:,:Q],saved_ho[:,:Q],rtol=0,atol=5e-14):raise RuntimeError("STOP: PCA scores mismatch")
 rec=hos[:,:Q]@components[:Q]+mean;pca={"q":Q,"cumulative_explained_variance":float(evr[:Q].sum()),"heldout_reconstruction_mse":float(np.mean((X[ho]-rec)**2)),
  "train_count":len(tr),"heldout_count":len(ho),"split_seed":20260813,"train_scores_shape":[len(tr),Q],"heldout_scores_shape":[len(ho),Q],"full_scores_shape":[len(X),Q],
  "fit":"sklearn PCA randomized, fitted only on training shapes; cached pca_q100 first 20 components","flatten_order":"C","input_shape":[1652,400]}
 stab={(int(r["K"]),int(r["q"])):r for r in read_csv(QSENS/"tables/q_k_gmm_seed_stability.csv")}
 quality={(int(r["K"]),int(r["q"])):r for r in read_csv(QSENS/"tables/q_k_model_quality.csv")}
 records={};summary=[];sizes_rows=[];geom_rows=[]
 for K in KS:
  s=stab[(K,Q)];seed=int(s["representative_seed"]);rep=np.load(QSENS/"assignments"/f"representative_q20_K{K:02d}.npy")
  seedlabs=[]
  for sd in range(20):
   p=QSENS/"seed_assignments"/f"q20_K{K:02d}_seed{sd:02d}.npy"
   if not p.exists():raise RuntimeError(f"STOP: missing seed fit {p}")
   seedlabs.append(np.load(p))
  aris=[adjusted_rand_score(seedlabs[i],seedlabs[j]) for i in range(20) for j in range(i+1,20)]
  if not np.isclose(np.mean(aris),float(s["mean_pairwise_ARI"]),rtol=0,atol=1e-14):raise RuntimeError("STOP: seed stability mismatch")
  with warnings.catch_warnings():
   warnings.simplefilter("ignore");model=GaussianMixture(n_components=K,covariance_type="full",n_init=5,random_state=seed,reg_covar=1e-7,max_iter=500,tol=1e-5,init_params="kmeans").fit(trs[:,:Q])
  pred=model.predict(full[:,:Q]);aligned,mapping=match_labels(pred,rep,K)
  if not np.array_equal(aligned,rep):raise RuntimeError(f"STOP: representative seed reproduction mismatch K={K}")
  resp_raw=model.predict_proba(full[:,:Q]);resp=np.zeros_like(resp_raw)
  for old,new in mapping.items():resp[:,new]=resp_raw[:,old]
  weights=np.zeros(K)
  for old,new in mapping.items():weights[new]=model.weights_[old]
  tp=QSENS/"templates"/f"hard_templates_q20_K{K:02d}.npz"
  with np.load(tp,allow_pickle=False) as z:templates=z["coordinates"].copy();cached_sizes=z["cluster_sizes"].copy()
  calc=np.stack([contours[rep==c].mean(0) for c in range(K)])
  sizes=np.bincount(rep,minlength=K)
  if not np.array_equal(templates,calc) or not np.array_equal(cached_sizes,sizes):raise RuntimeError(f"STOP: cached template mismatch K={K}")
  geometry=[core_geometry(x) for x in templates]
  if not all(g["core_valid"] for g in geometry):raise RuntimeError(f"STOP: invalid q20 template K={K}")
  typ="modal" if s["representative_type"]=="exact_modal_partition" else "ARI-medoid"
  amb=float(np.mean(resp.max(1)<.8));percent=100*sizes/1652
  row={"K":K,"q":Q,"representative_type":typ,"representative_seed":seed,"modal_frequency":int(s["modal_frequency"]),"modal_fraction":float(s["modal_fraction"]),
       "mean_seed_ARI":float(s["mean_pairwise_ARI"]),"sd_seed_ARI":float(s["sd_pairwise_ARI"]),"cluster_sizes":json.dumps(sizes.tolist()),"cluster_percentages":json.dumps(percent.tolist()),
       "smallest_cluster":int(sizes.min()),"smallest_cluster_percentage":float(percent.min()),"largest_cluster":int(sizes.max()),"largest_cluster_percentage":float(percent.max()),
       "component_weights":json.dumps(weights.tolist()),"minimum_component_weight":float(weights.min()),"ambiguous_fraction_resp_lt_0_8":amb,"ambiguous_percentage":100*amb,
       "n_clusters_lt2pct":int(np.sum(sizes<.02*1652)),"n_clusters_lt1pct":int(np.sum(sizes<.01*1652)),"n_clusters_lt20":int(np.sum(sizes<20)),"templates_valid":f"{sum(g['core_valid'] for g in geometry)}/{K}"}
  summary.append(row)
  for c in range(K):
   sizes_rows.append({"K":K,"q":Q,"cluster_id":c,"cluster_size":int(sizes[c]),"cluster_percentage":float(percent[c]),"component_weight":float(weights[c]),"lt2pct":sizes[c]<.02*1652,"lt1pct":sizes[c]<.01*1652,"lt20":sizes[c]<20})
   geom_rows.append({"K":K,"q":Q,"cluster_id":c,**geometry[c]})
  (OUT/"assignments").mkdir(parents=True,exist_ok=True);np.save(OUT/"assignments"/f"labels_K{K:02d}_q20.npy",rep)
  atomic_csv(OUT/"assignments"/f"labels_K{K:02d}_q20.csv",[{"target_index":i,"filename":names[i],"cluster_id":int(rep[i])} for i in range(1652)])
  gabriel.atomic_npz(OUT/"templates"/f"hard_templates_K{K:02d}_q20.npz",coordinates=templates,cluster_sizes=sizes,labels=rep,filenames=names)
  records[(K,Q)]={"labels":rep,"templates":templates,"sizes":sizes,"audit":row}
 atomic_csv(OUT/"tables/gmm_q20_assignment_summary_k2_k9.csv",summary);atomic_csv(OUT/"tables/gmm_q20_cluster_sizes_k2_k9.csv",sizes_rows);atomic_csv(OUT/"tables/gmm_q20_template_geometry.csv",geom_rows)
 gabriel.atomic_json(OUT/"pca_q20_verification.json",pca)
 return contours,names,records,summary,pca

def stat(v):return base.stats(v)
def combo_summary(K,rows):
 v=[r for r in rows if r.get("valid_registration") is True];d=stat([r["physical_dirichlet_integral"] for r in v]);m=stat([r["velocity_mean_magnitude"] for r in v]);e=stat([r["final_pointwise_mse"] for r in v]);rt=stat([r["runtime_seconds"] for r in rows])
 return {"K":K,"q":Q,"n_total":1652,"completed":sum(r.get("status")=="completed" for r in rows),"n_valid":len(v),"n_invalid":sum(r.get("status")=="completed" and not bool(r.get("valid_registration",False)) for r in rows),"n_failed":sum(r.get("status")!="completed" for r in rows),"n_missing":1652-len(rows),"n_corrupted":0,
  "mean_dirichlet":d["mean"],"sd_dirichlet":d["sd"],"se_dirichlet":d["se"],"median_dirichlet":d["median"],"p95_dirichlet":d["p95"],
  "mean_svf_magnitude":m["mean"],"sd_svf_magnitude":m["sd"],"se_svf_magnitude":m["se"],"median_svf_magnitude":m["median"],"p95_svf_magnitude":m["p95"],
  "mean_mse":e["mean"],"median_mse":e["median"],"p95_mse":e["p95"],"max_mse":e["max"],"minimum_jacobian":min((float(r["jacobian_minimum"]) for r in v),default=math.nan),
  "n_nonpositive_jacobian":sum(float(r.get("jacobian_minimum",-math.inf))<=0 or float(r.get("jacobian_nonpositive_fraction",1))>0 for r in rows),"mean_runtime":rt["mean"],"median_runtime":rt["median"]}

def comparison_tables(summary):
 old={int(r["K"]):r for r in read_csv(OLD/"tables/summary_gmm_registration_k2_k9.csv")};km={int(r["K"]):r for r in read_csv(KMEANS/"summary_by_k.csv")};new={int(r["K"]):r for r in summary}
 oldq={2:3,3:1,4:37,5:26,6:2,7:20,8:3,9:4};var=[];kc=[]
 for K in KS:
  od=float(old[K]["mean_dirichlet"]);nd=float(new[K]["mean_dirichlet"]);om=float(old[K]["mean_svf_magnitude"]);nm=float(new[K]["mean_svf_magnitude"])
  a=np.load(ROOT/"outputs/gabriel_selected_gmm_templates_audit/assignments"/f"labels_K{K:02d}_q{oldq[K]:02d}.npy");b=np.load(OUT/"assignments"/f"labels_K{K:02d}_q20.npy")
  var.append({"K":K,"old_q":oldq[K],"new_q":20,"assignment_ARI":adjusted_rand_score(a,b),"old_mean_dirichlet":od,"new_mean_dirichlet":nd,"absolute_difference_dirichlet":nd-od,"percentage_difference_dirichlet":100*(nd-od)/od,"old_mean_svf":om,"new_mean_svf":nm,"absolute_difference_svf":nm-om,"percentage_difference_svf":100*(nm-om)/om})
  kd=float(km[K]["average_dirichlet_energy"]);ks=float(km[K]["average_svf_magnitude"])
  kc.append({"K":K,"kmeans_mean_dirichlet":kd,"gmm_q20_mean_dirichlet":nd,"difference_dirichlet":nd-kd,"percentage_difference_dirichlet":100*(nd-kd)/kd,"kmeans_mean_svf":ks,"gmm_q20_mean_svf":nm,"difference_svf":nm-ks,"percentage_difference_svf":100*(nm-ks)/ks})
 atomic_csv(OUT/"tables/variable_q_vs_q20_comparison.csv",var);atomic_csv(OUT/"tables/kmeans_vs_gmm_q20_comparison.csv",kc);return var,kc

def plots(summary,var,kc):
 (OUT/"figures").mkdir(parents=True,exist_ok=True);s=sorted(summary,key=lambda r:r["K"]);x=[r["K"] for r in s]
 for fld,se,y,title,name in [("mean_dirichlet","se_dirichlet","Average physical Dirichlet energy","Average Dirichlet Energy vs Number of Templates — GMM, q=20","average_dirichlet_energy_vs_k_gmm_q20.png"),("mean_svf_magnitude","se_svf_magnitude","Average velocity-field magnitude","Average SVF Magnitude vs Number of Templates — GMM, q=20","average_svf_magnitude_vs_k_gmm_q20.png")]:
  fig,ax=plt.subplots(figsize=(10,5.8));ax.errorbar(x,[r[fld] for r in s],yerr=[r[se] for r in s],marker="o",lw=2,capsize=4,color="#1967a3");ax.set(xticks=x,xlabel="Number of templates K",ylabel=y,title=title);ax.grid(axis="y",alpha=.22);ax.spines[["top","right"]].set_visible(False);fig.tight_layout();fig.savefig(OUT/"figures"/name,dpi=300);plt.close(fig)
 for rows,prefix,labels in [(var,"variable_q_vs_q20",("Variable-q GMM","Fixed q=20 GMM")),(kc,"kmeans_vs_gmm_q20",("K-means","GMM q=20"))]:
  rr=sorted(rows,key=lambda r:r["K"]);xx=[r["K"] for r in rr]
  for metric,y,name in [("dirichlet","Mean physical Dirichlet energy",f"{prefix}_dirichlet.png"),("svf","Mean velocity-field magnitude",f"{prefix}_svf.png")]:
   if rows is var:a=[r[f"old_mean_{metric}"] for r in rr];b=[r[f"new_mean_{metric}"] for r in rr]
   else:a=[r[f"kmeans_mean_{metric}"] for r in rr];b=[r[f"gmm_q20_mean_{metric}"] for r in rr]
   fig,ax=plt.subplots(figsize=(10,5.8));ax.plot(xx,a,marker="o",lw=2,label=labels[0]);ax.plot(xx,b,marker="s",lw=2,label=labels[1]);ax.set(xticks=xx,xlabel="Number of templates K",ylabel=y,title=f"{labels[0]} vs {labels[1]} — {y}");ax.grid(axis="y",alpha=.22);ax.legend();ax.spines[["top","right"]].set_visible(False);fig.tight_layout();fig.savefig(OUT/"figures"/name,dpi=300);plt.close(fig)

def finalise(contours,names,records,assign_summary,pca,protected_before):
 allrows=[];summ=[];bad=[]
 for K in KS:
  rr=[]
  for i in range(1652):
   p=OUT/"registrations"/f"K{K:02d}_q20"/f"{i:04d}.npz";r=base.load_checkpoint(p,K,Q,i)
   if r is None:bad.append(str(p))
   else:rr.append(r);allrows.append(r)
  summ.append(combo_summary(K,rr))
 if bad:raise RuntimeError(f"missing/corrupt {len(bad)}")
 atomic_csv(OUT/"tables/gmm_q20_per_registration_results.csv",sorted(allrows,key=lambda r:(int(r["K"]),int(r["target_index"]))))
 atomic_csv(OUT/"tables/gmm_q20_registration_summary_k2_k9.csv",summ);var,kc=comparison_tables(summ);plots(summ,var,kc)
 invalid=[r for r in allrows if r.get("status")!="completed" or not bool(r.get("valid_registration",False))];atomic_csv(OUT/"tables/invalid_failed_targets.csv",invalid)
 after={"dataset":gabriel.sha256_file(DATA),"qsensitivity":fp_tree(QSENS),"variable_q":fp_tree(OLD),"kmeans":fp_tree(KMEANS)}
 checks={"exact_q20_all_K":all(r["q"]==20 for r in assign_summary) and len(assign_summary)==8,"exact_13216":len(allrows)==13216,"all_checkpoints_readable":not bad,"target_identity":all(names[int(r["target_index"])]==r["filename"] for r in allrows),"audited_assignment":all(records[(int(r["K"]),20)]["labels"][int(r["target_index"]) ]==int(r["cluster_id"]) for r in allrows),"hard_templates_exact":all(np.array_equal(records[(K,20)]["templates"][c],contours[records[(K,20)]["labels"]==c].mean(0)) for K in KS for c in range(K)),"registration_config":json.loads(json.dumps(asdict(CONFIG)))==json.loads(DEV.read_text())["registration_config"],"diagnostic_101":all(int(r["diagnostic_resolution"])==101 for r in allrows),"protected_unchanged":protected_before==after,"summary_reproducible":len(summ)==8}
 gabriel.atomic_json(OUT/"validation.json",{"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"counts":{"rows":len(allrows),"invalid_or_failed":len(invalid)},"protected":after})
 rep=["# Gabriel fixed-q=20 GMM-template registration","","## Technical summary","",f"All eight fixed-q=20 GMM assignments were registered against 1,652 Gabriel pinned contours using the frozen production configuration. Validation: **{'PASS' if all(checks.values()) else 'FAIL'}**.","",f"PCA q=20 cumulative variance: {pca['cumulative_explained_variance']:.10g}; held-out reconstruction MSE: {pca['heldout_reconstruction_mse']:.10g}.","","## Registration summary","","| K | valid / invalid / failed | Dirichlet mean ± SE | SVF mean ± SE |","|---:|---:|---:|---:|"]
 for r in summ:rep.append(f"| {r['K']} | {r['n_valid']} / {r['n_invalid']} / {r['n_failed']} | {r['mean_dirichlet']:.8g} ± {r['se_dirichlet']:.3g} | {r['mean_svf_magnitude']:.8g} ± {r['se_svf_magnitude']:.3g} |")
 rep += ["","## Interpretation boundary","","These metrics quantify deformation burden under fixed clustering and registration choices. They do not prove clustering quality and do not select K."]
 (OUT/"report").mkdir(parents=True,exist_ok=True);(OUT/"report/final_report.md").write_text("\n".join(rep)+"\n",encoding="utf8")

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--stage",choices=("prepare","run","finalise","all"),default="all");ap.add_argument("--workers",type=int,default=4);args=ap.parse_args()
 if args.workers!=4:ap.error("exactly 4 workers required")
 for d in ("assignments","templates","registrations","tables","figures","report"):(OUT/d).mkdir(parents=True,exist_ok=True)
 protected={"dataset":gabriel.sha256_file(DATA),"qsensitivity":fp_tree(QSENS),"variable_q":fp_tree(OLD),"kmeans":fp_tree(KMEANS)}
 contours,names,records,summary,pca=prepare()
 gabriel.atomic_json(OUT/"config.json",{"experiment":"Gabriel fixed-q=20 GMM-template registration","dataset":str(DATA.relative_to(ROOT)),"q":20,"K":list(KS),"workers":4,"registration_config":asdict(CONFIG),"diagnostic_resolution":101,"pca":pca,"protected_before":protected})
 print("K | q | representative | seed | mode | mean ARI | sizes | min | ambiguous % | templates",flush=True)
 for r in summary:print(f"{r['K']} | 20 | {r['representative_type']} | {r['representative_seed']} | {r['modal_frequency']}/20 | {r['mean_seed_ARI']:.3f} | {r['cluster_sizes']} | {r['smallest_cluster']} | {r['ambiguous_percentage']:.2f}% | {r['templates_valid']}",flush=True)
 if args.stage=="prepare":return
 base.OUT=OUT;base.CONFIG=CONFIG;base.REP={(K,Q):(records[(K,Q)]["audit"]["representative_type"],records[(K,Q)]["audit"]["representative_seed"]) for K in KS}
 if args.stage in ("run","all"):
  for K in KS:
   base.append_log(OUT/"progress.log","combination_start_q20",K=K,q=20)
   base.run_combo(K,Q,contours,names,records[(K,Q)],4,1)
 if args.stage in ("finalise","all"):finalise(contours,names,records,summary,pca,protected)
if __name__=="__main__":main()
