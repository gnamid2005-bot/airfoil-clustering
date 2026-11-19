"""Gabriel big registration K=2-7 using the full-range leftmost-q rule.

Stops after constructing six new coefficient matrices.  It deliberately
contains no second-stage PCA or GMM analysis.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,time,warnings
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from pathlib import Path

for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
os.environ.setdefault("MPLCONFIGDIR","/tmp/mpl-gabriel-big-registration")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.mixture import GaussianMixture

import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
import gabriel_pinned_registration as gabriel

SOURCE=ROOT/"outputs/gabriel_globalmean_field_pca_gmm"
DATA=ROOT/"data/airfoils/processed_gabriel/airfoils_pinned_arclength_n200.npz"
OUT=ROOT/"outputs/gabriel_big_registration_k2_k7_full_leftmost"
DEV_CONFIG=ROOT/"outputs/gabriel_pinned_registration_dev/config.json"
PAIRS=((2,1),(3,89),(4,5),(5,3),(6,2),(7,5));CONFIG=gabriel.RegistrationConfig()
SEED,N_INIT,REG_COVAR,MAX_ITER,TOL=20260813,5,1e-7,500,1e-5
DIAG=101;DATA_SHA="f357610c338b7a820c1f09522052d29951e9e62e41d3faffc61de19621995e33"

def branch_id(K,q):return f"K{K:02d}_q{q:02d}"
def sha_arr(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def atomic_json(p,x):gabriel.atomic_json(p,x)
def atomic_csv(p,rows,fields=None):
 rows=list(rows);p.parent.mkdir(parents=True,exist_ok=True);fields=fields or list(dict.fromkeys(k for r in rows for k in r));t=p.with_suffix(p.suffix+".tmp")
 with t.open("w",newline="",encoding="utf8") as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows([{k:json.dumps(gabriel.jsonable(v),separators=(",",":")) if isinstance(v,(list,dict,tuple,np.ndarray)) else v for k,v in r.items()} for r in rows])
 os.replace(t,p)
def read_csv(p):
 with p.open(newline="",encoding="utf8") as f:return list(csv.DictReader(f))
def protected_fingerprint():
 bases=[SOURCE,ROOT/"outputs/gabriel_k2_k8_energy_magnitude",ROOT/"outputs/gabriel_modal_kmeans_registration",ROOT/"outputs/gabriel_gmm_template_registration"]
 rows=[]
 for b in bases:
  for p in sorted(b.rglob("*")):
   if p.is_file() and p.name!=".DS_Store":s=p.stat();rows.append((str(p.relative_to(ROOT)),s.st_size,s.st_mtime_ns))
 return {"algorithm":"sha256(path,size,mtime_ns)","file_count":len(rows),"digest":hashlib.sha256(json.dumps(rows).encode()).hexdigest()}

def load_base():
 if gabriel.sha256_file(DATA)!=DATA_SHA:raise RuntimeError("STOP dataset SHA mismatch")
 with np.load(DATA,allow_pickle=False) as z:contours=z["contours"].copy();names=z["filenames"].astype(str)
 if contours.shape!=(1652,200,2) or not np.isfinite(contours).all():raise RuntimeError("STOP Gabriel data mismatch")
 if json.loads(DEV_CONFIG.read_text())["registration_config"]!=json.loads(json.dumps(asdict(CONFIG))):raise RuntimeError("STOP registration config drift")
 with np.load(SOURCE/"pca/pca_q100.npz",allow_pickle=False) as z:
  pca={k:z[k].copy() for k in z.files}
 train=np.load(SOURCE/"splits/train_indices_valid_used.npy");held=np.load(SOURCE/"splits/heldout_indices_valid_used.npy")
 # Reconstruct all 1652 old global-mean fields, including the one diagnostically invalid registration.
 X=[]
 for i in range(1652):
  p=SOURCE/"registrations"/f"{i:04d}.npz"
  with np.load(p,allow_pickle=False) as z:c=z["coefficients"].copy();m=json.loads(str(z["metrics_json"]))
  if c.shape!=(30,30,2) or not np.isfinite(c).all() or int(m["target_index"])!=i or m["filename"]!=names[i]:raise RuntimeError(f"STOP old coefficient mismatch {i}")
  X.append(c.reshape(-1,order="C"))
 X=np.asarray(X);scores=(X-pca["mean"])@pca["components"].T
 if not np.array_equal(train,pca["train_target_indices"]) or not np.array_equal(held,pca["heldout_target_indices"]):raise RuntimeError("STOP split mismatch")
 if not np.allclose(scores[train],pca["train_scores"],rtol=0,atol=5e-13) or not np.allclose(scores[held],pca["heldout_scores"],rtol=0,atol=5e-13):raise RuntimeError("STOP PCA score reproduction mismatch")
 return contours,names,X,scores,train,held

def template_diag(t):
 d=gabriel.validate_pinned_contour(t);segments=np.linalg.norm(np.roll(t,-1,axis=0)-t,axis=1);chord=float(t[:,0].max()-t[:,0].min())
 critical=bool(t.shape==(200,2) and np.isfinite(t).all() and segments.min()>1e-12 and abs(t[:,0].min())<1e-10 and abs(t[:,0].max()-1)<.03 and gabriel.signed_polygon_area(t)>1e-8 and gabriel.self_intersection_count(t)==0 and d["index0_is_te"] and d["index100_is_le"] and d["upper_is_above_lower"] and d["upper_expected_direction_fraction"]==1 and d["lower_expected_direction_fraction"]==1)
 return {"valid":critical,"shape":list(t.shape),"finite":bool(np.isfinite(t).all()),"minimum_segment_length":float(segments.min()),"chord":chord,"signed_area":gabriel.signed_polygon_area(t),"self_intersections":gabriel.self_intersection_count(t),"pinned_diagnostics":d}

def prepare():
 contours,names,X,scores,train,held=load_base();pref=read_csv(SOURCE/"gmm/preferred_k_by_q.csv");grid=read_csv(SOURCE/"gmm/full_model_selection_grid.csv")
 for d in ("selections","assignments","templates","registrations","coefficient_matrices","tables","figures","report","logs","smoke"):(OUT/d).mkdir(parents=True,exist_ok=True)
 selected=[];audits=[]
 for K in range(2,8):
  hits=[]
  for r in pref:
   q=int(r["q"]);b=int(r["preferred_k_bic"])==K;i=int(r["preferred_k_icl"])==K
   if b or i:hits.append((q,b,i))
  if not hits:raise RuntimeError(f"STOP no full-range selection K={K}")
  q,b,i=hits[0]
  if (K,q) not in PAIRS:raise RuntimeError(f"STOP audited pair discrepancy K={K}, q={q}")
  selected.append({"K":K,"selected_q":q,"BIC_selects_K":b,"ICL_selects_K":i,"rule":"smallest q in 1..100 where BIC and/or ICL preferred K"})
  model=GaussianMixture(n_components=K,covariance_type="full",n_init=N_INIT,reg_covar=REG_COVAR,max_iter=MAX_ITER,tol=TOL,init_params="kmeans",random_state=SEED).fit(scores[train,:q])
  resp=model.predict_proba(scores[train,:q]);entropy=float(-np.sum(resp*np.log(np.clip(resp,np.finfo(float).tiny,1))))
  got=(float(model.bic(scores[train,:q])),float(model.bic(scores[train,:q])+2*entropy),float(model.score(scores[held,:q])))
  saved=next(r for r in grid if int(r["q"])==q and int(r["K"])==K);want=tuple(float(saved[x]) for x in ("BIC","ICL","heldout_log_likelihood_per_sample"))
  if not np.allclose(got,want,rtol=2e-12,atol=2e-7):raise RuntimeError(f"STOP GMM numerical reproduction K={K},q={q}: {got} != {want}")
  labels=model.predict(scores[:,:q]).astype(np.int64);sizes=np.bincount(labels,minlength=K)
  if len(labels)!=1652 or sizes.sum()!=1652 or np.any(sizes==0):raise RuntimeError("STOP assignment completeness")
  bid=branch_id(K,q);rows=[]
  for idx,(name,label) in enumerate(zip(names,labels)):rows.append({"target_index":idx,"filename":name,"cluster_id":int(label),"K":K,"q":q})
  atomic_csv(OUT/"assignments"/f"{bid}.csv",rows);np.save(OUT/"assignments"/f"{bid}_labels.npy",labels)
  tdir=OUT/"templates"/bid;tdir.mkdir(parents=True,exist_ok=True);templates=[];valid=[]
  fig,ax=plt.subplots(figsize=(11,5.7))
  for c in range(K):
   members=np.flatnonzero(labels==c);t=contours[members].mean(0);diag=template_diag(t);templates.append(t);valid.append(diag["valid"])
   gabriel.atomic_npz(tdir/f"template_{c:02d}.npz",coordinates=t,member_indices=members,member_filenames=names[members],diagnostics_json=np.asarray(json.dumps(gabriel.jsonable(diag),sort_keys=True)),template_sha256=np.asarray(sha_arr(t)))
   ax.plot(np.r_[t[:,0],t[0,0]],np.r_[t[:,1],t[0,1]],label=f"c{c} (n={len(members)})")
  if not all(valid):raise RuntimeError(f"STOP invalid template {bid}")
  ax.set_aspect("equal");ax.set(xlabel="x/chord",ylabel="y/chord",title=f"Gabriel cluster-mean templates — K={K}, q={q}");ax.grid(alpha=.2);ax.legend(ncol=min(4,K),fontsize=8);fig.tight_layout();fig.savefig(OUT/"figures"/f"templates_{bid}.png",dpi=280,bbox_inches="tight");plt.close(fig)
  gabriel.atomic_npz(tdir/"all_templates.npz",coordinates=np.asarray(templates),cluster_sizes=sizes,labels=labels,filenames=names)
  audits.append({"K":K,"q":q,"total_samples":1652,"cluster_sizes":sizes.tolist(),"cluster_sizes_ascending":np.sort(sizes).tolist(),"smallest_cluster":int(sizes.min()),"smallest_percent":100*float(sizes.min())/1652,"largest_cluster":int(sizes.max()),"largest_percent":100*float(sizes.max())/1652,"clusters_lt20":int(np.sum(sizes<20)),"clusters_lt1pct":int(np.sum(sizes<16.52)),"component_weights":model.weights_.tolist(),"templates":K,"all_templates_valid":all(valid),"GMM_converged":bool(model.converged_),"reproduced_BIC":got[0],"reproduced_ICL":got[1],"reproduced_heldout_LL":got[2]})
 atomic_csv(OUT/"selections/selected_kq_pairs_k2_k7_full_leftmost.csv",selected);atomic_csv(OUT/"tables/assignment_template_audit.csv",audits)
 atomic_json(OUT/"config.json",{"experiment":"Gabriel big registration K=2-7 full-range leftmost extension","source_experiment":str(SOURCE.relative_to(ROOT)),"dataset":str(DATA.relative_to(ROOT)),"dataset_sha256":DATA_SHA,"pairs":[list(x) for x in PAIRS],"selection_rule":"smallest q in 1..100 where BIC and/or ICL preferred K","features":"PCA scores of existing single-global-mean Gabriel B-spline coefficients","GMM":{"covariance_type":"full","n_init":N_INIT,"reg_covar":REG_COVAR,"max_iter":MAX_ITER,"tol":TOL,"init_params":"kmeans","random_state":SEED},"registration":asdict(CONFIG),"workers":4,"diagnostic_resolution":DIAG,"coefficient_shape":[30,30,2],"flatten":"C order control_x,control_y,component[x,y]","protected_before":protected_fingerprint(),"second_stage_PCA_GMM_run":False})
 return audits

def geo(deformed):
 finite=bool(np.isfinite(deformed).all());area=gabriel.signed_polygon_area(deformed);inter=gabriel.self_intersection_count(deformed);return finite,area,inter,bool(finite and area>1e-8 and inter==0)
def run_one(payload):
 K,q,c,i,name,template,target,path_text,template_sha=payload;torch.set_num_threads(1)
 try:torch.set_num_interop_threads(1)
 except RuntimeError:pass
 start=time.perf_counter();base={"K":K,"q":q,"target_index":i,"filename":name,"cluster_id":c,"template_id":f"{branch_id(K,q)}_cluster{c:02d}","template_sha256":template_sha,"coefficient_tensor_shape":[30,30,2],"flattened_coefficient_length":1800,"flattening_convention":"C order: control_x, control_y, component [x,y]"}
 try:
  r=gabriel.optimise_registration(template,target,CONFIG);m=gabriel.correspondence_metrics(r["deformed"],target);g=gabriel.diagnostic_grid(r["coefficients"],r["control_x"],r["control_y"],CONFIG.rk_steps,DIAG);finite,area,inter,gv=geo(r["deformed"]);valid=bool(r["status"]=="completed" and r["finite"] and gv and g["jacobian_minimum"]>0 and g["jacobian_nonpositive_fraction"]==0)
  row={**base,"status":r["status"],"failure_reason":r["failure_reason"],"converged":r["status"]=="completed","valid_registration":valid,"geometry_valid":gv,"runtime_seconds":time.perf_counter()-start,"optimisation_runtime_seconds":r["runtime_seconds"],"closure_evaluations":r["closure_evaluations"],"correspondence_mse":m["pointwise_mse"],"correspondence_rmse":m["pointwise_rmse"],"maximum_pointwise_error":m["maximum_correspondence_error"],"physical_dirichlet_energy":g["physical_dirichlet_integral"],"velocity_mean_magnitude":g["velocity_mean_magnitude"],"velocity_rms_magnitude":g["velocity_rms_magnitude"],"minimum_jacobian":g["jacobian_minimum"],"nonpositive_jacobian_fraction":g["jacobian_nonpositive_fraction"],"signed_area":area,"self_intersections":inter,"coefficient_smoothness":r["final_coefficient_smoothness"]}
  coeff=r["coefficients"]
  gabriel.atomic_npz(Path(path_text),metrics_json=np.asarray(json.dumps(gabriel.jsonable(row),sort_keys=True)),coefficients=coeff,flattened_coefficients=coeff.reshape(-1,order="C"),control_x=r["control_x"],control_y=r["control_y"],deformed=r["deformed"])
 except Exception as e:
  row={**base,"status":"failed","failure_reason":f"{type(e).__name__}:{e}","converged":False,"valid_registration":False,"geometry_valid":False,"runtime_seconds":time.perf_counter()-start};gabriel.atomic_npz(Path(path_text),metrics_json=np.asarray(json.dumps(row,sort_keys=True)))
 return row
def load_ckpt(p,K,q,i,name,c,tsha):
 try:
  with np.load(p,allow_pickle=False) as z:r=json.loads(str(z["metrics_json"]));ok=(int(r["K"]),int(r["q"]),int(r["target_index"]),r["filename"],int(r["cluster_id"]),r["template_sha256"])==(K,q,i,name,c,tsha)
  if not ok:return None
  if r["status"]=="completed":
   with np.load(p,allow_pickle=False) as z:
    if z["coefficients"].shape!=(30,30,2) or z["flattened_coefficients"].shape!=(1800,) or not np.isfinite(z["coefficients"]).all():return None
  return r
 except Exception:return None
def stats(v):
 a=np.asarray(v,float);n=len(a);sd=float(a.std(ddof=1)) if n>1 else math.nan
 return {"mean":float(a.mean()),"median":float(np.median(a)),"p95":float(np.quantile(a,.95)),"sd":sd,"se":sd/math.sqrt(n) if n>1 else math.nan}

def run_branch(K,q,workers=4,smoke=False):
 contours,names,_,_,_,_=load_base();bid=branch_id(K,q);labels=np.load(OUT/"assignments"/f"{bid}_labels.npy");
 with np.load(OUT/"templates"/bid/"all_templates.npz",allow_pickle=False) as z:templates=z["coordinates"].copy()
 indices=[0,int(np.argmax(np.linalg.norm((contours-contours.mean(0)).reshape(1652,-1),axis=1)))] if smoke else list(range(1652));base=OUT/("smoke" if smoke else "registrations")/bid;base.mkdir(parents=True,exist_ok=True);rows=[];todo=[]
 for i in indices:
  c=int(labels[i]);tsha=sha_arr(templates[c]);p=base/f"{i:04d}.npz";old=load_ckpt(p,K,q,i,str(names[i]),c,tsha)
  if old is not None:rows.append(old)
  else:todo.append((K,q,c,i,str(names[i]),templates[c],contours[i],str(p),tsha))
 start=time.perf_counter()
 if workers==1:iterator=(run_one(x) for x in todo)
 else:
  pool=ProcessPoolExecutor(max_workers=workers);futures=[pool.submit(run_one,x) for x in todo];iterator=(f.result() for f in as_completed(futures))
 new=0
 try:
  for r in iterator:
   rows.append(r);new+=1
   if new==1 or new%25==0 or new==len(todo):
    elapsed=time.perf_counter()-start;rate=new/elapsed if elapsed else 0;print(f"{bid} {len(rows)}/{len(indices)} rate={rate:.3f}/s invalid={sum(not bool(x.get('valid_registration')) for x in rows)}",flush=True)
 finally:
  if workers!=1:pool.shutdown()
 rows.sort(key=lambda x:int(x["target_index"]));atomic_csv(OUT/"tables"/(f"smoke_{bid}.csv" if smoke else f"per_registration_{bid}.csv"),rows)
 if smoke:
  ok=all(r["status"]=="completed" and r["flattened_coefficient_length"]==1800 for r in rows);atomic_json(OUT/"logs"/f"smoke_{bid}.json",{"status":"PASS" if ok else "FAIL","rows":rows,"wall_seconds":time.perf_counter()-start,"projected_branch_hours_4workers":(time.perf_counter()-start)/max(new,1)*1652/3600});
  if not ok:raise RuntimeError(f"STOP smoke failure {bid}")
  return
 finalise_branch(K,q,contours,names,labels,templates,rows,time.perf_counter()-start)

def finalise_branch(K,q,contours,names,labels,templates,rows,wall):
 bid=branch_id(K,q);bad=[];ordered=[];X=[];valid=[]
 for i in range(1652):
  c=int(labels[i]);p=OUT/"registrations"/bid/f"{i:04d}.npz";r=load_ckpt(p,K,q,i,str(names[i]),c,sha_arr(templates[c]))
  if r is None:bad.append(i);continue
  with np.load(p,allow_pickle=False) as z:flat=z["flattened_coefficients"].copy()
  if flat.shape!=(1800,) or not np.isfinite(flat).all():bad.append(i);continue
  ordered.append(r);X.append(flat);valid.append(bool(r["valid_registration"]))
 if bad:raise RuntimeError(f"STOP incomplete {bid}: {bad[:20]}")
 X=np.asarray(X);gabriel.atomic_npz(OUT/"coefficient_matrices"/bid/"registration_coefficients.npz",X_field=X,target_indices=np.arange(1652),filenames=names,labels=labels,coefficient_tensor_shape=np.asarray([30,30,2]),flatten_order=np.asarray("C: control_x,control_y,component[x,y]"),source_representation=np.asarray("Gabriel assigned cluster-mean template registration B-spline coefficients"))
 vr=[r for r in ordered if r["valid_registration"]];m=stats([r["correspondence_mse"] for r in vr]);d=stats([r["physical_dirichlet_energy"] for r in vr]);v=stats([r["velocity_mean_magnitude"] for r in vr])
 s={"K":K,"q":q,"total":1652,"completed":sum(r["status"]=="completed" for r in ordered),"valid":len(vr),"invalid":sum(r["status"]=="completed" and not r["valid_registration"] for r in ordered),"failed":sum(r["status"]!="completed" for r in ordered),"corrupted_or_missing":0,"mean_MSE":m["mean"],"median_MSE":m["median"],"p95_MSE":m["p95"],"minimum_Jacobian":min(float(r["minimum_jacobian"]) for r in ordered if r["status"]=="completed"),"mean_Dirichlet":d["mean"],"SE_Dirichlet":d["se"],"mean_SVF_magnitude":v["mean"],"SE_SVF_magnitude":v["se"],"coefficient_shape":[1652,1800],"wall_seconds_new_run":wall,"sample_order_preserved":all(int(r["target_index"])==i and r["filename"]==names[i] for i,r in enumerate(ordered))}
 atomic_json(OUT/"tables"/f"summary_{bid}.json",s);return s

def complete_report():
 summaries=[json.loads((OUT/"tables"/f"summary_{branch_id(K,q)}.json").read_text()) for K,q in PAIRS];audits=read_csv(OUT/"tables/assignment_template_audit.csv");checks={"six_branches":len(summaries)==6,"each_1652":all(s["total"]==1652 and s["completed"]+s["failed"]==1652 for s in summaries),"no_missing":all(s["corrupted_or_missing"]==0 for s in summaries),"six_matrices":all((OUT/"coefficient_matrices"/branch_id(K,q)/"registration_coefficients.npz").exists() for K,q in PAIRS),"all_ordered":all(s["sample_order_preserved"] for s in summaries),"all_templates_valid":all(r["all_templates_valid"]=="True" for r in audits),"protected_unchanged":json.loads((OUT/"config.json").read_text())["protected_before"]==protected_fingerprint(),"second_stage_PCA_GMM_not_run":not (OUT/"second_stage_pca").exists() and not (OUT/"second_stage_gmm").exists()}
 atomic_json(OUT/"validation.json",{"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"summaries":summaries})
 lines=["# Gabriel big registration K=2–7 — full-range leftmost extension","",f"Validation: **{'PASS' if all(checks.values()) else 'FAIL'}**.","","No second-stage PCA/GMM was run.","","## Branches","","| K | q | completed | valid | invalid | failed | mean Dirichlet ± SE | mean SVF magnitude ± SE | wall hours |","|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
 for s in summaries:lines.append(f"| {s['K']} | {s['q']} | {s['completed']} | {s['valid']} | {s['invalid']} | {s['failed']} | {s['mean_Dirichlet']:.8g} ± {s['SE_Dirichlet']:.3g} | {s['mean_SVF_magnitude']:.8g} ± {s['SE_SVF_magnitude']:.3g} | {s['wall_seconds_new_run']/3600:.3f} |")
 lines += ["","Each of the 1,652 targets in each branch was registered from its assigned pointwise cluster-mean Gabriel contour. Six ordered `(1652,1800)` B-spline coefficient matrices are ready for the separate next-stage PCA/GMM task."]
 (OUT/"report/final_report.md").write_text("\n".join(lines)+"\n",encoding="utf8")

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--stage",choices=("prepare","smoke","run","finalise","all"),default="all");ap.add_argument("--workers",type=int,default=4);ap.add_argument("--branch",choices=[branch_id(*x) for x in PAIRS]);a=ap.parse_args()
 if a.stage in ("prepare","all"):prepare()
 pairs=[x for x in PAIRS if not a.branch or branch_id(*x)==a.branch]
 if a.stage in ("smoke","all"):
  for K,q in pairs:run_branch(K,q,min(a.workers,2),True)
 if a.stage in ("run","all"):
  for K,q in pairs:run_branch(K,q,a.workers,False)
 if a.stage in ("finalise","all"):complete_report()
if __name__=="__main__":main()
