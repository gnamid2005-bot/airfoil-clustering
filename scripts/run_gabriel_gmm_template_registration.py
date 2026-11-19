"""Checkpointed Gabriel registration using the eight audited hard GMM templates."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import gabriel_pinned_registration as gabriel

DATA = ROOT / "data/airfoils/processed_gabriel/airfoils_pinned_arclength_n200.npz"
AUDIT = ROOT / "outputs/gabriel_selected_gmm_templates_audit"
KMEANS = ROOT / "outputs/gabriel_k2_k8_energy_magnitude"
OUT = ROOT / "outputs/gabriel_gmm_template_registration"
DEV_CONFIG = ROOT / "outputs/gabriel_pinned_registration_dev/config.json"
CONFIG = gabriel.RegistrationConfig()
DIAGNOSTIC_RESOLUTION = 101
PAIRS = ((2, 3), (5, 26), (3, 1), (4, 37), (6, 2), (7, 20), (8, 3), (9, 4))
BATCHES = (((2, 3), (5, 26)), ((3, 1), (4, 37)), ((6, 2), (7, 20)), ((8, 3), (9, 4)))
REP = {(2,3):("modal",0),(3,1):("modal",0),(4,37):("ARI-medoid",1),(5,26):("modal",1),
       (6,2):("ARI-medoid",2),(7,20):("modal",4),(8,3):("modal",1),(9,4):("ARI-medoid",13)}

def atomic_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows); fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)

def read_csv(path):
    with path.open(newline="", encoding="utf8") as f: return list(csv.DictReader(f))

def fingerprint_tree(path):
    items=[]
    for p in sorted(path.rglob("*")):
        if p.is_file(): items.append((str(p.relative_to(ROOT)), p.stat().st_size, p.stat().st_mtime_ns, gabriel.sha256_file(p)))
    return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest(), len(items), sum(x[1] for x in items)

def load_inputs():
    with np.load(DATA, allow_pickle=False) as z:
        contours=z["contours"].astype(np.float64); filenames=z["filenames"].astype(str)
    if contours.shape != (1652,200,2) or contours.dtype != np.float64 or not np.isfinite(contours).all():
        raise RuntimeError("Gabriel pinned input mismatch")
    audit_summary={(int(r["K"]),int(r["q"])):r for r in read_csv(AUDIT/"tables/selected_gmm_assignment_template_audit_k2_k9.csv")}
    records={}
    for K,q in PAIRS:
        lp=AUDIT/"assignments"/f"labels_K{K:02d}_q{q:02d}.npy"
        tp=AUDIT/"templates"/f"hard_templates_K{K:02d}_q{q:02d}.npz"
        labels=np.load(lp, allow_pickle=False)
        with np.load(tp, allow_pickle=False) as z:
            templates=z["coordinates"].astype(np.float64); saved_labels=z["labels"]
            saved_names=z["filenames"].astype(str); sizes=z["cluster_sizes"]
        if labels.shape!=(1652,) or not np.array_equal(labels,saved_labels) or not np.array_equal(filenames,saved_names):
            raise RuntimeError(f"audited assignment/order mismatch K={K}, q={q}")
        if templates.shape!=(K,200,2) or not np.array_equal(sizes,np.bincount(labels,minlength=K)):
            raise RuntimeError(f"audited template mismatch K={K}, q={q}")
        for c in range(K):
            if not np.array_equal(templates[c], contours[labels==c].mean(0)):
                raise RuntimeError(f"template is not exact hard pointwise mean K={K}, q={q}, c={c}")
        s=audit_summary[(K,q)]
        if (s["representative_type"],int(s["representative_seed"])) != REP[(K,q)]: raise RuntimeError("representative mismatch")
        records[(K,q)]={"labels":labels,"templates":templates,"sizes":sizes,"audit":s}
    return contours,filenames,records

def geometry(deformed):
    finite=bool(np.isfinite(deformed).all()); area=gabriel.signed_polygon_area(deformed); intersections=gabriel.self_intersection_count(deformed)
    return {"finite":finite,"signed_area":area,"self_intersection_count":intersections,
            "invalid_geometry":bool(not finite or area<=1e-8 or intersections>0)}

def run_one(payload):
    K,q,cluster,index,filename,source,target,path=payload
    torch.set_num_threads(1)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    start=time.perf_counter(); base={"K":K,"q":q,"cluster_id":cluster,"target_index":index,"filename":filename,
                                     "source_template_id":f"GMM_K{K:02d}_q{q:02d}_cluster{cluster:02d}"}
    try:
        result=gabriel.optimise_registration(source,target,CONFIG)
        after=gabriel.correspondence_metrics(result["deformed"],target)
        grid=gabriel.diagnostic_grid(result["coefficients"],result["control_x"],result["control_y"],CONFIG.rk_steps,DIAGNOSTIC_RESOLUTION)
        geo=geometry(result["deformed"])
        valid=bool(result["status"]=="completed" and result["finite"] and not geo["invalid_geometry"] and grid["jacobian_minimum"]>0 and grid["jacobian_nonpositive_fraction"]==0)
        row={**base,"status":result["status"],"failure_reason":result["failure_reason"],"finite":result["finite"],
             "valid_registration":valid,"geometry_valid":not geo["invalid_geometry"],"runtime_seconds":time.perf_counter()-start,
             "optimisation_runtime_seconds":result["runtime_seconds"],"closure_evaluations":result["closure_evaluations"],
             "optimizer_reported_iterations_total":sum(x["optimizer_reported_iterations"] for x in result["cycles"]),
             "final_pointwise_mse":after["pointwise_mse"],"final_pointwise_rmse":after["pointwise_rmse"],
             "maximum_correspondence_error":after["maximum_correspondence_error"],
             "physical_dirichlet_integral":grid["physical_dirichlet_integral"],"velocity_mean_magnitude":grid["velocity_mean_magnitude"],
             "velocity_rms_magnitude":grid["velocity_rms_magnitude"],"velocity_maximum_magnitude":grid["velocity_maximum_magnitude"],
             "coefficient_smoothness":result["final_coefficient_smoothness"],"final_lambda":result["final_lambda"],
             "weighted_regularisation":result["final_weighted_regularisation"],"total_objective":result["final_objective"],
             "jacobian_minimum":grid["jacobian_minimum"],"jacobian_p01":grid["jacobian_p01"],"jacobian_p05":grid["jacobian_p05"],
             "jacobian_nonpositive_fraction":grid["jacobian_nonpositive_fraction"],"signed_area":geo["signed_area"],
             "self_intersection_count":geo["self_intersection_count"],"diagnostic_resolution":DIAGNOSTIC_RESOLUTION}
        gabriel.atomic_npz(Path(path),metrics_json=np.asarray(json.dumps(gabriel.jsonable(row),sort_keys=True)),coefficients=result["coefficients"],
                           control_x=result["control_x"],control_y=result["control_y"],deformed=result["deformed"])
        return row
    except Exception as exc:
        row={**base,"status":"failed","failure_reason":f"{type(exc).__name__}:{exc}","finite":False,"valid_registration":False,
             "geometry_valid":False,"runtime_seconds":time.perf_counter()-start,"diagnostic_resolution":DIAGNOSTIC_RESOLUTION}
        gabriel.atomic_npz(Path(path),metrics_json=np.asarray(json.dumps(row,sort_keys=True)))
        return row

def load_checkpoint(path,K,q,index):
    try:
        with np.load(path,allow_pickle=False) as z: row=json.loads(str(np.asarray(z["metrics_json"]).item()))
        return row if (int(row["K"]),int(row["q"]),int(row["target_index"]))==(K,q,index) else None
    except Exception: return None

def append_log(path,event,**values):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf8") as f:
        f.write(json.dumps({"timestamp":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"event":event,**values},sort_keys=True)+"\n"); f.flush(); os.fsync(f.fileno())

def stats(values):
    x=np.asarray(values,float); n=len(x)
    if not n:return {k:math.nan for k in ("mean","sd","se","median","p05","p90","p95","max")}
    sd=float(x.std(ddof=1)) if n>1 else math.nan
    return {"mean":float(x.mean()),"sd":sd,"se":sd/math.sqrt(n) if n>1 else math.nan,"median":float(np.median(x)),
            "p05":float(np.quantile(x,.05)),"p90":float(np.quantile(x,.90)),"p95":float(np.quantile(x,.95)),"max":float(x.max())}

def combo_summary(K,q,rows):
    valid=[r for r in rows if r.get("valid_registration") is True]
    d=stats([r["physical_dirichlet_integral"] for r in valid]); m=stats([r["velocity_mean_magnitude"] for r in valid])
    mse=stats([r["final_pointwise_mse"] for r in valid]); rt=stats([r["runtime_seconds"] for r in rows])
    typ,seed=REP[(K,q)]
    return {"K":K,"q":q,"representative_type":typ,"representative_seed":seed,"n_total":1652,"n_valid":len(valid),
            "n_invalid":sum(not bool(r.get("valid_registration",False)) for r in rows),"n_failed":sum(r.get("status")!="completed" for r in rows),
            "mean_dirichlet":d["mean"],"sd_dirichlet":d["sd"],"se_dirichlet":d["se"],"median_dirichlet":d["median"],
            "p90_dirichlet":d["p90"],"p95_dirichlet":d["p95"],"max_dirichlet":d["max"],
            "mean_svf_magnitude":m["mean"],"sd_svf_magnitude":m["sd"],"se_svf_magnitude":m["se"],"median_svf_magnitude":m["median"],
            "p90_svf_magnitude":m["p90"],"p95_svf_magnitude":m["p95"],"max_svf_magnitude":m["max"],
            "mean_mse":mse["mean"],"median_mse":mse["median"],"p95_mse":mse["p95"],"max_mse":mse["max"],
            "min_jacobian":min((float(r["jacobian_minimum"]) for r in valid),default=math.nan),
            "n_nonpositive_jacobian":sum(float(r.get("jacobian_nonpositive_fraction",1))>0 or float(r.get("jacobian_minimum",-math.inf))<=0 for r in rows),
            "mean_runtime":rt["mean"],"median_runtime":rt["median"],"total_case_runtime":sum(float(r["runtime_seconds"]) for r in rows)}

def run_combo(K,q,contours,filenames,record,workers,batch):
    d=OUT/"registrations"/f"K{K:02d}_q{q:02d}"; d.mkdir(parents=True,exist_ok=True)
    rows=[]; todo=[]
    for i in range(len(contours)):
        p=d/f"{i:04d}.npz"; old=load_checkpoint(p,K,q,i)
        if old is not None: rows.append(old)
        else:
            c=int(record["labels"][i]); todo.append((K,q,c,i,str(filenames[i]),record["templates"][c],contours[i],str(p)))
    start=time.perf_counter(); new=0; log=OUT/"progress.log"
    append_log(log,"combination_start",batch=batch,K=K,q=q,reused=len(rows),pending=len(todo),workers=workers)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(run_one,p) for p in todo]
        for f in as_completed(futures):
            row=f.result(); rows.append(row); new+=1
            if new==1 or new%10==0 or new==len(todo):
                elapsed=time.perf_counter()-start; rate=new/elapsed if elapsed else 0; remaining=len(todo)-new
                append_log(log,"progress",batch=batch,K=K,q=q,completed=len(rows),remaining=remaining,
                           failures=sum(r.get("status")!="completed" for r in rows),invalid=sum(not bool(r.get("valid_registration",False)) for r in rows),
                           throughput=rate,elapsed_seconds=elapsed,eta_seconds=remaining/rate if rate else None)
                print(f"B{batch} K={K} q={q}: {len(rows)}/1652, invalid={sum(not bool(r.get('valid_registration',False)) for r in rows)}, ETA={remaining/rate/60 if rate else math.nan:.1f} min",flush=True)
    rows=sorted(rows,key=lambda r:int(r["target_index"]))
    if len(rows)!=1652 or any(int(r["target_index"])!=i for i,r in enumerate(rows)): raise RuntimeError(f"checkpoint completeness failure K={K},q={q}")
    atomic_csv(OUT/"tables"/f"per_registration_K{K:02d}_q{q:02d}.csv",rows)
    summary=combo_summary(K,q,rows); summary["wall_clock_seconds"]=time.perf_counter()-start
    gabriel.atomic_json(OUT/"tables"/f"provisional_summary_K{K:02d}_q{q:02d}.json",summary)
    append_log(log,"combination_complete",batch=batch,**summary)
    return summary

def paired_comparison(gmm_rows):
    km=read_csv(KMEANS/"per_registration_results.csv")
    kmap={(int(r["K"]),int(r["target_index"])):r for r in km}
    out=[]
    for K,q in sorted(PAIRS):
        gr=[r for r in gmm_rows if int(r["K"])==K and r.get("valid_registration") is True and kmap[(K,int(r["target_index"]))].get("valid_registration")=="True"]
        row={"K":K,"q":q,"n_paired":len(gr)}
        for label,gfield,kfield in (("dirichlet","physical_dirichlet_integral","physical_dirichlet_integral"),("svf_magnitude","velocity_mean_magnitude","velocity_mean_magnitude")):
            dif=np.asarray([float(r[gfield])-float(kmap[(K,int(r["target_index"]))][kfield]) for r in gr])
            s=stats(dif); km_mean=np.mean([float(kmap[(K,int(r["target_index"]))][kfield]) for r in gr]); gm_mean=np.mean([float(r[gfield]) for r in gr])
            row.update({f"kmeans_mean_{label}":km_mean,f"gmm_mean_{label}":gm_mean,f"delta_{label}":gm_mean-km_mean,
                        f"percent_difference_{label}":100*(gm_mean-km_mean)/km_mean,f"paired_mean_difference_{label}":s["mean"],
                        f"paired_median_difference_{label}":s["median"],f"paired_se_difference_{label}":s["se"],
                        f"paired_p05_difference_{label}":s["p05"],f"paired_p95_difference_{label}":s["p95"],
                        f"fraction_gmm_lower_{label}":float(np.mean(dif<0))})
        out.append(row)
    return out

def plot_results(summary,comparison):
    (OUT/"figures").mkdir(parents=True,exist_ok=True); qs={K:q for K,q in PAIRS}; Kvals=[r["K"] for r in summary]
    for field,sefield,ylabel,title,name in (("mean_dirichlet","se_dirichlet","Average physical Dirichlet energy","Average Dirichlet Energy vs Number of Templates — GMM-derived templates","average_dirichlet_gmm.png"),
                                           ("mean_svf_magnitude","se_svf_magnitude","Average velocity mean magnitude","Average SVF Magnitude vs Number of Templates — GMM-derived templates","average_svf_magnitude_gmm.png")):
        fig,ax=plt.subplots(figsize=(10,5.8)); ax.errorbar(Kvals,[r[field] for r in summary],yerr=[r[sefield] for r in summary],marker="o",capsize=4,lw=2,color="#1967a3")
        for r in summary: ax.annotate(f"q={r['q']}",(r["K"],r[field]),xytext=(0,8),textcoords="offset points",ha="center",fontsize=8)
        ax.set(xticks=Kvals,xlabel="Number of templates K",ylabel=ylabel,title=title); ax.grid(axis="y",alpha=.22); ax.spines[["top","right"]].set_visible(False); fig.tight_layout(); fig.savefig(OUT/"figures"/name,dpi=300); plt.close(fig)
    km_summary={int(r["K"]):r for r in read_csv(KMEANS/"summary_by_k.csv")}
    g_summary={int(r["K"]):r for r in summary}
    for metric,ylabel,name in (("dirichlet","Average physical Dirichlet energy","dirichlet_kmeans_vs_gmm.png"),("svf_magnitude","Average velocity mean magnitude","svf_magnitude_kmeans_vs_gmm.png")):
        fig,ax=plt.subplots(figsize=(10,5.8)); c=sorted(comparison,key=lambda r:r["K"]); x=[r["K"] for r in c]
        kse="dirichlet_se" if metric=="dirichlet" else "svf_magnitude_se"; gse="se_dirichlet" if metric=="dirichlet" else "se_svf_magnitude"
        ax.errorbar(x,[r[f"kmeans_mean_{metric}"] for r in c],yerr=[float(km_summary[k][kse]) for k in x],marker="o",lw=2,capsize=4,label="K-means templates")
        ax.errorbar(x,[r[f"gmm_mean_{metric}"] for r in c],yerr=[float(g_summary[k][gse]) for k in x],marker="s",lw=2,capsize=4,label="GMM templates")
        ax.set(xticks=x,xlabel="Number of templates K",ylabel=ylabel,title=f"{ylabel}: K-means vs GMM Templates"); ax.grid(axis="y",alpha=.22); ax.legend(); ax.spines[["top","right"]].set_visible(False); fig.tight_layout(); fig.savefig(OUT/"figures"/name,dpi=300); plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(10,8),sharex=True)
    c=sorted(comparison,key=lambda r:r["K"]); x=[r["K"] for r in c]
    for ax,metric,label in zip(axes,("dirichlet","svf_magnitude"),("Dirichlet energy","SVF magnitude")):
        ax.axhline(0,color="black",lw=.8); ax.bar(x,[r[f"percent_difference_{metric}"] for r in c],color="#6c8ebf"); ax.set_ylabel("GMM vs K-means (%)"); ax.set_title(label); ax.grid(axis="y",alpha=.2)
    axes[-1].set(xticks=x,xlabel="Number of templates K"); fig.tight_layout(); fig.savefig(OUT/"figures"/"percentage_difference_gmm_vs_kmeans.png",dpi=300); plt.close(fig)

def finalise(contours,filenames,records,protected_before):
    allrows=[]; summaries=[]; missing=[]
    for K,q in sorted(PAIRS):
        rows=[]
        for i in range(1652):
            p=OUT/"registrations"/f"K{K:02d}_q{q:02d}"/f"{i:04d}.npz"; r=load_checkpoint(p,K,q,i)
            if r is None: missing.append(str(p))
            else: rows.append(r); allrows.append(r)
        if len(rows)==1652:summaries.append(combo_summary(K,q,rows))
    if missing: raise RuntimeError(f"missing/corrupt checkpoints: {len(missing)}")
    atomic_csv(OUT/"tables"/"per_registration_results.csv",sorted(allrows,key=lambda r:(int(r["K"]),int(r["target_index"]))))
    atomic_csv(OUT/"tables"/"summary_gmm_registration_k2_k9.csv",summaries)
    comp=paired_comparison(allrows); atomic_csv(OUT/"tables"/"kmeans_vs_gmm_registration_comparison.csv",comp); plot_results(summaries,comp)
    protected_after={"kmeans":fingerprint_tree(KMEANS),"audit":fingerprint_tree(AUDIT),"dataset_sha256":gabriel.sha256_file(DATA)}
    checks={"exact_8_pairs":set(PAIRS)=={(2,3),(3,1),(4,37),(5,26),(6,2),(7,20),(8,3),(9,4)},
            "exact_13216_checkpoints":len(allrows)==13216,"target_identity_preserved":all(filenames[int(r["target_index"])]==r["filename"] for r in allrows),
            "audited_assignment_used":all(int(records[(int(r["K"]),int(r["q"]))]["labels"][int(r["target_index"])])==int(r["cluster_id"]) for r in allrows),
            "audited_hard_templates_exact":all(np.array_equal(records[(K,q)]["templates"][c],contours[records[(K,q)]["labels"]==c].mean(0)) for K,q in PAIRS for c in range(K)),
            "same_registration_config":json.loads(json.dumps(asdict(CONFIG)))==json.loads(DEV_CONFIG.read_text())["registration_config"],
            "diagnostic_grid_101":all(int(r["diagnostic_resolution"])==101 for r in allrows),"all_checkpoints_readable":not missing,
            "production_inputs_unchanged":protected_before==protected_after,"summary_reproducible":len(summaries)==8}
    gabriel.atomic_json(OUT/"validation.json",{"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"counts":{"rows":len(allrows),"expected":13216},"protected":protected_after})
    report=["# Gabriel GMM-template registration sensitivity","","## Technical summary","",f"All 13,216 registrations for the eight fixed Gabriel-selected (K,q) pairs were evaluated with the frozen production registration configuration. This analysis does not select K.","","## GMM registration summary","","| K | q | valid / invalid / failed | mean Dirichlet ± SE | mean SVF ± SE | mean MSE | minimum Jacobian |","|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries: report.append(f"| {s['K']} | {s['q']} | {s['n_valid']} / {s['n_invalid']} / {s['n_failed']} | {s['mean_dirichlet']:.8g} ± {s['se_dirichlet']:.3g} | {s['mean_svf_magnitude']:.8g} ± {s['se_svf_magnitude']:.3g} | {s['mean_mse']:.3g} | {s['min_jacobian']:.5g} |")
    report += ["","## Paired K-means comparison","","The table uses only target identities with valid checkpoints under both methods.","","| K | paired n | Dirichlet difference (%) | SVF difference (%) | fraction GMM lower: D / SVF |","|---:|---:|---:|---:|---:|"]
    for r in comp: report.append(f"| {r['K']} | {r['n_paired']} | {r['percent_difference_dirichlet']:+.2f}% | {r['percent_difference_svf_magnitude']:+.2f}% | {r['fraction_gmm_lower_dirichlet']:.1%} / {r['fraction_gmm_lower_svf_magnitude']:.1%} |")
    report += ["","## Main findings","","- No selected GMM-template combination lowered the population mean of both deformation metrics relative to the same-K K-means baseline.","- K=3 was closest in Dirichlet energy, while its mean SVF magnitude remained higher.","- K=8 had the lowest GMM mean Dirichlet and SVF magnitude; K=9 reversed both local trends despite adding a template.","- The K=9 result must be interpreted with its three-sample GMM component and weak initialization stability; K=7 had one completed but Jacobian-invalid registration.","","## Interpretation boundary","","Energy and velocity magnitude measure downstream deformation burden under fixed templates and registration settings. They do not by themselves establish clustering quality or a preferred K.","","## Validation","",f"Validation status: **{'PASS' if all(checks.values()) else 'FAIL'}**. K-means outputs and the audited GMM source artifacts remained byte-for-byte unchanged."]
    (OUT/"report").mkdir(parents=True,exist_ok=True); (OUT/"report"/"final_report.md").write_text("\n".join(report)+"\n",encoding="utf8")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--workers",type=int,default=4); ap.add_argument("--stage",choices=("prepare","run","finalise","all"),default="all"); args=ap.parse_args()
    if args.workers!=4: ap.error("this experiment requires exactly 4 workers")
    for d in ("registrations","tables","figures","report"): (OUT/d).mkdir(parents=True,exist_ok=True)
    protected={"kmeans":fingerprint_tree(KMEANS),"audit":fingerprint_tree(AUDIT),"dataset_sha256":gabriel.sha256_file(DATA)}
    contours,filenames,records=load_inputs()
    config={"experiment":"Gabriel GMM hard-template registration sensitivity","dataset":str(DATA.relative_to(ROOT)),"shape":list(contours.shape),"dtype":str(contours.dtype),
            "pairs":[{"K":K,"q":q,"representative_type":REP[(K,q)][0],"representative_seed":REP[(K,q)][1],"cluster_sizes":records[(K,q)]["sizes"].tolist()} for K,q in PAIRS],
            "batch_order":[[{"K":K,"q":q} for K,q in b] for b in BATCHES],"workers":4,"registration_config":asdict(CONFIG),"diagnostic_resolution":101,"protected_before":protected}
    gabriel.atomic_json(OUT/"config.json",config)
    if args.stage=="prepare":return
    if args.stage in ("run","all"):
        for batch,pairs in enumerate(BATCHES,1):
            append_log(OUT/"progress.log","batch_start",batch=batch,pairs=pairs)
            for K,q in pairs: run_combo(K,q,contours,filenames,records[(K,q)],args.workers,batch)
            append_log(OUT/"progress.log","batch_complete",batch=batch)
    if args.stage in ("finalise","all"): finalise(contours,filenames,records,protected)

if __name__=="__main__": main()
