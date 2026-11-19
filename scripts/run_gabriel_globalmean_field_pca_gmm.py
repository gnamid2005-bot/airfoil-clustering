"""Isolated Gabriel global-mean registration -> coefficient PCA -> GMM experiment.

The PCA input produced here is exclusively the C-order flattening of Gabriel's
30 x 30 x 2 cubic B-spline coefficient tensors.  Raw contour coordinates are
used only as registration targets and to construct the single global mean.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-gabriel-globalmean-field")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import gabriel_pinned_registration as gabriel

DATA = ROOT / "data/airfoils/processed_gabriel/airfoils_pinned_arclength_n200.npz"
SAVED_MEAN = ROOT / "data/airfoils/processed_gabriel/global_mean_gabriel.npy"
DEV_CONFIG = ROOT / "outputs/gabriel_pinned_registration_dev/config.json"
OLD = ROOT / "outputs/gabriel_pinned_gmm_vs_kmeans"
OUT = ROOT / "outputs/gabriel_globalmean_field_pca_gmm"
CONFIG = gabriel.RegistrationConfig()
SEED, N_INIT, REG_COVAR, MAX_ITER, TOL = 20260813, 5, 1e-7, 500, 1e-5
QS, KS, DIAGNOSTIC_RESOLUTION = tuple(range(1, 101)), tuple(range(1, 21)), 101
DATA_SHA = "f357610c338b7a820c1f09522052d29951e9e62e41d3faffc61de19621995e33"


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(gabriel.jsonable(obj), indent=2, sort_keys=True), encoding="utf8")
    os.replace(tmp, path)


def atomic_csv(path: Path, rows, fields=None) -> None:
    rows = list(rows); path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)


def sha_bytes(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def protected_fingerprint() -> dict:
    paths = [
        ROOT / "outputs/gabriel_k2_k8_energy_magnitude",
        ROOT / "outputs/gabriel_modal_kmeans_registration",
        ROOT / "outputs/gabriel_gmm_template_registration",
        ROOT / "outputs/gabriel_pinned_gmm_vs_kmeans",
        ROOT / "outputs/airfoil_shape_pca_gmm_selection",
    ]
    rows = []
    for base in paths:
        for p in sorted(base.rglob("*")) if base.exists() else []:
            if p.is_file():
                s = p.stat(); rows.append((str(p.relative_to(ROOT)), s.st_size, s.st_mtime_ns))
    return {"algorithm": "sha256(path,size,mtime_ns)", "file_count": len(rows),
            "digest": hashlib.sha256(json.dumps(rows).encode()).hexdigest()}


def load_inputs():
    if gabriel.sha256_file(DATA) != DATA_SHA:
        raise RuntimeError("STOP: Gabriel pinned dataset SHA-256 mismatch")
    with np.load(DATA, allow_pickle=False) as z:
        contours = z["contours"].copy(); filenames = z["filenames"].astype(str)
        keys = list(z.files)
    if contours.shape != (1652, 200, 2) or contours.dtype != np.float64 or not np.isfinite(contours).all():
        raise RuntimeError(f"STOP: unexpected Gabriel data {contours.shape} {contours.dtype}")
    if not np.array_equal(np.arange(len(contours)), np.arange(len(filenames))):
        raise RuntimeError("STOP: sample count/order failure")
    mean = contours.mean(axis=0)
    if not np.array_equal(mean, np.load(SAVED_MEAN, allow_pickle=False)):
        raise RuntimeError("STOP: saved global mean is not exact all-sample pointwise mean")
    geom = gabriel.validate_pinned_contour(mean)
    valid_mean = bool(np.isfinite(mean).all() and gabriel.signed_polygon_area(mean) > 1e-8
                      and gabriel.self_intersection_count(mean) == 0
                      and geom["index0_is_te"] and geom["index100_is_le"]
                      and geom["upper_is_above_lower"])
    if not valid_mean: raise RuntimeError(f"STOP: invalid global mean: {geom}")
    frozen = json.loads(DEV_CONFIG.read_text())["registration_config"]
    if json.loads(json.dumps(asdict(CONFIG))) != frozen:
        raise RuntimeError("STOP: registration defaults differ from validated Gabriel production config")
    return contours, filenames, mean, keys, geom


def prepare() -> None:
    contours, filenames, mean, keys, geom = load_inputs()
    for d in ("global_mean", "registrations", "smoke", "coefficients", "splits", "pca", "gmm", "tables", "figures", "report", "logs"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    gabriel.atomic_npz(OUT / "global_mean/global_mean.npz", coordinates=mean,
        population_size=np.asarray(len(contours)), filenames=filenames,
        dataset_sha256=np.asarray(DATA_SHA), coordinates_sha256=np.asarray(sha_bytes(mean)))
    audit = {
        "dataset": str(DATA.relative_to(ROOT)), "dataset_sha256": DATA_SHA,
        "npz_keys": keys, "contours_shape": list(contours.shape), "dtype": str(contours.dtype),
        "finite": True, "sample_order": "exact stored NPZ order", "filenames_sha256": sha_bytes(filenames),
        "global_mean_method": "pointwise arithmetic mean over all 1652 Gabriel pinned contours",
        "global_mean_shape": list(mean.shape), "global_mean_coordinates_sha256": sha_bytes(mean),
        "global_mean_geometry": geom, "exactly_one_global_mean": True,
        "registration_function": "src/gabriel_pinned_registration.py::optimise_registration",
        "registration_direction": "source=single global mean; target=individual airfoil; flow(source)->target",
        "registration_config": asdict(CONFIG), "coefficient_tensor_shape": [30,30,2],
        "flattening": "numpy C order: control_x major, then control_y, then component [x,y]",
        "pca_input": "REGISTRATION-DERIVED B-SPLINE COEFFICIENTS ONLY (never contour coordinates)",
        "gmm": {"covariance_type":"full", "n_init":N_INIT, "reg_covar":REG_COVAR,
                "max_iter":MAX_ITER, "tol":TOL, "init_params":"kmeans", "random_state":SEED,
                "ICL":"BIC + 2 * positive posterior classification entropy"},
        "protected_outputs_before": protected_fingerprint(),
    }
    atomic_json(OUT / "audit.json", audit)


def geometry(deformed: np.ndarray) -> dict:
    finite=bool(np.isfinite(deformed).all()); area=gabriel.signed_polygon_area(deformed)
    intersections=gabriel.self_intersection_count(deformed)
    return {"finite":finite,"signed_area":area,"self_intersection_count":intersections,
            "valid":bool(finite and area>1e-8 and intersections==0)}


def run_one(payload):
    index, filename, source, target, path_text, dataset_sha, mean_sha = payload
    import torch
    torch.set_num_threads(1)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    start=time.perf_counter(); base={"target_index":index,"filename":filename,
        "dataset_sha256":dataset_sha,"global_mean_coordinates_sha256":mean_sha,
        "source_template_id":"SINGLE_GLOBAL_MEAN_ALL_1652"}
    try:
        result=gabriel.optimise_registration(source,target,CONFIG)
        metric=gabriel.correspondence_metrics(result["deformed"],target)
        grid=gabriel.diagnostic_grid(result["coefficients"],result["control_x"],result["control_y"],CONFIG.rk_steps,DIAGNOSTIC_RESOLUTION)
        geo=geometry(result["deformed"]); coeff=result["coefficients"]
        valid=bool(result["status"]=="completed" and result["finite"] and geo["valid"]
                   and grid["jacobian_minimum"]>0 and grid["jacobian_nonpositive_fraction"]==0
                   and coeff.shape==(30,30,2) and np.isfinite(coeff).all())
        row={**base,"status":result["status"],"failure_reason":result["failure_reason"],
             "valid_registration":valid,"geometry_valid":geo["valid"],"runtime_seconds":time.perf_counter()-start,
             "optimisation_runtime_seconds":result["runtime_seconds"],"closure_evaluations":result["closure_evaluations"],
             "coefficient_tensor_shape":[30,30,2],"flattened_coefficient_length":1800,
             "flatten_order":"C(control_x,control_y,component[x,y])","final_pointwise_mse":metric["pointwise_mse"],
             "final_pointwise_rmse":metric["pointwise_rmse"],"minimum_jacobian_determinant":grid["jacobian_minimum"],
             "jacobian_nonpositive_fraction":grid["jacobian_nonpositive_fraction"],
             "coefficient_smoothness":result["final_coefficient_smoothness"],"final_lambda":result["final_lambda"]}
        gabriel.atomic_npz(Path(path_text), metrics_json=np.asarray(json.dumps(gabriel.jsonable(row),sort_keys=True)),
            coefficients=coeff, control_x=result["control_x"], control_y=result["control_y"], deformed=result["deformed"])
    except Exception as exc:
        row={**base,"status":"failed","failure_reason":f"{type(exc).__name__}:{exc}","valid_registration":False,
             "geometry_valid":False,"runtime_seconds":time.perf_counter()-start,
             "coefficient_tensor_shape":[],"flattened_coefficient_length":0}
        gabriel.atomic_npz(Path(path_text),metrics_json=np.asarray(json.dumps(row,sort_keys=True)))
    return row


def load_checkpoint(path: Path, index: int, filename: str, mean_sha: str):
    try:
        with np.load(path, allow_pickle=False) as z:
            row=json.loads(str(np.asarray(z["metrics_json"]).item()))
            if (int(row["target_index"])!=index or row["filename"]!=filename or
                row["dataset_sha256"]!=DATA_SHA or row["global_mean_coordinates_sha256"]!=mean_sha): return None
            if row.get("valid_registration") is True:
                c=z["coefficients"]
                if c.shape!=(30,30,2) or not np.isfinite(c).all(): return None
            return row
    except Exception: return None


def register(indices, workers: int, smoke: bool=False):
    contours, filenames, mean, _, _=load_inputs(); mean_sha=sha_bytes(mean)
    base=OUT/("smoke" if smoke else "registrations"); base.mkdir(parents=True,exist_ok=True)
    rows=[]; todo=[]
    for i in indices:
        p=base/f"{i:04d}.npz"; old=load_checkpoint(p,int(i),str(filenames[i]),mean_sha)
        if old is not None: rows.append(old)
        else: todo.append((int(i),str(filenames[i]),mean,contours[i],str(p),DATA_SHA,mean_sha))
    start=time.perf_counter(); new=0
    if workers == 1:
        results = (run_one(p) for p in todo)
        for row in results:
            rows.append(row); new+=1
            if new==1 or new%25==0 or new==len(todo):
                elapsed=time.perf_counter()-start; rate=new/elapsed if elapsed else 0
                print(f"{len(rows)}/{len(indices)} complete; {rate:.3f} registrations/s",flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(run_one,p) for p in todo]
            for future in as_completed(futures):
                rows.append(future.result()); new+=1
                if new==1 or new%25==0 or new==len(todo):
                    elapsed=time.perf_counter()-start; rate=new/elapsed if elapsed else 0
                    print(f"{len(rows)}/{len(indices)} complete; {rate:.3f} registrations/s",flush=True)
    rows.sort(key=lambda r:int(r["target_index"]))
    atomic_csv(OUT/("logs/smoke_results.csv" if smoke else "tables/per_registration.csv"),rows)
    elapsed=time.perf_counter()-start
    summary={"requested":len(indices),"reused":len(indices)-len(todo),"new":new,"wall_seconds":elapsed,
             "mean_case_seconds":float(np.mean([r["runtime_seconds"] for r in rows])),
             "valid":sum(bool(r.get("valid_registration")) for r in rows),
             "failed":sum(r.get("status")!="completed" for r in rows),
             "projected_full_serial_hours":float(np.mean([r["runtime_seconds"] for r in rows]))*1652/3600,
             "projected_full_wall_hours_at_workers":elapsed/max(new,1)*1652/3600,"workers":workers}
    atomic_json(OUT/("logs/smoke_summary.json" if smoke else "logs/registration_summary.json"),summary)
    smoke_extractable = all(r.get("status")=="completed" and r.get("flattened_coefficient_length")==1800 for r in rows)
    summary["smoke_coefficient_extraction_passed"] = smoke_extractable
    atomic_json(OUT/("logs/smoke_summary.json" if smoke else "logs/registration_summary.json"),summary)
    if smoke and not smoke_extractable: raise RuntimeError("STOP: smoke coefficient extraction failed")
    return rows


def deterministic_split(n):
    order=np.random.default_rng(SEED).permutation(n); n_train=int(round(.8*n))
    return np.sort(order[:n_train]),np.sort(order[n_train:])


def build_matrix():
    contours, filenames, mean, _, _=load_inputs(); mean_sha=sha_bytes(mean); rows=[]; coeff=[]; valid_idx=[]
    control_x=control_y=None
    for i,name in enumerate(filenames):
        p=OUT/"registrations"/f"{i:04d}.npz"; row=load_checkpoint(p,i,str(name),mean_sha)
        if row is None: raise RuntimeError(f"STOP: missing/corrupt checkpoint {i}")
        rows.append(row)
        if row.get("valid_registration") is True:
            with np.load(p,allow_pickle=False) as z:
                c=z["coefficients"].copy(); cx=z["control_x"].copy(); cy=z["control_y"].copy()
            if control_x is None: control_x,control_y=cx,cy
            if not np.array_equal(cx,control_x) or not np.array_equal(cy,control_y): raise RuntimeError("STOP: control-grid mismatch")
            coeff.append(c.reshape(-1,order="C")); valid_idx.append(i)
    X=np.asarray(coeff,dtype=np.float64); valid_idx=np.asarray(valid_idx,dtype=np.int64)
    if X.ndim!=2 or X.shape[1]!=1800 or not np.isfinite(X).all(): raise RuntimeError(f"STOP: invalid coefficient matrix {X.shape}")
    gabriel.atomic_npz(OUT/"coefficients/registration_coefficients.npz",X_field=X,target_indices=valid_idx,
        filenames=filenames[valid_idx],coefficient_tensor_shape=np.asarray([30,30,2]),flatten_order=np.asarray("C"),
        component_order=np.asarray("last axis [x,y]"),control_x=control_x,control_y=control_y,
        dataset_sha256=np.asarray(DATA_SHA),global_mean_coordinates_sha256=np.asarray(mean_sha),
        pca_input_definition=np.asarray("REGISTRATION-DERIVED B-SPLINE COEFFICIENTS; NO RAW COORDINATES"))
    return X,valid_idx,filenames[valid_idx],rows


def fit_cell(train, heldout, q, k):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model=GaussianMixture(n_components=k,covariance_type="full",n_init=N_INIT,reg_covar=REG_COVAR,
            max_iter=MAX_ITER,tol=TOL,init_params="kmeans",random_state=SEED).fit(train[:,:q])
    resp=model.predict_proba(train[:,:q])
    entropy=float(-np.sum(resp*np.log(np.clip(resp, np.finfo(float).tiny, 1.0))))
    bic=float(model.bic(train[:,:q])); ll=float(model.score(heldout[:,:q]))
    return {"q":q,"K":k,"BIC":bic,"ICL":bic+2*entropy,"posterior_classification_entropy":entropy,
            "heldout_log_likelihood_per_sample":ll,"heldout_log_likelihood_total":ll*len(heldout),
            "converged":bool(model.converged_),"n_iter":int(model.n_iter_),"warnings":" | ".join(str(x.message) for x in caught)}


def plateaus(values, criterion):
    out=[]; start=1; current=values[0]
    for q,value in enumerate(values[1:],2):
        if value!=current: out.append({"q_start":start,"q_end":q-1,"preferred_K":current,"criterion":criterion});start=q;current=value
    out.append({"q_start":start,"q_end":len(values),"preferred_K":current,"criterion":criterion});return out


def analyse(jobs: int):
    X,valid_idx,filenames,reg_rows=build_matrix(); n=len(X)
    if n<101: raise RuntimeError("STOP: fewer than 101 valid registrations")
    full_train,full_held=deterministic_split(1652)
    pos={int(idx):j for j,idx in enumerate(valid_idx)}
    train_global=np.asarray([i for i in full_train if int(i) in pos]); held_global=np.asarray([i for i in full_held if int(i) in pos])
    train=np.asarray([pos[int(i)] for i in train_global]); held=np.asarray([pos[int(i)] for i in held_global])
    np.save(OUT/"splits/train_indices_original_1652.npy",full_train);np.save(OUT/"splits/heldout_indices_original_1652.npy",full_held)
    np.save(OUT/"splits/train_indices_valid_used.npy",train_global);np.save(OUT/"splits/heldout_indices_valid_used.npy",held_global)
    # Compatibility names are the exact indices actually used by PCA/GMM.
    np.save(OUT/"splits/train_indices.npy",train_global);np.save(OUT/"splits/heldout_indices.npy",held_global)
    pca=PCA(n_components=100,svd_solver="full").fit(X[train])
    train_scores=pca.transform(X[train]);held_scores=pca.transform(X[held])
    gabriel.atomic_npz(OUT/"pca/pca_q100.npz",mean=pca.mean_,components=pca.components_,
        explained_variance=pca.explained_variance_,explained_variance_ratio=pca.explained_variance_ratio_,
        singular_values=pca.singular_values_,train_scores=train_scores,heldout_scores=held_scores,
        train_target_indices=train_global,heldout_target_indices=held_global,pca_fit_scope=np.asarray("training only"),
        pca_input=np.asarray("registration-derived 30x30x2 B-spline coefficients"))
    pca_rows=[]
    for q in QS:
        tr_rec=train_scores[:,:q]@pca.components_[:q]+pca.mean_;ho_rec=held_scores[:,:q]@pca.components_[:q]+pca.mean_
        pca_rows.append({"q":q,"cumulative_explained_variance":float(pca.explained_variance_ratio_[:q].sum()),
            "train_reconstruction_mse":float(np.mean((X[train]-tr_rec)**2)),
            "heldout_reconstruction_mse":float(np.mean((X[held]-ho_rec)**2))})
    atomic_csv(OUT/"tables/pca_diagnostics.csv",pca_rows)
    tasks=[(q,k) for q in QS for k in KS]
    grid=Parallel(n_jobs=jobs,verbose=10)(delayed(fit_cell)(train_scores,held_scores,q,k) for q,k in tasks)
    atomic_csv(OUT/"gmm/full_model_selection_grid.csv",grid)
    preferred=[]
    for q in QS:
        rr=[r for r in grid if r["q"]==q]; p=pca_rows[q-1]
        b=min(rr,key=lambda r:r["BIC"]);i=min(rr,key=lambda r:r["ICL"]);h=max(rr,key=lambda r:r["heldout_log_likelihood_per_sample"])
        preferred.append({"q":q,"preferred_k_bic":b["K"],"preferred_k_icl":i["K"],"preferred_k_heldout_ll":h["K"],
            "bic_min":b["BIC"],"icl_min":i["ICL"],"heldout_ll_max":h["heldout_log_likelihood_per_sample"],
            "cumulative_explained_variance":p["cumulative_explained_variance"]})
    atomic_csv(OUT/"gmm/preferred_k_by_q.csv",preferred)
    plats=plateaus([r["preferred_k_bic"] for r in preferred],"BIC")+plateaus([r["preferred_k_icl"] for r in preferred],"ICL")+plateaus([r["preferred_k_heldout_ll"] for r in preferred],"Held-out LL")
    atomic_csv(OUT/"tables/preferred_k_plateaus.csv",plats)
    left=[]
    for k in range(2,10):
        b=next((r["q"] for r in preferred if r["preferred_k_bic"]==k),None);i=next((r["q"] for r in preferred if r["preferred_k_icl"]==k),None)
        left.append({"K":k,"leftmost_q_bic":b,"leftmost_q_icl":i})
    atomic_csv(OUT/"tables/leftmost_q_k2_k9.csv",left)
    agreement=[{"q":r["q"],"bic_k":r["preferred_k_bic"],"icl_k":r["preferred_k_icl"],"agree":r["preferred_k_bic"]==r["preferred_k_icl"]} for r in preferred]
    atomic_csv(OUT/"tables/bic_icl_agreement.csv",agreement)
    plot_results(preferred,pca_rows)
    checks={"dataset_is_Gabriel_pinned":gabriel.sha256_file(DATA)==DATA_SHA,"exactly_one_global_mean":True,
        "all_coefficients_same_global_mean":all(r.get("global_mean_coordinates_sha256")==sha_bytes(np.load(SAVED_MEAN)) for r in reg_rows),
        "coefficient_matrix_is_N_by_1800":X.shape==(len(valid_idx),1800),"coefficient_matrix_finite":bool(np.isfinite(X).all()),
        "PCA_fit_training_only":True,"PCA_input_is_registration_coefficients":True,"raw_shape_coordinates_absent_from_X_field":X.shape[1]!=400,
        "GMM_fit_training_scores_only":True,"all_model_values_finite":all(math.isfinite(r[x]) for r in grid for x in ("BIC","ICL","heldout_log_likelihood_per_sample")),
        "all_2000_cells":len(grid)==2000,"all_GMM_cells_converged":all(r["converged"] for r in grid),
        "sample_order_preserved":bool(np.all(np.diff(valid_idx)>0)),"protected_outputs_unchanged":json.loads((OUT/"audit.json").read_text())["protected_outputs_before"]==protected_fingerprint()}
    validation={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"N_valid":len(valid_idx),"N_invalid":1652-len(valid_idx),
        "train_count":len(train),"heldout_count":len(held),"bic_icl_agreement_rate":float(np.mean([r["agree"] for r in agreement]))}
    atomic_json(OUT/"validation.json",validation)
    make_report(preferred,plats,left,validation,X.shape,reg_rows)


def plot_results(preferred,pca_rows):
    curves=[("preferred_k_bic","BIC","#2563eb","o"),("preferred_k_icl","ICL","#d97706","s"),("preferred_k_heldout_ll","Held-out LL","#6b7280","^")]
    def save(fig,name):
        fig.tight_layout();fig.savefig(OUT/"figures"/(name+".png"),dpi=300,bbox_inches="tight");fig.savefig(OUT/"figures"/(name+".pdf"),bbox_inches="tight");plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5.7))
    for field,label,color,marker in curves: ax.plot(QS,[r[field] for r in preferred],label=label,color=color,marker=marker,ms=3,lw=1.4)
    ax.set(xlabel="PCA dimension q",ylabel="Preferred GMM component count K",title="Preferred component count across registration-field PCA dimensions (q=1–100)")
    ax.set_xticks(range(0,101,10));ax.set_yticks(range(1,21));ax.grid(alpha=.22);ax.legend();save(fig,"preferred_k_field_q1_q100")
    fig,ax=plt.subplots(figsize=(10,5));ax.plot(QS,[r["cumulative_explained_variance"] for r in pca_rows],color="#0f766e")
    ax.set(xlabel="PCA dimension q",ylabel="Cumulative explained variance",title="Registration-coefficient PCA explained variance");ax.grid(alpha=.22);save(fig,"cumulative_explained_variance")
    for field,label,color,_ in curves:
        fig,ax=plt.subplots(figsize=(10,4.8));ax.plot(QS,[r[field] for r in preferred],color=color,marker="o",ms=3)
        ax.set(xlabel="PCA dimension q",ylabel="Preferred K",title=f"{label} preferred K vs q");ax.set_yticks(range(1,21));ax.grid(alpha=.22);save(fig,f"{field}_vs_q")


def make_report(preferred,plats,left,validation,shape,reg_rows):
    seq=lambda f:",".join(str(r[f]) for r in preferred)
    lines=["# Gabriel single-global-mean field PCA/GMM report","","## Result","",
        f"Validation: **{validation['status']}**. Valid registrations: {validation['N_valid']}; invalid/failed: {validation['N_invalid']}.","",
        "The PCA in this experiment was performed on registration-derived vector-field/B-spline coefficients relative to a single common mean, not on raw airfoil shape coordinates.","",
        "## Exact method","",f"- Dataset: `{DATA.relative_to(ROOT)}`; SHA-256 `{DATA_SHA}`; N=1652; 200 points per contour.",
        "- Global mean: exact pointwise arithmetic mean of all 1652 eligible Gabriel pinned contours.",
        "- Registration: `gabriel_pinned_registration.optimise_registration`; direction global mean → individual target.",
        f"- Registration configuration: `{json.dumps(asdict(CONFIG),sort_keys=True)}`.",
        f"- Coefficients: `(30,30,2)`, C-order flattened to D={shape[1]}; axes are control-x, control-y, component [x,y].",
        f"- Split: seed {SEED}; training {validation['train_count']}; held-out {validation['heldout_count']}; PCA fitted on training only.",
        "- PCA: sklearn PCA, n_components=100, svd_solver=full, centered, no whitening.",
        f"- GMM: full covariance, n_init={N_INIT}, reg_covar={REG_COVAR}, max_iter={MAX_ITER}, tol={TOL}, init_params=kmeans, random_state={SEED}.","",
        "## Preferred-K sequences (q=1..100)","",f"- BIC: `{seq('preferred_k_bic')}`",f"- ICL: `{seq('preferred_k_icl')}`",f"- Held-out LL: `{seq('preferred_k_heldout_ll')}`","",
        f"BIC/ICL agreement rate: {validation['bic_icl_agreement_rate']:.1%}.","",
        "Plateaus and leftmost q for K=2..9 are in the tables listed below. No downstream template registration was launched.","",
        "## Outputs","","- `coefficients/registration_coefficients.npz`","- `gmm/full_model_selection_grid.csv`","- `gmm/preferred_k_by_q.csv`","- `tables/pca_diagnostics.csv`","- `tables/preferred_k_plateaus.csv`","- `tables/leftmost_q_k2_k9.csv`","- `figures/preferred_k_field_q1_q100.png` and `.pdf`","- `validation.json`","",
        "## Qualitative comparison","","The previous pinned-shape figure is methodologically comparable only at the PCA/GMM selection layer; it used flattened coordinates. The new curves should be interpreted as deformation-field behaviour, without claiming superiority from any difference."]
    (OUT/"report/final_report.md").write_text("\n".join(lines)+"\n",encoding="utf8")


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--stage",choices=("prepare","smoke","register","analyse","all"),default="all")
    ap.add_argument("--workers",type=int,default=4);ap.add_argument("--jobs",type=int,default=4);args=ap.parse_args()
    if args.stage in ("prepare","all"): prepare()
    if args.stage in ("smoke","all"):
        c,_,m,_,_=load_inputs();dist=np.linalg.norm((c-m).reshape(len(c),-1),axis=1)
        register([int(np.argsort(dist)[len(c)//2]),int(np.argmax(dist)),0],min(args.workers,3),True)
    if args.stage in ("register","all"): register(range(1652),args.workers,False)
    if args.stage in ("analyse","all"): analyse(args.jobs)


if __name__=="__main__": main()
