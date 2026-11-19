"""Assigned-template q=20/K=1..8 registration data generation.

This is deliberately a data-generation driver.  It contains no model ranking,
selection, hypothesis testing, or combined score.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import logsumexp

import run_airfoil_registration_improvement_stage2 as s2
import run_airfoil_registration_improvement_stage3 as s3
import run_airfoil_template_number_screening as s5
from airfoil_registration import validate_airfoil_shape

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
STAGE5 = ROOT / "outputs" / "airfoil_template_number_stage5_screening"
OUT = ROOT / "outputs" / "airfoil_q20_k1_k8_velocity_properties"
DATA = ROOT / "data" / "airfoils" / "processed" / "airfoils_n100_seed0.npz"
Q = 20
KS = tuple(range(1, 9))
DTYPE = torch.float64
REG_GRID = 51
HF_THRESHOLD = 0.5
ENTROPY_BINS = 32
SCHEMA = [
    "K", "q", "target_index", "target_filename", "GMM_component", "source_template_id", "source_cluster_size",
    "converged", "failure", "geometric_valid", "runtime", "iterations", "initial_distance", "total_objective",
    "raw_regularisation", "weighted_regularisation", "jacobian_barrier_raw", "jacobian_barrier_weighted",
    "velocity_magnitude_mean", "velocity_magnitude_rms", "velocity_magnitude_max",
    "dirichlet_energy_raw", "dirichlet_energy_weighted",
    "correspondence_mse", "correspondence_rmse", "correspondence_max_error",
    "lddmm_geodesic_energy", "lddmm_geodesic_energy_status", "velocity_magnitude_variance",
    "maximum_displacement", "mean_displacement", "rms_displacement",
    "jacobian_min", "jacobian_mean", "jacobian_median", "jacobian_p01", "jacobian_p05", "jacobian_p95", "jacobian_max",
    "jacobian_nonpositive_fraction", "jacobian_variance",
    "curl_rms", "curl_mean_abs", "curl_max_abs", "divergence_rms", "divergence_mean_abs", "divergence_max_abs",
    "bending_energy", "flow_path_length_mean", "flow_path_length_rms", "flow_path_length_max", "kinetic_energy",
    "rkhs_norm", "rkhs_norm_status", "spectral_total_energy", "spectral_high_frequency_fraction", "field_entropy",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or sorted({k for row in rows for k in row})
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    tmp.replace(path)


def load_inputs() -> tuple[dict, dict, np.ndarray, np.ndarray]:
    data = s5.load_dataset()
    cfg = s5.frozen_config(data["shapes"])
    population = np.load(STAGE5 / "velocity_field_population_indices.npy")
    matrix = np.load(STAGE5 / "velocity_field_matrix.npy", mmap_mode="r")
    if population.shape != (1651,) or matrix.shape != (1651, 10368):
        raise RuntimeError("Stage 5 safe population/matrix is not exactly 1651 x 10368")
    if len(np.unique(population)) != 1651 or set(population.tolist()) != set(range(1654)) - {3, 206, 408}:
        raise RuntimeError("Stage 5 population indices differ from the audited safe population")
    return data, cfg, population.astype(np.int64), matrix


def pca_scores(matrix: np.ndarray) -> np.ndarray:
    with np.load(STAGE5 / "pca" / "pca_q20.npz") as p:
        scores = (np.asarray(matrix) - p["mean"]) @ p["components"].T
        positions = np.concatenate([p["train_matrix_positions"], p["heldout_matrix_positions"]])
        saved = np.vstack([p["train_scores"], p["heldout_scores"]])
        if not np.allclose(scores[positions], saved, rtol=1e-10, atol=1e-12):
            raise RuntimeError("Reconstructed q=20 PCA scores differ from saved scores")
    return scores


def gmm_probabilities(scores: np.ndarray, k: int) -> tuple[np.ndarray, dict]:
    path = STAGE5 / "gmm" / "models" / f"q20_k{k:02d}_full.npz"
    with np.load(path) as z:
        weights, means, precisions = z["weights"], z["means"], z["precisions_cholesky"]
        covariances = z["covariances"]
    logp = np.empty((len(scores), k), dtype=np.float64)
    d = scores.shape[1]
    for c in range(k):
        y = (scores - means[c]) @ precisions[c]
        log_det = np.log(np.diag(precisions[c])).sum()
        logp[:, c] = -.5 * (d * np.log(2 * np.pi) + np.sum(y * y, axis=1)) + log_det + np.log(weights[c])
    prob = np.exp(logp - logsumexp(logp, axis=1, keepdims=True))
    primary = next(r for r in csv.DictReader((STAGE5 / "gmm" / "primary_grid.csv").open()) if int(r["q"]) == 20 and int(r["K"]) == k)
    meta = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "seed": int(primary["seed"]),
            "covariance_type": primary["covariance_type"], "components": k, "converged": primary["converged"] == "True",
            "weights": weights.tolist(), "training_hard_component_sizes": json.loads(primary["hard_component_sizes"]),
            "covariance_shapes": list(covariances.shape)}
    return prob, meta


def make_templates(data: dict, population: np.ndarray, scores: np.ndarray) -> tuple[dict, list[dict], list[dict], dict]:
    templates, summary, assignments, model_meta = {}, [], [], {}
    shapes = data["shapes"]
    for k in KS:
        if k == 1:
            prob = np.ones((len(population), 1)); meta = {"path": "not_applicable_global_mean_baseline", "seed": None,
                "covariance_type": "not_applicable", "components": 1, "converged": True, "weights": [1.0],
                "training_hard_component_sizes": [1321]}
        else:
            prob, meta = gmm_probabilities(scores, k)
        model_meta[str(k)] = meta
        labels = prob.argmax(axis=1)
        sizes = np.bincount(labels, minlength=k)
        for pos, idx in enumerate(population):
            ent = -float(np.sum(prob[pos] * np.log(np.clip(prob[pos], 1e-300, 1.0))))
            assignments.append({"target_index": int(idx), "filename": str(data["filenames"][idx]), "K": k,
                                "MAP_component": int(labels[pos]), "maximum_posterior_probability": float(prob[pos].max()),
                                "posterior_entropy": ent})
        for c in range(k):
            members = population[labels == c]
            template = shapes[members].mean(axis=0) if k > 1 else data["mean"]
            validity = validate_airfoil_shape(template)
            if template.shape != (200, 2) or not validity["valid"]:
                raise RuntimeError(f"Invalid coordinate-space template K={k} component={c}: {validity}")
            tid = f"k{k:02d}_component{c:02d}"
            templates[(k, c)] = {"coordinates": template, "members": members, "id": tid, "size": int(len(members))}
            d = OUT / "templates" / f"K{k:02d}"; d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / f"{tid}.npz", K=k, component=c, member_indices=members,
                member_filenames=data["filenames"][members], coordinates=template, validity_json=json.dumps(validity, sort_keys=True))
            summary.append({"K": k, "component": c, "cluster_size": int(len(members)), "GMM_weight": float(meta["weights"][c]),
                            "template_validity": True, "template_id": tid})
            fig, ax = plt.subplots(figsize=(7, 3)); ax.plot(template[:100,0], template[:100,1]); ax.plot(template[100:,0], template[100:,1]);
            ax.set_aspect("equal"); ax.grid(alpha=.2); ax.set_title(f"K={k}, component={c}, n={len(members)}"); fig.tight_layout();
            fig.savefig(OUT / "figures" / f"{tid}.png", dpi=180); plt.close(fig)
        fig, ax = plt.subplots(figsize=(8,4))
        for c in range(k):
            t = templates[(k,c)]["coordinates"]; ax.plot(t[:100,0],t[:100,1],label=f"c{c} n={sizes[c]}"); ax.plot(t[100:,0],t[100:,1])
        ax.set_aspect("equal"); ax.grid(alpha=.2); ax.legend(ncol=min(4,k),fontsize=7); ax.set_title(f"K={k} templates"); fig.tight_layout()
        fig.savefig(OUT / "figures" / f"templates_K{k:02d}_overlay.png",dpi=180); plt.close(fig)
    return templates, summary, assignments, model_meta


def rich_properties(source: torch.Tensor, target: torch.Tensor, g: torch.Tensor, le: torch.Tensor, te: torch.Tensor,
                    bounds: tuple, cfg: dict, metrics: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    field, dx, dy = s5.composite_field_array(g, le, te, bounds)
    vx, vy = field[:,:,0], field[:,:,1]; mag = np.linalg.norm(field, axis=2)
    dvx_dy, dvx_dx = np.gradient(vx,dy,dx,edge_order=2); dvy_dy, dvy_dx = np.gradient(vy,dy,dx,edge_order=2)
    curl = dvy_dx-dvx_dy; div = dvx_dx+dvy_dy
    lapx = np.gradient(dvx_dx,dx,axis=1,edge_order=2)+np.gradient(dvx_dy,dy,axis=0,edge_order=2)
    lapy = np.gradient(dvy_dx,dx,axis=1,edge_order=2)+np.gradient(dvy_dy,dy,axis=0,edge_order=2)
    area=(bounds[1]-bounds[0])*(bounds[3]-bounds[2])
    directory=area*float(np.mean(dvx_dx**2+dvx_dy**2+dvy_dx**2+dvy_dy**2))
    fftx=np.fft.fftshift(np.fft.fft2(vx-vx.mean())); ffty=np.fft.fftshift(np.fft.fft2(vy-vy.mean()))
    spectral=np.abs(fftx)**2+np.abs(ffty)**2; fy=np.fft.fftshift(np.fft.fftfreq(REG_GRID,d=dy)); fx=np.fft.fftshift(np.fft.fftfreq(REG_GRID,d=dx))
    fyy,fxx=np.meshgrid(fy,fx,indexing="ij"); radial=np.sqrt((fxx/np.max(np.abs(fx)))**2+(fyy/np.max(np.abs(fy)))**2); st=float(spectral.sum())
    counts,_=np.histogram(mag/max(float(mag.max()),1e-300),bins=ENTROPY_BINS,range=(0.,1.)); pp=counts[counts>0]/max(counts.sum(),1)
    with torch.no_grad():
        x=source.clone(); path=torch.zeros(len(source),dtype=DTYPE); dt=1/int(cfg["steps"])
        for _ in range(int(cfg["steps"])):
            vel=s3.hierarchical_velocity(x,g,le,te,bounds,.15); path+=dt*torch.linalg.norm(vel,dim=1); x=x+dt*vel
        errors=torch.linalg.norm(x-target,dim=1); disp=torch.linalg.norm(x-source,dim=1)
        jac=s3.jacobian(g,le,te,bounds,.15,int(cfg["steps"]),REG_GRID).cpu().numpy()
    row={
      "velocity_magnitude_mean":float(mag.mean()),"velocity_magnitude_rms":float(np.sqrt(np.mean(mag**2))),"velocity_magnitude_max":float(mag.max()),
      "dirichlet_energy_raw":directory,"dirichlet_energy_weighted":float(metrics["weighted_dirichlet"]),
      "correspondence_mse":float(torch.mean(errors**2)),"correspondence_rmse":float(torch.sqrt(torch.mean(errors**2))),"correspondence_max_error":float(errors.max()),
      "lddmm_geodesic_energy":np.nan,"lddmm_geodesic_energy_status":"not_defined: stationary SVF with no RKHS V-norm",
      "velocity_magnitude_variance":float(np.var(mag)),"maximum_displacement":float(disp.max()),"mean_displacement":float(disp.mean()),"rms_displacement":float(torch.sqrt(torch.mean(disp**2))),
      "jacobian_min":float(jac.min()),"jacobian_mean":float(jac.mean()),"jacobian_median":float(np.median(jac)),"jacobian_p01":float(np.quantile(jac,.01)),
      "jacobian_p05":float(np.quantile(jac,.05)),"jacobian_p95":float(np.quantile(jac,.95)),"jacobian_max":float(jac.max()),"jacobian_nonpositive_fraction":float(np.mean(jac<=0)),"jacobian_variance":float(np.var(jac)),
      "curl_rms":float(np.sqrt(np.mean(curl**2))),"curl_mean_abs":float(np.mean(np.abs(curl))),"curl_max_abs":float(np.max(np.abs(curl))),
      "divergence_rms":float(np.sqrt(np.mean(div**2))),"divergence_mean_abs":float(np.mean(np.abs(div))),"divergence_max_abs":float(np.max(np.abs(div))),
      "bending_energy":area*float(np.mean(lapx**2+lapy**2)),"flow_path_length_mean":float(path.mean()),"flow_path_length_rms":float(torch.sqrt(torch.mean(path**2))),"flow_path_length_max":float(path.max()),
      "kinetic_energy":area*float(np.mean(vx**2+vy**2)),"rkhs_norm":np.nan,"rkhs_norm_status":"not_defined: no RKHS kernel/operator or V-norm",
      "spectral_total_energy":st,"spectral_high_frequency_fraction":float(spectral[radial>=HF_THRESHOLD].sum()/max(st,1e-300)),
      "field_entropy":float(-np.sum(pp*np.log(pp))/np.log(ENTROPY_BINS)) if len(pp) else 0.,
      "_jacobian_barrier_raw_computed":float(s2.barrier_raw(torch.tensor(jac),float(cfg["barrier_tau"]))),
    }
    return row,x.cpu().numpy(),s5.vectorize_coefficients(g.numpy(),le.numpy(),te.numpy())


def run_one(payload: tuple) -> dict:
    k, idx, filename, source_np, target_np, component, tid, size, cfg, artifact_text = payload
    torch.set_num_threads(1)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    start=time.perf_counter(); artifact=Path(artifact_text); bounds=tuple(cfg["bounds"])
    source=torch.tensor(source_np,dtype=DTYPE); target=torch.tensor(target_np,dtype=DTYPE)
    base={"steps":cfg["steps"],"basis":"cartesian","barrier_resolution":cfg["barrier_resolution"],"lbfgs_max_iter":cfg["global_lbfgs_max_iter"],"tolerance_grad":cfg["tolerance_grad"],"tolerance_change":cfg["tolerance_change"]}
    ident={"K":k,"q":Q,"target_index":idx,"target_filename":filename,"GMM_component":component,"source_template_id":tid,"source_cluster_size":size,
           "initial_distance":float(np.mean(np.sum((source_np-target_np)**2,axis=1)))}
    try:
        bm,c24,*_=s2.run_schedule(source_np,target_np,bounds,base,s5.SCHEDULE,cfg["barrier_tau"],cfg["barrier_weight"],(6,12,24))
        lm,coeffs,*_=s3.run_local(source,target,torch.tensor(c24,dtype=DTYPE),bounds,{"steps":cfg["steps"],"barrier_resolution":cfg["barrier_resolution"],"barrier_tau":cfg["barrier_tau"],"barrier_weight":cfg["barrier_weight"],"local_lbfgs_max_iter":cfg["local_lbfgs_max_iter"]},48,.15,1.,"joint")
        lm["runtime_seconds"]+=bm["runtime_seconds"]; g,le,te=(torch.tensor(x,dtype=DTYPE) for x in coeffs)
        props,deformed,fieldvec=rich_properties(source,target,g,le,te,bounds,cfg,lm); validity=validate_airfoil_shape(deformed)
        row={**ident,"converged":lm["convergence_status"],"failure":"","geometric_valid":bool(validity["valid"] and props["jacobian_min"]>0 and props["jacobian_nonpositive_fraction"]==0),
             "runtime":time.perf_counter()-start,"iterations":int(lm["closure_evaluations"]),"total_objective":float(lm["total_objective"]),
             "raw_regularisation":float(lm["raw_dirichlet"]),"weighted_regularisation":float(lm["weighted_dirichlet"]),
             "jacobian_barrier_raw":float(lm["raw_jacobian_barrier"]),"jacobian_barrier_weighted":float(lm["weighted_jacobian_barrier"]),**props}
        artifact.parent.mkdir(parents=True,exist_ok=True); tmp=artifact.with_suffix(".tmp.npz")
        np.savez_compressed(tmp,metrics_json=json.dumps(row,sort_keys=True),global_coefficients=coeffs[0],le_coefficients=coeffs[1],te_coefficients=coeffs[2],field_vector=fieldvec,deformed=deformed)
        tmp.replace(artifact); return row
    except Exception as exc:
        row={**ident,"converged":"exception","failure":f"{type(exc).__name__}: {exc}","geometric_valid":False,"runtime":time.perf_counter()-start,"iterations":0}
        artifact.parent.mkdir(parents=True,exist_ok=True); tmp=artifact.with_suffix(".tmp.npz"); np.savez_compressed(tmp,metrics_json=json.dumps(row,sort_keys=True)); tmp.replace(artifact); return row


def load_artifact(path: Path) -> dict | None:
    try:
        with np.load(path) as z: return json.loads(str(z["metrics_json"]))
    except Exception: return None


def reuse_k1(data: dict, cfg: dict, population: np.ndarray, templates: dict, assignments: list[dict]) -> list[dict]:
    amap={(r["K"],r["target_index"]):r for r in assignments}; rows=[]; bounds=tuple(cfg["bounds"]); source_np=data["mean"]
    for idx in population:
        old=STAGE5/"registrations"/f"{idx:04d}.npz"
        with np.load(old) as z:
            oldm=json.loads(str(z["metrics_json"])); g=torch.tensor(z["global_coefficients"],dtype=DTYPE); le=torch.tensor(z["le_coefficients"],dtype=DTYPE); te=torch.tensor(z["te_coefficients"],dtype=DTYPE)
        source=torch.tensor(source_np,dtype=DTYPE); target=torch.tensor(data["shapes"][idx],dtype=DTYPE); props,_,_=rich_properties(source,target,g,le,te,bounds,cfg,{"weighted_dirichlet":oldm["registration_regularisation_surrogate"]*1e-5})
        a=amap[(1,int(idx))]; row={"K":1,"q":Q,"target_index":int(idx),"target_filename":str(data["filenames"][idx]),"GMM_component":0,"source_template_id":"k01_component00","source_cluster_size":1651,
          "converged":oldm["convergence_status"],"failure":oldm["failure_reason"],"geometric_valid":bool(oldm["safe"]),"runtime":float(oldm["wall_runtime_seconds"]),"iterations":int(oldm["closure_evaluations"]),
          "initial_distance":float(np.mean(np.sum((source_np-data["shapes"][idx])**2,axis=1))),
          "total_objective":float(props["correspondence_mse"]+float(oldm["registration_regularisation_surrogate"])*1e-5+props["_jacobian_barrier_raw_computed"]*float(cfg["barrier_weight"])),
          "raw_regularisation":float(oldm["registration_regularisation_surrogate"]),"weighted_regularisation":float(oldm["registration_regularisation_surrogate"])*1e-5,
          "jacobian_barrier_raw":props["_jacobian_barrier_raw_computed"],"jacobian_barrier_weighted":props["_jacobian_barrier_raw_computed"]*float(cfg["barrier_weight"]),**props}
        rows.append(row)
    return rows


def run_registrations(data: dict,cfg: dict,population: np.ndarray,templates: dict,assignments: list[dict],workers:int,only:list[tuple[int,int]]|None=None) -> list[dict]:
    amap={(r["K"],r["target_index"]):r for r in assignments}; rows=[]; todo=[]
    if only is None: rows.extend(reuse_k1(data,cfg,population,templates,assignments)); pairs=[(k,int(i)) for k in range(2,9) for i in population]
    else: pairs=only
    for k,idx in pairs:
        a=amap[(k,idx)]; c=int(a["MAP_component"]); t=templates[(k,c)]; path=OUT/"registrations"/f"K{k:02d}"/f"{idx:04d}.npz"
        existing=load_artifact(path)
        if existing is not None and int(existing.get("K",-1))==k and int(existing.get("target_index",-1))==idx: rows.append(existing); continue
        todo.append((k,idx,str(data["filenames"][idx]),t["coordinates"],data["shapes"][idx],c,t["id"],t["size"],cfg,str(path)))
    print(f"resume: {len(rows)} rows reused; {len(todo)} new registrations",flush=True)
    if workers==1:
        for n,p in enumerate(todo,1):
            r=run_one(p); rows.append(r); print(f"[{n}/{len(todo)}] K={r['K']} target={r['target_index']} valid={r['geometric_valid']} runtime={r['runtime']:.2f}s",flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(run_one,p):p[:2] for p in todo}
            for n,f in enumerate(as_completed(futures),1):
                r=f.result(); rows.append(r); print(f"[{n}/{len(todo)}] K={r['K']} target={r['target_index']} valid={r['geometric_valid']} runtime={r['runtime']:.2f}s",flush=True)
    return sorted(rows,key=lambda r:(int(r["target_index"]),int(r["K"])))


def final_tables(rows:list[dict],population:np.ndarray) -> dict:
    full=[]
    for r in rows: full.append({c:r.get(c,np.nan) for c in SCHEMA})
    write_csv(OUT/"tables"/"velocity_field_properties_by_target_and_k.csv",full,SCHEMA)
    numeric=[c for c in SCHEMA if c not in {"K","q","target_index","target_filename","GMM_component","source_template_id","converged","failure","lddmm_geodesic_energy_status","rkhs_norm_status"}]
    summary=[]
    for k in KS:
        rr=[r for r in full if r["K"]==k]
        for col in numeric:
            vals=np.asarray([r[col] for r in rr if isinstance(r[col],(int,float,np.integer,np.floating)) and np.isfinite(r[col])],dtype=float)
            if not len(vals): continue
            summary.append({"K":k,"property":col,"count":len(vals),"mean":float(vals.mean()),"standard_deviation":float(vals.std(ddof=1)) if len(vals)>1 else np.nan,
             "standard_error":float(vals.std(ddof=1)/math.sqrt(len(vals))) if len(vals)>1 else np.nan,"median":float(np.median(vals)),"p10":float(np.quantile(vals,.1)),"p25":float(np.quantile(vals,.25)),
             "p75":float(np.quantile(vals,.75)),"p90":float(np.quantile(vals,.9)),"p95":float(np.quantile(vals,.95)),"minimum":float(vals.min()),"maximum":float(vals.max())})
    write_csv(OUT/"tables"/"velocity_field_properties_summary_by_k.csv",summary)
    by={(int(r["target_index"]),int(r["K"])):r for r in full}; coverage=[]
    for idx in population:
        row={"target_index":int(idx)}
        for k in KS:
            r=by.get((int(idx),k)); row[f"K{k}_valid"]=bool(r and not r["failure"] and r["geometric_valid"])
        row["valid_all_K"]=all(row[f"K{k}_valid"] for k in KS); coverage.append(row)
    write_csv(OUT/"tables"/"paired_target_coverage.csv",coverage)
    failures=[]
    for k in KS:
        rr=[r for r in full if r["K"]==k]; failures.append({"K":k,"attempted":len(rr),"failures":sum(bool(r["failure"]) for r in rr),"geometrically_invalid":sum(not bool(r["geometric_valid"]) for r in rr)})
    write_csv(OUT/"tables"/"registration_failure_summary.csv",failures)
    return {"rows":len(full),"unique_pairs":len(by),"complete_case_targets":sum(r["valid_all_K"] for r in coverage),"failures":failures}


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--stage",choices=("prepare","smoke","full","finalise","all"),default="all"); ap.add_argument("--workers",type=int,default=2); args=ap.parse_args()
    for d in ("templates","assignments","registrations","properties","tables","logs","figures"):(OUT/d).mkdir(parents=True,exist_ok=True)
    data,cfg,population,matrix=load_inputs(); scores=pca_scores(matrix); templates,ts,assignments,models=make_templates(data,population,scores)
    write_csv(OUT/"templates"/"template_summary.csv",ts); write_csv(OUT/"assignments"/"airfoil_gmm_assignments_q20_k1_k8.csv",assignments)
    hashes={str(p.relative_to(ROOT)):sha256(p) for p in [DATA,STAGE5/"pca"/"pca_q20.npz",STAGE5/"velocity_field_population_indices.npy",STAGE5/"config.json"]+[STAGE5/"gmm"/"models"/f"q20_k{k:02d}_full.npz" for k in range(2,9)]}
    config={"q":20,"K":list(KS),"population_size":1651,"excluded_auxiliary_indices":[3,206,408],"workers":args.workers,"frozen_registration":cfg,"pca_file":"outputs/airfoil_template_number_stage5_screening/pca/pca_q20.npz",
      "gmm_models":models,"new_registration_upper_bound":11557,"resume":"per-(K,target) atomic NPZ; valid metadata artifacts skipped","property_definitions":{"grid":"51x51 common physical grid","dirichlet":"domain area * mean(sum of squared first spatial derivatives)",
      "bending":"domain area * mean(squared componentwise velocity Laplacians)","kinetic":"domain area * mean(||v||^2)","spectral":"unnormalised squared 2D FFT after component mean removal; high frequency radial normalised radius >=0.5",
      "entropy":"Shannon entropy/log(32), 32 bins on each field magnitude/max magnitude in [0,1]","flow_path":"15-step Euler trajectory quadrature"},"input_sha256_before":hashes}
    atomic_json(OUT/"experiment_config.json",config); atomic_json(OUT/"config.json",config)
    if args.stage=="prepare": return
    if args.stage in ("smoke","all"):
        ordinary=int(population[len(population)//2]); extreme=int(population[np.argmax(np.linalg.norm(scores,axis=1))]); smoke=run_registrations(data,cfg,population,templates,assignments,1,[(k,i) for k in (2,8) for i in (ordinary,extreme)])
        write_csv(OUT/"tables"/"smoke_test_results.csv",smoke,SCHEMA)
        if any(r["failure"] for r in smoke): raise RuntimeError("Smoke test registration failure; full run not started")
        atomic_json(OUT/"logs"/"smoke_validation.json",{"status":"PASS","ordinary_target":ordinary,"extreme_target":extreme,"rows":len(smoke),"schema":SCHEMA})
        if args.stage=="smoke": return
    rows=[]
    if args.stage in ("full","all"): rows=run_registrations(data,cfg,population,templates,assignments,args.workers)
    elif args.stage=="finalise":
        rows=reuse_k1(data,cfg,population,templates,assignments)
        for k in range(2,9):
            for idx in population:
                r=load_artifact(OUT/"registrations"/f"K{k:02d}"/f"{idx:04d}.npz")
                if r is not None: rows.append(r)
    result=final_tables(rows,population); hashes_after={p:sha256(ROOT/p) for p in hashes}
    checks={"q_exact":Q==20,"K_exact":list(KS)==list(range(1,9)),"population_exact":len(population)==1651,"unique_pairs":result["unique_pairs"]==1651*8,
      "templates_valid":all(r["template_validity"] for r in ts),"templates_shape":all(t["coordinates"].shape==(200,2) for t in templates.values()),"input_hashes_unchanged":hashes==hashes_after,
      "lddmm_not_fabricated":all(np.isnan(r.get("lddmm_geodesic_energy",np.nan)) for r in rows),"rkhs_not_fabricated":all(np.isnan(r.get("rkhs_norm",np.nan)) for r in rows)}
    atomic_json(OUT/"validation.json",{"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"counts":result,"input_sha256_after":hashes_after})
    size=sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    report=f"""# q=20, K=1..8 assigned-template registration data generation\n\nThis report contains data-generation facts only; it makes no comparison, ranking, or K selection.\n\n- Population: 1651 Stage-5-safe targets; excluded auxiliary indices: 3, 206, 408.\n- K=1: reused saved global-mean registrations; properties missing from the old CSV were re-extracted from saved coefficients without optimisation.\n- K=2..8: MAP assignment from the exact saved q=20 full-covariance GMM; one hard-assignment 200x2 coordinate mean per component; only the assigned template was registered.\n- Frozen registration: `{json.dumps(cfg,sort_keys=True)}`.\n- Property definitions: `{json.dumps(config['property_definitions'],sort_keys=True)}`. Classical LDDMM energy and RKHS norm are not defined for this SVF model.\n- Rows: {result['rows']}; unique target/K pairs: {result['unique_pairs']}; complete-case targets: {result['complete_case_targets']}.\n- Failure diagnostics: `{json.dumps(result['failures'],sort_keys=True)}`.\n- Output storage: {size/1024/1024:.1f} MiB.\n\nNo model-selection conclusion is included.\n"""
    (OUT/"data_generation_report.md").write_text(report,encoding="utf-8")

if __name__=="__main__": main()
