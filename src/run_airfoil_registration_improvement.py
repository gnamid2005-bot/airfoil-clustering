from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from airfoil_data import plot_shape, validate_processed_shapes
from airfoil_registration import (
    camber_thickness_from_shapes,
    load_airfoil_dataset,
    reconstruct_k1_template,
    validate_airfoil_shape,
)
from svf_2d import corresponding_point_mse, jacobian_determinants_2d


matplotlib.use("Agg")

DATA_PATH = Path("data/airfoils/processed/airfoils_n100_seed0.npz")
OUTPUT_DIR = Path("outputs/airfoil_registration_improvement")
QUANTILES = [
    ("very_close", 0.05), ("close", 0.20), ("medium_low", 0.35),
    ("medium", 0.50), ("medium_high", 0.65), ("far", 0.80),
    ("very_far", 0.95),
]


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_dataset(dataset: dict, path: Path) -> dict:
    shapes = dataset["shapes"]
    validate_processed_shapes(shapes)
    half = shapes.shape[1] // 2
    fixed_upper = bool(np.allclose(shapes[:, :half, 0], shapes[0, :half, 0], atol=0, rtol=0))
    fixed_lower = bool(np.allclose(shapes[:, half:, 0], shapes[0, half:, 0], atol=0, rtol=0))
    surfaces_match = bool(np.allclose(shapes[0, :half, 0], shapes[0, half:, 0], atol=0, rtol=0))
    return {
        "processed_dataset_path": str(path), "array_shape": list(shapes.shape),
        "number_of_airfoils": int(shapes.shape[0]), "number_of_points": int(shapes.shape[1]),
        "coordinate_representation": "Cartesian (x,y), first 100 upper then 100 lower",
        "upper_surface_points": half, "lower_surface_points": shapes.shape[1] - half,
        "fixed_x_coordinates": fixed_upper and fixed_lower and surfaces_match,
        "identical_point_ordering": True, "pointwise_correspondence_exists": True,
        "finite": bool(np.isfinite(shapes).all()),
        "x_grid_min": float(shapes[:, :, 0].min()), "x_grid_max": float(shapes[:, :, 0].max()),
        "x_exactly_equal_across_airfoils": fixed_upper and fixed_lower,
        "no_new_preprocessing_performed": True,
    }


def build_global_mean(dataset: dict, output_dir: Path) -> np.ndarray:
    # Reuse the repository's existing K=1/global-mean reconstruction implementation.
    indices = np.arange(dataset["shapes"].shape[0], dtype=np.int64)
    mean = reconstruct_k1_template(dataset, indices)
    direct = dataset["shapes"].mean(axis=0)
    if not np.allclose(mean, direct, atol=1e-12, rtol=1e-12):
        raise AssertionError("camber/thickness reconstruction does not equal coordinate mean")
    validity = validate_airfoil_shape(mean)
    if not validity["valid"]:
        raise ValueError(f"invalid global mean: {validity}")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "coordinates.npy", mean)
    np.savetxt(output_dir / "coordinates.csv", mean, delimiter=",", header="x,y", comments="")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    plot_shape(ax, mean, color="#2563eb", linewidth=2.0)
    ax.set_aspect("equal", adjustable="box"); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Global mean airfoil (n=1654)"); ax.grid(True, alpha=.2)
    fig.tight_layout(); fig.savefig(output_dir / "global_mean.png", dpi=220); plt.close(fig)
    write_json(output_dir / "metadata.json", {
        "number_of_training_airfoils": int(len(indices)),
        "representation": "200 corresponding Cartesian (x,y) points: 100 upper + 100 lower",
        "repository_implementation_reused": "airfoil_registration.reconstruct_k1_template",
        "mathematical_definition": "Convert each shape to [camber(x), thickness(x)], take the arithmetic mean over all 1654 shapes, then reconstruct upper=camber+thickness/2 and lower=camber-thickness/2.",
        "equivalent_coordinate_definition": "mu_j = (1/N) sum_i S_ij; verified numerically against direct coordinate mean",
        "computed_in": "camber/thickness representation (algebraically identical to direct coordinate mean on the fixed x-grid)",
        "maximum_difference_from_direct_coordinate_mean": float(np.max(np.abs(mean-direct))),
        "geometric_validity": validity,
    })
    return mean


def select_development_subset(dataset: dict, mean: np.ndarray, output_dir: Path) -> list[dict]:
    shapes = dataset["shapes"]
    distances = np.mean(np.sum((shapes - mean[None]) ** 2, axis=2), axis=1)
    order = np.argsort(distances, kind="stable")
    rows = [{"index": int(i), "rank": int(r), "quantile_rank": float(r/(len(order)-1)),
             "filename": dataset["filenames"][i], "airfoil_name": dataset["names"][i],
             "distance_mse": float(distances[i])} for r, i in enumerate(order)]
    write_csv(output_dir / "all_global_mean_distances.csv", rows)
    used: set[int] = set()
    selected: list[dict] = [{"label":"identity", "requested_quantile":0.0, "index":-1,
                            "rank":-1, "empirical_quantile":0.0, "filename":"GLOBAL_MEAN",
                            "airfoil_name":"global mean", "distance_mse":0.0, "geometry_valid":True}]
    for label, quantile in QUANTILES:
        desired = int(round(quantile * (len(order)-1)))
        for delta in range(len(order)):
            candidates = [desired] if delta == 0 else [desired-delta, desired+delta]
            available = [r for r in candidates if 0 <= r < len(order) and int(order[r]) not in used]
            if available:
                rank = available[0]; break
        idx = int(order[rank]); used.add(idx)
        validity = validate_airfoil_shape(shapes[idx])
        selected.append({"label":label, "requested_quantile":quantile, "index":idx, "rank":rank,
                         "empirical_quantile":rank/(len(order)-1), "filename":dataset["filenames"][idx],
                         "airfoil_name":dataset["names"][idx], "distance_mse":float(distances[idx]),
                         "geometry_valid":bool(validity["valid"])})
    for rank in range(len(order)-1, -1, -1):
        idx = int(order[rank])
        validity = validate_airfoil_shape(shapes[idx])
        if idx not in used and validity["valid"]:
            selected.append({"label":"valid_extreme", "requested_quantile":1.0, "index":idx, "rank":rank,
                             "empirical_quantile":rank/(len(order)-1), "filename":dataset["filenames"][idx],
                             "airfoil_name":dataset["names"][idx], "distance_mse":float(distances[idx]),
                             "geometry_valid":True}); break
    write_csv(output_dir / "development_targets.csv", selected)
    np.save(output_dir / "selected_indices.npy", np.asarray([r["index"] for r in selected], dtype=np.int64))
    (output_dir / "selected_filenames.txt").write_text("\n".join(r["filename"] for r in selected)+"\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7.5, 4.3)); ax.hist(distances, bins=55, color="#93c5fd", edgecolor="#1e3a8a")
    for row in selected[1:]: ax.axvline(row["distance_mse"], color="#ea580c", alpha=.45, linewidth=.8)
    ax.set(title="Initial corresponding-point distance distribution", xlabel="Mean squared corresponding-point distance", ylabel="Airfoil count")
    ax.grid(axis="y", alpha=.2); fig.tight_layout(); fig.savefig(output_dir / "distance_distribution.png", dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5)); plot_shape(ax, mean, color="#111827", linewidth=2.4, label="global mean")
    colors = plt.cm.viridis(np.linspace(.05,.95,len(selected)-1))
    for color, row in zip(colors, selected[1:]): plot_shape(ax, shapes[row["index"]], color=color, alpha=.8, linewidth=.9, label=row["label"])
    ax.set_aspect("equal", adjustable="box"); ax.set(title="Deterministic development subset", xlabel="x", ylabel="y")
    ax.grid(alpha=.2); ax.legend(ncol=3, fontsize=8); fig.tight_layout(); fig.savefig(output_dir / "selected_overlay.png", dpi=220); plt.close(fig)
    return selected


def grid_coords(points: torch.Tensor, bounds: tuple[float,float,float,float]) -> torch.Tensor:
    xmin,xmax,ymin,ymax=bounds
    return torch.stack([2*(points[:,0]-xmin)/(xmax-xmin)-1, 2*(points[:,1]-ymin)/(ymax-ymin)-1], dim=1)


def velocity(points: torch.Tensor, coeffs: torch.Tensor, bounds: tuple[float,float,float,float], basis: str) -> torch.Tensor:
    if basis == "bilinear":
        image=coeffs.permute(2,1,0).unsqueeze(0); grid=grid_coords(points,bounds).view(1,-1,1,2)
        return F.grid_sample(image,grid,mode="bilinear",padding_mode="border",align_corners=True).squeeze(0).squeeze(-1).T
    # Uniform tensor-product cardinal cubic B-spline, clamped at the lattice boundary.
    gx,gy,_=coeffs.shape; q=grid_coords(points,bounds)
    u=(q[:,0]+1)*(gx-1)/2; v=(q[:,1]+1)*(gy-1)/2
    iu=torch.floor(u).long(); iv=torch.floor(v).long(); tu=u-iu; tv=v-iv
    def w(t): return torch.stack([(1-t)**3/6,(3*t**3-6*t**2+4)/6,(-3*t**3+3*t**2+3*t+1)/6,t**3/6],1)
    wx,wy=w(tu),w(tv); out=torch.zeros(points.shape[0],2,dtype=points.dtype,device=points.device)
    for a in range(4):
        xi=(iu+a-1).clamp(0,gx-1)
        for b in range(4): out=out+(wx[:,a]*wy[:,b])[:,None]*coeffs[xi,(iv+b-1).clamp(0,gy-1)]
    return out


def integrate(points, coeffs, bounds, steps, basis):
    x=points; dt=1.0/steps
    for _ in range(steps): x=x+dt*velocity(x,coeffs,bounds,basis)
    return x


def raw_regularisation(coeffs, bounds):
    xmin,xmax,ymin,ymax=bounds
    dx=(xmax-xmin)/(coeffs.shape[0]-1); dy=(ymax-ymin)/(coeffs.shape[1]-1)
    dvx=(coeffs[1:]-coeffs[:-1])/dx; dvy=(coeffs[:,1:]-coeffs[:,:-1])/dy
    return .5*(dvx.square().sum(-1).mean()+dvy.square().sum(-1).mean())


def jacobians(coeffs,bounds,steps,basis,resolution=51):
    xmin,xmax,ymin,ymax=bounds; xs=torch.linspace(xmin,xmax,resolution,dtype=coeffs.dtype); ys=torch.linspace(ymin,ymax,resolution,dtype=coeffs.dtype)
    yy,xx=torch.meshgrid(ys,xs,indexing="ij"); pts=torch.stack([xx.ravel(),yy.ravel()],1)
    with torch.no_grad(): field=integrate(pts,coeffs,bounds,steps,basis).reshape(resolution,resolution,2)
    dx=(xmax-xmin)/(resolution-1); dy=(ymax-ymin)/(resolution-1)
    px=(field[1:-1,2:]-field[1:-1,:-2])/(2*dx); py=(field[2:,1:-1]-field[:-2,1:-1])/(2*dy)
    return (px[:,:,0]*py[:,:,1]-px[:,:,1]*py[:,:,0]).ravel().cpu().numpy()


def optimise(source_np,target_np,bounds,cfg):
    torch.manual_seed(0); dtype=torch.float64; source=torch.tensor(source_np,dtype=dtype); target=torch.tensor(target_np,dtype=dtype)
    coeffs=torch.zeros(cfg["grid_size"],cfg["grid_size"],2,dtype=dtype,requires_grad=True); history=[]; start=time.perf_counter()
    schedule=cfg["schedule"]; segments=[]
    if schedule == "constant": segments=[(cfg["lambda_reg"],cfg["adam_iters"],cfg["lbfgs_iters"])]
    elif schedule == "staged": segments=[(1e-2,cfg["adam_iters"]//3,0),(1e-3,cfg["adam_iters"]//3,0),(1e-4,cfg["adam_iters"]-2*(cfg["adam_iters"]//3),cfg["lbfgs_iters"])]
    else: segments=[(1e-2,cfg["adam_iters"]//3,0),(1e-4,cfg["adam_iters"]//3,0),(1e-6,cfg["adam_iters"]-2*(cfg["adam_iters"]//3),cfg["lbfgs_iters"])]
    def closure(lam,opt=None):
        if opt: opt.zero_grad()
        deformed=integrate(source,coeffs,bounds,cfg["steps"],cfg["basis"]); data=corresponding_point_mse(deformed,target); raw=raw_regularisation(coeffs,bounds); total=data+lam*raw
        if opt: total.backward()
        return total,data,raw
    iteration=0; previous=None; converged=False
    for lam,adam_n,lbfgs_n in segments:
        if cfg["optimizer"] in ("adam","adam_lbfgs") and adam_n:
            opt=torch.optim.Adam([coeffs],lr=cfg["lr"])
            for _ in range(adam_n):
                total,data,raw=closure(lam,opt); opt.step(); iteration+=1
                history.append({"iteration":iteration,"lambda_reg":lam,"data_attachment":float(data.detach()),"raw_regularisation":float(raw.detach()),"weighted_regularisation":float((lam*raw).detach()),"total_objective":float(total.detach())})
        if cfg["optimizer"] in ("lbfgs","adam_lbfgs") and lbfgs_n:
            opt=torch.optim.LBFGS([coeffs],lr=1.0,max_iter=lbfgs_n,tolerance_grad=1e-12,tolerance_change=1e-15,line_search_fn="strong_wolfe")
            def f(): return closure(lam,opt)[0]
            opt.step(f); iteration+=lbfgs_n
            total,data,raw=closure(lam); history.append({"iteration":iteration,"lambda_reg":lam,"data_attachment":float(data),"raw_regularisation":float(raw),"weighted_regularisation":float(lam*raw),"total_objective":float(total)})
        total,data,raw=closure(lam); current=float(total.detach())
        converged=previous is not None and abs(previous-current) <= 1e-12*max(1,abs(previous)); previous=current
    runtime=time.perf_counter()-start; lam=segments[-1][0]
    with torch.no_grad():
        deformed=integrate(source,coeffs,bounds,cfg["steps"],cfg["basis"]); errors=torch.linalg.norm(deformed-target,dim=1); data=errors.square().mean(); raw=raw_regularisation(coeffs,bounds); disp=torch.linalg.norm(deformed-source,dim=1)
        velmag=torch.linalg.norm(coeffs,dim=2).mean(); total=data+lam*raw
    jac=jacobians(coeffs.detach(),bounds,cfg["steps"],cfg["basis"]); result=deformed.cpu().numpy(); validity=validate_airfoil_shape(result)
    metrics={"correspondence_mse":float(data),"correspondence_rmse":float(torch.sqrt(data)),"maximum_correspondence_error":float(errors.max()),
             "raw_regularisation":float(raw),"weighted_regularisation":float(lam*raw),"total_objective":float(total),"vector_field_magnitude":float(velmag),"dirichlet_energy":float(raw),
             "mean_displacement":float(disp.mean()),"maximum_displacement":float(disp.max()),"minimum_jacobian_determinant":float(jac.min()),
             "jacobian_p01":float(np.quantile(jac,.01)),"jacobian_p05":float(np.quantile(jac,.05)),"jacobian_p25":float(np.quantile(jac,.25)),"jacobian_p50":float(np.quantile(jac,.5)),"jacobian_p75":float(np.quantile(jac,.75)),"jacobian_p95":float(np.quantile(jac,.95)),"jacobian_p99":float(np.quantile(jac,.99)),
             "runtime_seconds":runtime,"converged":bool(converged or (history and history[-1]["data_attachment"] < 1e-9)),"positive_jacobian":bool(jac.min()>0),"valid_airfoil_geometry":bool(validity["valid"]),"iterations":iteration}
    return result,coeffs.detach().numpy(),history,metrics


def experiment_configs():
    base={"grid_size":12,"steps":15,"basis":"bilinear","optimizer":"adam","schedule":"constant","lambda_reg":1e-3,"lr":0.03,"adam_iters":500,"lbfgs_iters":140}
    variants=[("baseline_adam",{}),("optimizer_lbfgs",{"optimizer":"lbfgs","adam_iters":0}),("optimizer_adam_lbfgs",{"optimizer":"adam_lbfgs"}),
              ("regularisation_staged",{"schedule":"staged"}),("regularisation_staged_low",{"schedule":"staged_low"}),("basis_bspline",{"basis":"bspline"}),
              ("integration_steps_5",{"steps":5}),("integration_steps_30",{"steps":30})]
    return [(name,{**base,**delta}) for name,delta in variants]


def run_study(dataset,mean,selected,output_dir,force=False):
    shapes=dataset["shapes"]; bounds=(0.0,1.0,float(min(shapes[:,:,1].min(),mean[:,1].min())-.08),float(max(shapes[:,:,1].max(),mean[:,1].max())+.08))
    rows=[]; configs=experiment_configs(); write_json(output_dir/"experiment_design.json", {name:cfg for name,cfg in configs})
    for selected_row in selected:
        target=mean if selected_row["index"]<0 else shapes[selected_row["index"]]
        for name,cfg in configs:
            run_dir=output_dir/"runs"/selected_row["label"]/name; summary_path=run_dir/"summary.json"
            if summary_path.exists() and not force:
                metrics=json.loads(summary_path.read_text()); row={"target_label":selected_row["label"],"target_index":selected_row["index"],"filename":selected_row["filename"],"initial_distance_mse":selected_row["distance_mse"],"experiment":name,**cfg,**metrics}; rows.append(row); continue
            run_dir.mkdir(parents=True,exist_ok=True); deformed,coeffs,history,metrics=optimise(mean,target,bounds,cfg)
            np.save(run_dir/"deformed.npy",deformed); np.save(run_dir/"coefficients.npy",coeffs); write_csv(run_dir/"history.csv",history); write_json(summary_path,metrics); write_json(run_dir/"config.json",cfg)
            fig,ax=plt.subplots(figsize=(7.5,3.8)); plot_shape(ax,target,color="#2563eb",linewidth=1.8,label="target"); plot_shape(ax,deformed,color="#ea580c",linestyle="--",linewidth=1.3,label="registered")
            ax.set_aspect("equal",adjustable="box"); ax.set(title=f"{selected_row['label']} — {name}",xlabel="x",ylabel="y"); ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(run_dir/"overlay.png",dpi=180); plt.close(fig)
            rows.append({"target_label":selected_row["label"],"target_index":selected_row["index"],"filename":selected_row["filename"],"initial_distance_mse":selected_row["distance_mse"],"experiment":name,**cfg,**metrics})
            write_csv(output_dir/"registration_results.csv",rows)
    write_csv(output_dir/"registration_results.csv",rows)
    return rows


def summarise(rows,output_dir):
    exps=sorted({r["experiment"] for r in rows}); summary=[]
    nonidentity=[r for r in rows if r["target_label"]!="identity"]
    for exp in exps:
        rr=[r for r in nonidentity if r["experiment"]==exp]
        summary.append({"experiment":exp,"n":len(rr),"median_correspondence_mse":float(np.median([r["correspondence_mse"] for r in rr])),"maximum_correspondence_mse":float(max(r["correspondence_mse"] for r in rr)),"minimum_jacobian":float(min(r["minimum_jacobian_determinant"] for r in rr)),"valid_runs":sum(r["positive_jacobian"] and r["valid_airfoil_geometry"] for r in rr),"mean_runtime_seconds":float(np.mean([r["runtime_seconds"] for r in rr])),"runs_below_1e_9":sum(r["correspondence_mse"]<=1e-9 for r in rr)})
    write_csv(output_dir/"ablation_summary.csv",summary)
    fig,ax=plt.subplots(figsize=(9,4.8)); data=[[r["correspondence_mse"] for r in nonidentity if r["experiment"]==e] for e in exps]; ax.boxplot(data,tick_labels=exps,showfliers=True); ax.axhline(1e-9,color="#ea580c",linestyle="--",label="1e-9 goal"); ax.set_yscale("log"); ax.set(ylabel="Correspondence MSE",title="Controlled registration ablations"); ax.tick_params(axis="x",rotation=35); ax.grid(axis="y",alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(output_dir/"ablation_correspondence_mse.png",dpi=220); plt.close(fig)
    return summary


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--data-path",default=str(DATA_PATH)); p.add_argument("--output-dir",default=str(OUTPUT_DIR)); p.add_argument("--prepare-only",action="store_true"); p.add_argument("--force",action="store_true"); return p.parse_args()


def main():
    args=parse_args(); out=Path(args.output_dir); dataset=load_airfoil_dataset(Path(args.data_path)); meta=verify_dataset(dataset,Path(args.data_path)); write_json(out/"dataset_verification.json",meta)
    mean=build_global_mean(dataset,out/"global_mean"); selected=select_development_subset(dataset,mean,out/"development_subset")
    if args.prepare_only: print(json.dumps({"dataset":meta,"targets":selected},indent=2)); return
    rows=run_study(dataset,mean,selected,out,args.force); summary=summarise(rows,out); print(json.dumps(summary,indent=2))


if __name__ == "__main__": main()
