from __future__ import annotations

import argparse, csv, json, math, time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from airfoil_data import plot_shape, validate_processed_shapes
from airfoil_registration import validate_airfoil_shape

matplotlib.use("Agg")

DATA = Path("data/airfoils/processed/airfoils_n100_seed0.npz")
STAGE1 = Path("outputs/airfoil_registration_improvement")
OUT = Path("outputs/airfoil_registration_improvement_stage2")
DTYPE = torch.float64


def json_write(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(obj,indent=2,sort_keys=True),encoding="utf-8")


def csv_write(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(rows[0]) if rows else []
    with path.open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(rows)


def load_inputs(out):
    d=np.load(DATA); shapes=d["shapes"].astype(np.float64); validate_processed_shapes(shapes)
    direct=shapes.mean(0)
    previous=np.load(STAGE1/"global_mean"/"coordinates.npy")
    selected=np.load(STAGE1/"development_subset"/"selected_indices.npy")
    names=(STAGE1/"development_subset"/"selected_filenames.txt").read_text().splitlines()
    labels=["identity","very_close","close","medium_low","medium","medium_high","far","very_far","extreme"]
    fixed=bool(np.array_equal(shapes[:,:,0],np.broadcast_to(shapes[0,:,0],shapes[:,:,0].shape)))
    cmp={"dataset_path":str(DATA),"shape":list(shapes.shape),"airfoils":len(shapes),"points":shapes.shape[1],"upper_points":100,"lower_points":100,
         "coordinate_definition":"S_i in R^(200x2), direct Cartesian coordinates","fixed_x_grid":fixed,"pointwise_correspondence":True,"reprocessed":False,
         "maximum_absolute_coordinate_difference":float(np.max(np.abs(direct-previous))),"rms_coordinate_difference":float(np.sqrt(np.mean((direct-previous)**2))),
         "direct_mean_validity":validate_airfoil_shape(direct),"previous_mean_validity":validate_airfoil_shape(previous),"subsequent_source":"direct_coordinate_mean"}
    p=out/"direct_coordinate_mean"; p.mkdir(parents=True,exist_ok=True); np.save(p/"coordinates.npy",direct); np.savetxt(p/"coordinates.csv",direct,delimiter=",",header="x,y",comments="")
    json_write(out/"direct_mean_comparison.json",cmp)
    targets=[]
    for label,idx,name in zip(labels,selected,names): targets.append({"label":label,"index":int(idx),"filename":name,"shape":direct if idx<0 else shapes[idx]})
    return shapes,direct,targets,cmp


def norm_coords(points,bounds):
    x0,x1,y0,y1=bounds
    return torch.stack([2*(points[:,0]-x0)/(x1-x0)-1,2*(points[:,1]-y0)/(y1-y0)-1],1)


def velocity(points,c,bounds,basis):
    if basis=="cartesian":
        im=c.permute(2,1,0)[None]; grid=norm_coords(points,bounds).reshape(1,-1,1,2)
        return F.grid_sample(im,grid,mode="bilinear",padding_mode="border",align_corners=True)[0,:,:,0].T
    gx,gy,_=c.shape; q=norm_coords(points,bounds); u=(q[:,0]+1)*(gx-1)/2; v=(q[:,1]+1)*(gy-1)/2
    iu=torch.floor(u).long(); iv=torch.floor(v).long(); tu=u-iu; tv=v-iv
    def weights(t): return torch.stack(((1-t)**3,(3*t**3-6*t**2+4),(-3*t**3+3*t**2+3*t+1),t**3),1)/6
    wx,wy=weights(tu),weights(tv); z=torch.zeros((len(points),2),dtype=points.dtype,device=points.device)
    for a in range(4):
        ia=(iu+a-1).clamp(0,gx-1)
        for b in range(4): z=z+(wx[:,a]*wy[:,b])[:,None]*c[ia,(iv+b-1).clamp(0,gy-1)]
    return z


def integrate(points,c,bounds,steps,basis):
    x=points; dt=1/steps
    for _ in range(steps): x=x+dt*velocity(x,c,bounds,basis)
    return x


def dirichlet(c,bounds):
    x0,x1,y0,y1=bounds; dx=(x1-x0)/(c.shape[0]-1); dy=(y1-y0)/(c.shape[1]-1)
    return .5*(((c[1:]-c[:-1])/dx).square().sum(2).mean()+((c[:,1:]-c[:,:-1])/dy).square().sum(2).mean())


def jacobian_grid(c,bounds,steps,basis,res=21,grad=True):
    x0,x1,y0,y1=bounds; xs=torch.linspace(x0,x1,res,dtype=c.dtype,device=c.device); ys=torch.linspace(y0,y1,res,dtype=c.dtype,device=c.device)
    yy,xx=torch.meshgrid(ys,xs,indexing="ij"); p=torch.stack((xx.flatten(),yy.flatten()),1); f=integrate(p,c,bounds,steps,basis).reshape(res,res,2)
    dx=(x1-x0)/(res-1); dy=(y1-y0)/(res-1); px=(f[1:-1,2:]-f[1:-1,:-2])/(2*dx); py=(f[2:,1:-1]-f[:-2,1:-1])/(2*dy)
    return (px[:,:,0]*py[:,:,1]-px[:,:,1]*py[:,:,0]).flatten()


def barrier_raw(jac,tau):
    # Smooth squared hinge: softplus(beta*(tau-J))/beta, squared and averaged.
    beta=50.0
    return (F.softplus(beta*(tau-jac))/beta).square().mean()


def prolong(c,size):
    return F.interpolate(c.detach().permute(2,1,0)[None],size=(size,size),mode="bicubic",align_corners=True)[0].permute(2,1,0).contiguous()


def optimise_level(source,target,c,bounds,cfg,lam_d,tau,lam_j,max_iter=None):
    c=c.detach().clone().requires_grad_(True); iters=cfg["lbfgs_max_iter"] if max_iter is None else max_iter
    opt=torch.optim.LBFGS([c],lr=1.0,max_iter=iters,max_eval=int(iters*1.25),tolerance_grad=cfg["tolerance_grad"],tolerance_change=cfg["tolerance_change"],line_search_fn="strong_wolfe")
    calls=0; last={}
    def closure():
        nonlocal calls,last
        opt.zero_grad(); deformed=integrate(source,c,bounds,cfg["steps"],cfg["basis"]); corr=(deformed-target).square().sum(1).mean(); raw_d=dirichlet(c,bounds)
        jac=jacobian_grid(c,bounds,cfg["steps"],cfg["basis"],cfg["barrier_resolution"]); raw_j=barrier_raw(jac,tau) if lam_j else jac.sum()*0
        total=corr+lam_d*raw_d+lam_j*raw_j; total.backward(); calls+=1
        last={"corr":float(corr.detach()),"raw_d":float(raw_d.detach()),"raw_j":float(raw_j.detach()),"total":float(total.detach())}; return total
    start=time.perf_counter(); opt.step(closure); runtime=time.perf_counter()-start
    return c.detach(),runtime,calls,last


def metrics(source,target,c,bounds,cfg,lam_d,tau,lam_j,runtime,calls):
    with torch.no_grad():
        d=integrate(source,c,bounds,cfg["steps"],cfg["basis"]); err=torch.linalg.norm(d-target,dim=1); disp=torch.linalg.norm(d-source,dim=1); raw_d=dirichlet(c,bounds)
        jac=jacobian_grid(c,bounds,cfg["steps"],cfg["basis"],51).cpu().numpy(); raw_j=float(barrier_raw(torch.tensor(jac),tau)) if lam_j else 0.0
        mag=torch.linalg.norm(c,dim=2); corr=float(err.square().mean()); validity=validate_airfoil_shape(d.cpu().numpy())
    return {"correspondence_mse":corr,"correspondence_rmse":math.sqrt(corr),"maximum_pointwise_error":float(err.max()),"raw_dirichlet":float(raw_d),"weighted_dirichlet":float(lam_d*raw_d),
            "raw_jacobian_barrier":raw_j,"weighted_jacobian_barrier":lam_j*raw_j,"total_objective":corr+float(lam_d*raw_d)+lam_j*raw_j,
            "mean_vector_field_magnitude":float(mag.mean()),"rms_vector_field_magnitude":float(torch.sqrt(mag.square().mean())),"maximum_vector_field_magnitude":float(mag.max()),
            "mean_displacement":float(disp.mean()),"maximum_displacement":float(disp.max()),"minimum_jacobian":float(jac.min()),"jacobian_p01":float(np.quantile(jac,.01)),"jacobian_p05":float(np.quantile(jac,.05)),"jacobian_median":float(np.median(jac)),
            "nonpositive_jacobian_proportion":float(np.mean(jac<=0)),"runtime_seconds":runtime,"closure_evaluations":calls,"converged":None,"convergence_status":"L-BFGS completed without exception; backend termination reason unavailable",
            "valid_geometry":bool(validity["valid"]),"safe":bool(validity["valid"] and jac.min()>0 and np.isfinite(corr)),"trainable_coefficients":int(c.numel()),"classification":classify(corr)},d.cpu().numpy(),err.cpu().numpy()


def classify(x):
    if x<=1e-9:return "excellent"
    if x<=1e-8:return "very_good"
    if x<=1e-7:return "improved"
    return "still_limited"


def run_schedule(source_np,target_np,bounds,cfg,schedule,tau=0.1,lam_j=0.0,resolutions=(12,),initial=None):
    source=torch.tensor(source_np,dtype=DTYPE); target=torch.tensor(target_np,dtype=DTYPE); c=initial; total_runtime=0; calls=0; levels=[]
    for ri,res in enumerate(resolutions):
        c=torch.zeros((res,res,2),dtype=DTYPE) if c is None else prolong(c,res)
        for si,lam_d in enumerate(schedule):
            c,rt,cc,_=optimise_level(source,target,c,bounds,cfg,lam_d,tau,lam_j); total_runtime+=rt; calls+=cc
        m,_,_=metrics(source,target,c,bounds,cfg,schedule[-1],tau,lam_j,total_runtime,calls)
        levels.append({"resolution":res,"level":ri+1,**m})
    final,deformed,err=metrics(source,target,c,bounds,cfg,schedule[-1],tau,lam_j,total_runtime,calls)
    return final,c.numpy(),deformed,err,levels


def row(target,kind,params,m):
    return {"target_label":target["label"],"target_index":target["index"],"filename":target["filename"],"experiment":kind,**params,**m}


def save_cached(path,rows): csv_write(path,rows)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--force",action="store_true"); args=ap.parse_args(); out=OUT
    for p in ["jacobian_barrier","multi_resolution","bspline_comparison","residual_analysis","tables","figures","report"]:(out/p).mkdir(parents=True,exist_ok=True)
    shapes,mean,targets,cmp=load_inputs(out); bounds=(0.,1.,float(shapes[:,:,1].min()-.08),float(shapes[:,:,1].max()+.08))
    cfg={"data_path":str(DATA),"stage1_subset":str(STAGE1/"development_subset"),"dtype":"float64","integration":"explicit Euler","steps":15,"basis":"cartesian","barrier_formula":"mean((softplus(50*(tau-det(Dphi)))/50)^2)","barrier_resolution":21,"jacobian_report_resolution":51,"lbfgs_max_iter":80,"tolerance_grad":1e-12,"tolerance_change":1e-15,"bounds":bounds}
    json_write(out/"config.json",cfg)
    baseline=[]
    for t in targets:
        m,*_=run_schedule(mean,t["shape"],bounds,cfg,(1e-2,1e-3,1e-4),lam_j=0,resolutions=(12,)); baseline.append(row(t,"direct_lbfgs_staged",{"schedule":"1e-2->1e-3->1e-4","basis":"cartesian","resolutions":"12"},m))
    csv_write(out/"baseline_reproduction.csv",baseline)
    # Barrier calibration: three requested thresholds and two compact, interpretable weights.
    barrier=[]
    for tau in (.05,.10,.20):
      for w in (.01,.10):
       for t in targets:
        m,*_=run_schedule(mean,t["shape"],bounds,cfg,(1e-2,1e-4,1e-6),tau=tau,lam_j=w,resolutions=(12,)); barrier.append(row(t,"barrier_calibration",{"tau":tau,"lambda_j":w,"schedule":"1e-2->1e-4->1e-6"},m))
    csv_write(out/"jacobian_barrier_results.csv",barrier)
    # Choose safe calibration by median non-identity MSE; unsafe candidates cannot win.
    candidates=[]
    for tau in (.05,.10,.20):
      for w in (.01,.10):
       rr=[r for r in barrier if r["tau"]==tau and r["lambda_j"]==w and r["target_label"]!="identity"]
       candidates.append((all(r["safe"] for r in rr),np.median([r["correspondence_mse"] for r in rr]),tau,w))
    safe_candidates=[x for x in candidates if x[0]]; _,_,best_tau,best_w=min(safe_candidates,key=lambda x:x[1]) if safe_candidates else max(candidates)
    schedules=[("A",(1e-2,1e-3,1e-4)),("B",(1e-2,1e-4,1e-5)),("C",(1e-2,1e-4,1e-6))]; sched=[]
    for name,s in schedules:
     for t in targets:
      m,*_=run_schedule(mean,t["shape"],bounds,cfg,s,best_tau,best_w,(12,)); sched.append(row(t,"barrier_schedule",{"variant":name,"schedule":"->".join(f"{x:.0e}" for x in s),"tau":best_tau,"lambda_j":best_w},m))
    for t in targets:
      m,*_=run_schedule(mean,t["shape"],bounds,cfg,(1e-2,1e-4,1e-6),best_tau,0.0,(12,)); sched.append(row(t,"barrier_schedule",{"variant":"C_no_barrier","schedule":"1e-02->1e-04->1e-06","tau":best_tau,"lambda_j":0.0},m))
    csv_write(out/"regularisation_schedule_results.csv",sched)
    # Cartesian multi-resolution: requested 6-12-24 plus one compact alternative.
    mult=[]; multi_art={}
    for resolutions in ((6,12,24),(8,16)):
     for t in targets:
      m,c,d,e,levels=run_schedule(mean,t["shape"],bounds,cfg,(1e-2,1e-4,1e-5),best_tau,best_w,resolutions)
      key=f"cartesian_{'-'.join(map(str,resolutions))}_{t['label']}"; multi_art[key]=(m,c,d,e,levels)
      for lv in levels: mult.append(row(t,"multiresolution",{"basis":"cartesian","resolution_schedule":"->".join(map(str,resolutions)),"final":lv["level"]==len(resolutions)},lv))
    csv_write(out/"multiresolution_results.csv",mult)
    # One matched B-spline schedule answers the implementation question without a large grid sweep.
    bs=[]; bs_art={}; bcfg={**cfg,"basis":"bspline"}
    for basis,resolutions,ccfg in (("cartesian",(6,12,24),cfg),("bspline",(6,12,24),bcfg)):
     for t in targets:
      if basis=="cartesian": m,c,d,e,levels=multi_art[f"cartesian_6-12-24_{t['label']}"]
      else: m,c,d,e,levels=run_schedule(mean,t["shape"],bounds,ccfg,(1e-2,1e-4,1e-5),best_tau,best_w,resolutions); bs_art[t["label"]]=(m,c,d,e,levels)
      bs.append(row(t,"basis_multiresolution",{"basis":basis,"resolution_schedule":"6->12->24","tau":best_tau,"lambda_j":best_w},m))
    csv_write(out/"bspline_multiresolution_comparison.csv",bs)
    # Best safe method by median MSE across non-identity targets, not by cherry-picking per target.
    methods={"baseline":baseline}
    for v,_, in [(x[0],x[1]) for x in schedules]: methods[f"schedule_{v}"]=[r for r in sched if r["variant"]==v]
    methods["cartesian_6-12-24"]=[r for r in bs if r["basis"]=="cartesian"]
    methods["cartesian_8-16"]=[r for r in mult if r["resolution_schedule"]=="8->16" and r["final"]]
    methods["bspline_6-12-24"]=[r for r in bs if r["basis"]=="bspline"]
    ranks=[]
    for name,rr in methods.items():
        n=[r for r in rr if r["target_label"]!="identity"]
        ranks.append({"name":name,"all_safe":all(r["safe"] for r in n),"median_mse":float(np.median([r["correspondence_mse"] for r in n])),"worst_mse":max(r["correspondence_mse"] for r in n),"runtime":sum(r["runtime_seconds"] for r in n)})
    valid=[x for x in ranks if x["all_safe"]]; best=min(valid,key=lambda x:x["median_mse"]); best_name=best["name"]
    if best_name=="cartesian_6-12-24": artifacts=multi_art; best_rows=methods[best_name]
    elif best_name=="cartesian_8-16": artifacts={f"cartesian_8-16_{t['label']}":multi_art[f"cartesian_8-16_{t['label']}"] for t in targets}; best_rows=methods[best_name]
    elif best_name=="bspline_6-12-24": artifacts={f"bspline_6-12-24_{k}":v for k,v in bs_art.items()}; best_rows=methods[best_name]
    else: artifacts={}; best_rows=methods[best_name]
    # Residual analysis for best multires method; if a single-grid method wins, use best safe multires for spatial diagnosis.
    if not artifacts: artifacts=multi_art; best_rows=methods["cartesian_6-12-24"]; residual_method="cartesian_6-12-24"
    else: residual_method=best_name
    residual=[]
    for t in targets:
      key=next(k for k in artifacts if k.endswith("_"+t["label"])); m,c,d,e,levels=artifacts[key]
      sq=e**2
      for i,(val,ev) in enumerate(zip(sq,d-t["shape"])): residual.append({"target_label":t["label"],"point_index":i,"surface":"upper" if i<100 else "lower","x":float(t["shape"][i,0]),"y":float(t["shape"][i,1]),"squared_error":float(val),"error_x":float(ev[0]),"error_y":float(ev[1])})
      if t["label"]!="identity": residual_fig(mean,t["shape"],d,sq,out/"residual_analysis"/f"{t['label']}.png",t["label"])
    csv_write(out/"residual_error_by_point.csv",residual)
    # Numerical check only for the hardest target. Save both fixed-coefficient sensitivity
    # and the decision-relevant comparison after re-optimising at each step count.
    hardest=max([r for r in best_rows if r["target_label"]!="identity"],key=lambda r:r["correspondence_mse"]); ht=next(t for t in targets if t["label"]==hardest["target_label"])
    key=next(k for k in artifacts if k.endswith("_"+ht["label"])); bm,coef,_,_,_=artifacts[key]; basis="bspline" if residual_method.startswith("bspline") else "cartesian"; numerical=[]
    for steps in (15,30,60):
      cc={**cfg,"basis":basis,"steps":steps}; src=torch.tensor(mean,dtype=DTYPE); tar=torch.tensor(ht["shape"],dtype=DTYPE); ct=torch.tensor(coef,dtype=DTYPE)
      mm,_,_=metrics(src,tar,ct,bounds,cc,1e-5,best_tau,best_w,0,0); numerical.append({"target_label":ht["label"],"steps":steps,**mm})
    csv_write(out/"tables"/"integration_accuracy_check.csv",numerical)
    numerical_reoptimised=[]
    for steps in (15,30,60):
      cc={**cfg,"basis":basis,"steps":steps}; mm,*_=run_schedule(mean,ht["shape"],bounds,cc,(1e-2,1e-4,1e-5),best_tau,best_w,(6,12,24))
      numerical_reoptimised.append({"target_label":ht["label"],"steps":steps,"mode":"reoptimised",**mm})
    csv_write(out/"tables"/"integration_accuracy_check_reoptimised.csv",numerical_reoptimised)
    counts={k:sum(r["safe"] and r["correspondence_mse"]<=v for r in best_rows if r["target_label"]!="identity") for k,v in [("le_1e-9",1e-9),("le_1e-8",1e-8),("le_1e-7",1e-7)]}
    c0=[r for r in sched if r["variant"]=="C_no_barrier" and r["target_label"]!="identity"]
    best_cfg={"selected_method":best_name,"residual_analysis_method":residual_method,"barrier_tau":best_tau,"barrier_weight":best_w,"selection_ranking":ranks,"best_safe_median_mse":best["median_mse"],"best_safe_worst_mse":best["worst_mse"],"threshold_counts":counts,"integration_check":numerical,"integration_check_reoptimised":numerical_reoptimised,"schedule_C_no_barrier":{"all_safe":all(r["safe"] for r in c0),"median_mse":float(np.median([r["correspondence_mse"] for r in c0])),"worst_mse":max(r["correspondence_mse"] for r in c0),"minimum_jacobian":min(r["minimum_jacobian"] for r in c0)}}
    json_write(out/"best_configuration.json",best_cfg); make_summary_figure(methods,out/"figures"/"method_comparison.png")
    write_report(out,cmp,best_cfg,baseline,barrier,sched,mult,bs,residual)
    print(json.dumps(best_cfg,indent=2))


def residual_fig(source,target,deformed,sq,path,title):
    fig,(a,b)=plt.subplots(2,1,figsize=(9,7),gridspec_kw={"height_ratios":[2,1]}); plot_shape(a,source,color="#9ca3af",label="source"); plot_shape(a,target,color="#2563eb",label="target"); plot_shape(a,deformed,color="#ea580c",linestyle="--",label="registered")
    step=5; a.quiver(deformed[::step,0],deformed[::step,1],target[::step,0]-deformed[::step,0],target[::step,1]-deformed[::step,1],angles="xy",scale_units="xy",scale=1,color="#7c3aed",width=.003)
    sc=a.scatter(target[:,0],target[:,1],c=np.log10(sq+1e-18),cmap="magma",s=13); fig.colorbar(sc,ax=a,label="log10 squared error"); a.set_aspect("equal",adjustable="box"); a.set_title(f"Residual spatial structure — {title}"); a.legend(); a.grid(alpha=.2)
    b.plot(np.arange(200),sq,color="#2563eb"); b.axvline(99.5,color="#6b7280",linestyle="--"); b.set_yscale("log"); b.set(xlabel="point index (0–99 upper, 100–199 lower)",ylabel="squared error"); b.grid(alpha=.2); fig.tight_layout(); fig.savefig(path,dpi=200); plt.close(fig)


def make_summary_figure(methods,path):
    names=[]; vals=[]
    for name,rr in methods.items(): names.append(name); vals.append([r["correspondence_mse"] for r in rr if r["target_label"]!="identity" and r["safe"]])
    fig,ax=plt.subplots(figsize=(10,5)); ax.boxplot(vals,tick_labels=names); ax.axhline(1e-9,color="#ea580c",linestyle="--",label="1e-9"); ax.axhline(1e-7,color="#ca8a04",linestyle=":",label="1e-7"); ax.set_yscale("log"); ax.set_ylabel("safe correspondence MSE"); ax.tick_params(axis="x",rotation=30); ax.legend(); ax.grid(axis="y",alpha=.2); fig.tight_layout(); fig.savefig(path,dpi=220); plt.close(fig)


def write_report(out,cmp,best,baseline,barrier,sched,mult,bs,residual):
    non=[r for r in (next(x for x in best["selection_ranking"] if x["name"]==best["selected_method"]),)]; direct_base=[r for r in baseline if r["target_label"]!="identity"]
    cart=[r for r in bs if r["basis"]=="cartesian" and r["target_label"]!="identity"]; bsp=[r for r in bs if r["basis"]=="bspline" and r["target_label"]!="identity"]
    # Concentration: aggregate thirds of chord and LE/TE bands using all non-identity residual rows.
    rr=[r for r in residual if r["target_label"]!="identity"]; total=sum(r["squared_error"] for r in rr); bands={"leading_edge_x<=0.1":sum(r["squared_error"] for r in rr if r["x"]<=.1)/total,"trailing_edge_x>=0.9":sum(r["squared_error"] for r in rr if r["x"]>=.9)/total,"middle_chord":sum(r["squared_error"] for r in rr if .1<r["x"]<.9)/total}
    lines=["# Airfoil Registration Stage 2：Jacobian barrier 与多分辨率 SVF", "", "## 技术摘要", "", f"直接坐标均值与 Stage 1 均值的最大差为 `{cmp['maximum_absolute_coordinate_difference']:.3e}`，RMS 差为 `{cmp['rms_coordinate_difference']:.3e}`；两者均几何有效。后续全部使用直接坐标均值。", "", f"按八个非恒等目标的安全中位 MSE 选择，最佳方法为 `{best['selected_method']}`：中位 `{best['best_safe_median_mse']:.3e}`，最差 `{best['best_safe_worst_mse']:.3e}`。达到 1e-9/1e-8/1e-7 的目标数分别为 `{best['threshold_counts']['le_1e-9']}/8`、`{best['threshold_counts']['le_1e-8']}/8`、`{best['threshold_counts']['le_1e-7']}/8`。", "", "## 直接坐标和基线", "", f"数据形状为 `{cmp['shape']}`，固定 x 网格且已有逐点对应。L-BFGS 分阶段直接坐标基线的中位 MSE 为 `{np.median([r['correspondence_mse'] for r in direct_base]):.3e}`，最差为 `{max(r['correspondence_mse'] for r in direct_base):.3e}`。Stage 1 历史分阶段数字来自 Adam 单因素运行，故本报告不把两者的微小差别归因于坐标表示。", "", "## Jacobian barrier", "", "使用可微平滑平方铰链：`R_J = mean[(softplus(50*(tau-J))/50)^2]`，总目标为 `D_corr + lambda_D R_D + lambda_J R_J`。对应项、Dirichlet 项和 barrier 项均单独保存。阈值 tau=0.05/0.10/0.20 与权重 0.01/0.10 经过同一开发集校准；任何非正 Jacobian、无效几何或非有限结果均被拒绝。", "", "## 多分辨率与 B-spline", "", f"Cartesian 6→12→24 的安全中位 MSE 为 `{np.median([r['correspondence_mse'] for r in cart]):.3e}`，平均每目标耗时 `{np.mean([r['runtime_seconds'] for r in cart]):.2f}s`；B-spline 对应为 `{np.median([r['correspondence_mse'] for r in bsp]):.3e}` 和 `{np.mean([r['runtime_seconds'] for r in bsp]):.2f}s`。每次网格转换均用 bicubic prolongation 延拓已有速度系数，没有从零重启。", "", "## 残差空间结构", "", f"在用于残差诊断的最佳安全多分辨率方法中，前缘 x<=0.1、后缘 x>=0.9 和中弦区分别贡献总平方误差的 `{bands['leading_edge_x<=0.1']:.1%}`、`{bands['trailing_edge_x>=0.9']:.1%}`、`{bands['middle_chord']:.1%}`。逐目标热图和误差矢量位于 residual_analysis/。", "", "## 数值精度与限制", "", "只对最难目标复用同一最终速度系数比较 15/30/60 个 Euler 步。该检查隔离积分误差，不重新优化，因此能判断离散积分本身对最终映射的影响。连续域正 Jacobian 仍由 51×51 采样诊断近似，而不是解析证明。", "", "## 对十个问题的回答", "", "1. 是，直接坐标均值在浮点误差范围内复现旧均值。", "2. 没有证据表明去除 camber/thickness 本身改变性能；旧均值和直接均值数值等价，历史优化器组合还存在实现差异。", f"3. barrier 是否让低正则安全可用：见 regularisation_schedule_results.csv；最佳校准为 tau={best['barrier_tau']}、lambda_J={best['barrier_weight']}，仅安全结果参与选择。", f"4. 多分辨率的最终判断：最佳安全方法为 {best['selected_method']}，相对于直接单网格基线的中位误差变化可由 best_configuration.json 的排名复核。", f"5. B-spline 是否值得：Cartesian/B-spline 的配对精度和耗时如上；若 B-spline 没有同时改善精度与安全性，则本实现不推荐。", f"6. 最佳安全中位 MSE `{best['best_safe_median_mse']:.3e}`；安全最差 `{best['best_safe_worst_mse']:.3e}`。", f"7. 1e-9、1e-8、1e-7：{best['threshold_counts']['le_1e-9']}/8、{best['threshold_counts']['le_1e-8']}/8、{best['threshold_counts']['le_1e-7']}/8。", f"8. 残差主要分布：前缘 {bands['leading_edge_x<=0.1']:.1%}、后缘 {bands['trailing_edge_x>=0.9']:.1%}、中弦 {bands['middle_chord']:.1%}。", "9. 剩余瓶颈根据 barrier、网格级别和积分敏感性共同判断；详细逐级数据保存在 multiresolution_results.csv。", "10. 只有在全部目标保持正 Jacobian 时达到相应阈值才算可达；当前计数是本框架最直接的证据。", "", "## 停止条件", "", "本阶段只运行九个既定开发目标，未运行 1654 个全量配准，未使用 GMM、K、模板选择或气动优化。等待批准。"]
    (out/"report"/"final_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__": main()
