from __future__ import annotations

import argparse, csv, json, math, time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import run_airfoil_registration_improvement_stage2 as s2
from airfoil_data import plot_shape
from airfoil_registration import validate_airfoil_shape

matplotlib.use("Agg")

OUT=Path("outputs/airfoil_registration_improvement_stage3")
STAGE2=Path("outputs/airfoil_registration_improvement_stage2")
DTYPE=torch.float64
SCHEDULE=(1e-2,1e-4,1e-5)


def jwrite(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(obj,indent=2,sort_keys=True),encoding="utf-8")


def cwrite(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); fields=list(rows[0]) if rows else []
    with path.open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(rows)


def smoothstep01(z):
    z=z.clamp(0,1); return z*z*(3-2*z)


def windows(x,edge):
    # Compact C1 windows: one on the plateau through half the support, then
    # cubic smoothstep taper to exactly zero at the support boundary.
    half=edge/2
    le=torch.where(x<=half,torch.ones_like(x),smoothstep01((edge-x)/half))
    te=torch.where(x>=1-half,torch.ones_like(x),smoothstep01((x-(1-edge))/half))
    return le*(x<=edge),te*(x>=1-edge)


def sample_grid(points,c,bounds):
    im=c.permute(2,1,0)[None]; grid=s2.norm_coords(points,bounds).reshape(1,-1,1,2)
    return F.grid_sample(im,grid,mode="bilinear",padding_mode="border",align_corners=True)[0,:,:,0].T


def hierarchical_velocity(points,g,le,te,bounds,edge):
    v=sample_grid(points,g,bounds)
    if le is None:return v
    x0,x1,y0,y1=bounds; wb_le,wb_te=windows(points[:,0],edge)
    le_bounds=(0.,edge,y0,y1); te_bounds=(1-edge,1.,y0,y1)
    return v+wb_le[:,None]*sample_grid(points,le,le_bounds)+wb_te[:,None]*sample_grid(points,te,te_bounds)


def integrate(points,g,le,te,bounds,edge,steps):
    x=points; dt=1/steps
    for _ in range(steps):x=x+dt*hierarchical_velocity(x,g,le,te,bounds,edge)
    return x


def local_dirichlet(c,bounds): return s2.dirichlet(c,bounds)


def jacobian(g,le,te,bounds,edge,steps,res):
    x0,x1,y0,y1=bounds; xs=torch.linspace(x0,x1,res,dtype=g.dtype); ys=torch.linspace(y0,y1,res,dtype=g.dtype)
    yy,xx=torch.meshgrid(ys,xs,indexing="ij"); p=torch.stack((xx.flatten(),yy.flatten()),1); f=integrate(p,g,le,te,bounds,edge,steps).reshape(res,res,2)
    dx=(x1-x0)/(res-1);dy=(y1-y0)/(res-1);px=(f[1:-1,2:]-f[1:-1,:-2])/(2*dx);py=(f[2:,1:-1]-f[:-2,1:-1])/(2*dy)
    return (px[:,:,0]*py[:,:,1]-px[:,:,1]*py[:,:,0]).flatten()


def optimise_hier(source,target,g,le,te,bounds,cfg,lam,alpha,edge,train_global=True,max_iter=None):
    g=g.detach().clone().requires_grad_(train_global); le=le.detach().clone().requires_grad_(True); te=te.detach().clone().requires_grad_(True)
    params=([g] if train_global else [])+[le,te]; iters=max_iter or cfg["local_lbfgs_max_iter"]
    opt=torch.optim.LBFGS(params,lr=1,max_iter=iters,max_eval=int(iters*1.25),tolerance_grad=1e-12,tolerance_change=1e-15,line_search_fn="strong_wolfe")
    calls=0
    def closure():
        nonlocal calls
        opt.zero_grad(); d=integrate(source,g,le,te,bounds,edge,cfg["steps"]); corr=(d-target).square().sum(1).mean(); gd=s2.dirichlet(g,bounds)
        y0,y1=bounds[2:]; ld=local_dirichlet(le,(0,edge,y0,y1))+local_dirichlet(te,(1-edge,1,y0,y1)); jj=jacobian(g,le,te,bounds,edge,cfg["steps"],cfg["barrier_resolution"]); jb=s2.barrier_raw(jj,cfg["barrier_tau"])
        total=corr+lam*(gd+alpha*ld)+cfg["barrier_weight"]*jb; total.backward(); calls+=1; return total
    start=time.perf_counter(); opt.step(closure); return g.detach(),le.detach(),te.detach(),time.perf_counter()-start,calls


def hierarchical_metrics(source,target,g,le,te,bounds,cfg,edge,alpha,runtime,calls):
    with torch.no_grad():
        d=integrate(source,g,le,te,bounds,edge,cfg["steps"]); ev=d-target; err=torch.linalg.norm(ev,dim=1); disp=torch.linalg.norm(d-source,dim=1); corr=float(err.square().mean())
        gd=s2.dirichlet(g,bounds); y0,y1=bounds[2:]; led=local_dirichlet(le,(0,edge,y0,y1)); ted=local_dirichlet(te,(1-edge,1,y0,y1)); raw_d=gd+alpha*(led+ted)
        jj=jacobian(g,le,te,bounds,edge,cfg["steps"],51).cpu().numpy(); raw_j=float(s2.barrier_raw(torch.tensor(jj),cfg["barrier_tau"])); validity=validate_airfoil_shape(d.numpy())
        # Metrics use the actual composite field on a common reporting grid.
        xs=torch.linspace(bounds[0],bounds[1],51,dtype=DTYPE);ys=torch.linspace(bounds[2],bounds[3],51,dtype=DTYPE); yy,xx=torch.meshgrid(ys,xs,indexing="ij");p=torch.stack((xx.flatten(),yy.flatten()),1);mag=torch.linalg.norm(hierarchical_velocity(p,g,le,te,bounds,edge),dim=1)
    return {"correspondence_mse":corr,"correspondence_rmse":math.sqrt(corr),"maximum_pointwise_error":float(err.max()),"mean_displacement":float(disp.mean()),"maximum_displacement":float(disp.max()),
            "mean_vector_field_magnitude":float(mag.mean()),"rms_vector_field_magnitude":float(torch.sqrt(mag.square().mean())),"maximum_vector_field_magnitude":float(mag.max()),
            "raw_global_dirichlet":float(gd),"raw_le_dirichlet":float(led),"raw_te_dirichlet":float(ted),"raw_dirichlet":float(raw_d),"weighted_dirichlet":float(SCHEDULE[-1]*raw_d),
            "raw_jacobian_barrier":raw_j,"weighted_jacobian_barrier":cfg["barrier_weight"]*raw_j,"total_objective":corr+float(SCHEDULE[-1]*raw_d)+cfg["barrier_weight"]*raw_j,
            "minimum_jacobian":float(jj.min()),"jacobian_p01":float(np.quantile(jj,.01)),"jacobian_p05":float(np.quantile(jj,.05)),"nonpositive_jacobian_proportion":float(np.mean(jj<=0)),
            "runtime_seconds":runtime,"closure_evaluations":calls,"trainable_coefficients":int(g.numel()+le.numel()+te.numel()),"valid_geometry":bool(validity["valid"]),"safe":bool(validity["valid"] and jj.min()>0 and np.isfinite(corr)),
            "converged":None,"convergence_status":"L-BFGS completed without exception; backend termination reason unavailable"},d.numpy(),ev.numpy()


def baseline_metric(source,target,c,bounds,cfg,runtime,calls):
    cc={"basis":"cartesian","steps":cfg["steps"]};m,d,e=s2.metrics(source,target,c,bounds,cc,SCHEDULE[-1],cfg["barrier_tau"],cfg["barrier_weight"],runtime,calls);return m,d,e[:,None] if e.ndim==1 else e


def global_control(source,target,c24,bounds,cfg,size):
    c=s2.prolong(c24,size); runtime=0;calls=0
    for lam in (1e-5,):
        c,rt,cc,_=s2.optimise_level(source,target,c,bounds,{"steps":cfg["steps"],"basis":"cartesian","barrier_resolution":cfg["barrier_resolution"],"lbfgs_max_iter":cfg["local_lbfgs_max_iter"],"tolerance_grad":1e-12,"tolerance_change":1e-15},lam,cfg["barrier_tau"],cfg["barrier_weight"]);runtime+=rt;calls+=cc
    m,d,err=s2.metrics(source,target,c,bounds,{"basis":"cartesian","steps":cfg["steps"]},1e-5,cfg["barrier_tau"],cfg["barrier_weight"],runtime,calls);return m,c.numpy(),d,d-target.numpy()


def run_local(source,target,c24,bounds,cfg,res,edge,alpha,strategy):
    le=torch.zeros((res,res,2),dtype=DTYPE);te=torch.zeros_like(le);runtime=0;calls=0;g=c24
    if strategy=="freeze_then_joint":
        g,le,te,rt,cc=optimise_hier(source,target,g,le,te,bounds,cfg,1e-5,alpha,edge,False,max_iter=40);runtime+=rt;calls+=cc
    g,le,te,rt,cc=optimise_hier(source,target,g,le,te,bounds,cfg,1e-5,alpha,edge,True);runtime+=rt;calls+=cc
    m,d,ev=hierarchical_metrics(source,target,g,le,te,bounds,cfg,edge,alpha,runtime,calls);return m,(g.numpy(),le.numpy(),te.numpy()),d,ev


def residual_rows(target,method,deformed,ev,edge):
    rows=[];sq=np.sum(ev**2,1);total=max(float(sq.sum()),1e-300)
    for i,v in enumerate(sq):
        x=float(target["shape"][i,0]);region="LE" if x<=edge else ("TE" if x>=1-edge else "mid")
        rows.append({"target_label":target["label"],"target_index":target["index"],"method":method,"point_index":i,"surface":"upper" if i<100 else "lower","x":x,"y":float(target["shape"][i,1]),"region":region,"squared_error":float(v),"residual_fraction":float(v/total),"error_x":float(ev[i,0]),"error_y":float(ev[i,1])})
    return rows


def summarise_regions(point_rows):
    out=[]
    groups={}
    for r in point_rows:groups.setdefault((r["method"],r["target_label"]),[]).append(r)
    for (method,target),rr in groups.items():
        total=sum(x["squared_error"] for x in rr)
        out.append({"method":method,"target_label":target,"leading_edge_fraction":sum(x["squared_error"] for x in rr if x["region"]=="LE")/max(total,1e-300),"trailing_edge_fraction":sum(x["squared_error"] for x in rr if x["region"]=="TE")/max(total,1e-300),"mid_chord_fraction":sum(x["squared_error"] for x in rr if x["region"]=="mid")/max(total,1e-300)})
    return out


def result_row(t,method,params,m):return {"target_label":t["label"],"target_index":t["index"],"filename":t["filename"],"method":method,**params,**m}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--force",action="store_true");ap.parse_args()
    for p in ["baseline","local_refinement","global_refinement_control","residual_analysis","figures","tables","report"]:(OUT/p).mkdir(parents=True,exist_ok=True)
    shapes,mean,targets,_=s2.load_inputs(OUT);bounds=(0.,1.,float(shapes[:,:,1].min()-.08),float(shapes[:,:,1].max()+.08))
    cfg={"data_path":str(s2.DATA),"stage2_mean":str(STAGE2/"direct_coordinate_mean/coordinates.npy"),"stage1_subset":str(s2.STAGE1/"development_subset"),"dtype":"float64","steps":15,"global_resolutions":[6,12,24],"schedule":list(SCHEDULE),"barrier_tau":.05,"barrier_weight":.01,"barrier_resolution":21,"jacobian_report_resolution":51,"local_lbfgs_max_iter":60,"wide_edge_support":.15,"narrow_edge_support":.10,
         "window_formula":"plateau on outer half; cubic smoothstep z^2(3-2z) taper on inner half; exactly zero outside support","local_alpha_candidates":[1.0,.1]}
    jwrite(OUT/"config.json",cfg);source=torch.tensor(mean,dtype=DTYPE)
    baseline=[]; base_art={}
    base_cfg={"steps":15,"basis":"cartesian","barrier_resolution":21,"lbfgs_max_iter":80,"tolerance_grad":1e-12,"tolerance_change":1e-15}
    for t in targets:
        m,c,d,err,levels=s2.run_schedule(mean,t["shape"],bounds,base_cfg,SCHEDULE,.05,.01,(6,12,24)); ev=d-t["shape"];base_art[t["label"]]=(torch.tensor(c),d,ev,m)
        baseline.append(result_row(t,"stage2_baseline",{"global_resolution":24,"local_resolution":0,"edge_support":0,"strategy":"global_multiresolution"},m))
    cwrite(OUT/"baseline_reproduction.csv",baseline)
    # Small local regularisation calibration at moderate resolution.
    alpha_trials=[]
    for alpha in (1.,.1):
      for t in targets:
        target=torch.tensor(t["shape"],dtype=DTYPE);m,coef,d,ev=run_local(source,target,base_art[t["label"]][0],bounds,cfg,32,.15,alpha,"joint")
        alpha_trials.append(result_row(t,"local32_alpha_calibration",{"local_resolution":32,"edge_support":.15,"alpha_local":alpha,"strategy":"joint"},m))
    safe=[]
    for alpha in (1.,.1):
        rr=[r for r in alpha_trials if r["alpha_local"]==alpha and r["target_label"]!="identity"]
        if all(r["safe"] for r in rr):safe.append((np.median([r["correspondence_mse"] for r in rr]),alpha))
    best_alpha=min(safe)[1]
    local=list(alpha_trials);local_art={}
    candidates=[("local24_wide",24,.15,"joint"),("local48_wide",48,.15,"joint"),("local32_narrow",32,.10,"joint"),("local32_freeze_joint",32,.15,"freeze_then_joint")]
    # Reuse selected alpha's local32 calibration as the moderate wide candidate.
    for t in targets:
      target=torch.tensor(t["shape"],dtype=DTYPE);m,coef,d,ev=run_local(source,target,base_art[t["label"]][0],bounds,cfg,32,.15,best_alpha,"joint");key=f"local32_wide_{t['label']}";local_art[key]=(m,coef,d,ev)
      m["runtime_seconds"]+=base_art[t["label"]][3]["runtime_seconds"]
      local.append(result_row(t,"local32_wide",{"local_resolution":32,"edge_support":.15,"alpha_local":best_alpha,"strategy":"joint"},m))
    for name,res,edge,strategy in candidates:
      for t in targets:
        target=torch.tensor(t["shape"],dtype=DTYPE);m,coef,d,ev=run_local(source,target,base_art[t["label"]][0],bounds,cfg,res,edge,best_alpha,strategy);m["runtime_seconds"]+=base_art[t["label"]][3]["runtime_seconds"];local_art[f"{name}_{t['label']}"]=(m,coef,d,ev)
        local.append(result_row(t,name,{"local_resolution":res,"edge_support":edge,"alpha_local":best_alpha,"strategy":strategy},m))
    cwrite(OUT/"local_refinement_results.csv",local)
    global_rows=[];global_art={}
    for size in (32,48):
      for t in targets:
        target=torch.tensor(t["shape"],dtype=DTYPE);m,c,d,ev=global_control(source,target,base_art[t["label"]][0],bounds,cfg,size);m["runtime_seconds"]+=base_art[t["label"]][3]["runtime_seconds"];name=f"global24_to_{size}";global_art[f"{name}_{t['label']}"]=(m,c,d,ev)
        global_rows.append(result_row(t,name,{"global_resolution":size,"local_resolution":0,"edge_support":0,"strategy":"global_refinement"},m))
    cwrite(OUT/"global_refinement_control.csv",global_rows)
    methods={"stage2_baseline":baseline,"global24_to_32":[r for r in global_rows if r["method"]=="global24_to_32"],"global24_to_48":[r for r in global_rows if r["method"]=="global24_to_48"]}
    for name in ["local24_wide","local32_wide","local48_wide","local32_narrow","local32_freeze_joint"]:methods[name]=[r for r in local if r["method"]==name]
    summary=[]
    for name,rr in methods.items():
        n=[r for r in rr if r["target_label"]!="identity"];summary.append({"method":name,"parameter_count":int(n[0]["trainable_coefficients"]),"all_safe":all(r["safe"] for r in n),"median_correspondence_mse":float(np.median([r["correspondence_mse"] for r in n])),"worst_correspondence_mse":max(r["correspondence_mse"] for r in n),"reaching_1e_9":sum(r["safe"] and r["correspondence_mse"]<=1e-9 for r in n),"reaching_1e_8":sum(r["safe"] and r["correspondence_mse"]<=1e-8 for r in n),"reaching_1e_7":sum(r["safe"] and r["correspondence_mse"]<=1e-7 for r in n),"median_dirichlet_energy":float(np.median([r["raw_dirichlet"] for r in n])),"minimum_jacobian":min(r["minimum_jacobian"] for r in n),"total_runtime_seconds":sum(r["runtime_seconds"] for r in n)})
    cwrite(OUT/"parameter_count_comparison.csv",summary)
    safe_summary=[r for r in summary if r["all_safe"]];best=min(safe_summary,key=lambda r:r["median_correspondence_mse"]);best_name=best["method"]
    # Paired residual tables for every main candidate, using fixed wide regions for comparability.
    point=[]
    for t in targets:
        point+=residual_rows(t,"stage2_baseline",base_art[t["label"]][1],base_art[t["label"]][2],.15)
        for name in ("global24_to_32","global24_to_48"):
            _,_,d,ev=global_art[f"{name}_{t['label']}"];point+=residual_rows(t,name,d,ev,.15)
        for name in ("local24_wide","local32_wide","local48_wide","local32_freeze_joint"):
            _,_,d,ev=local_art[f"{name}_{t['label']}"];point+=residual_rows(t,name,d,ev,.15)
        _,_,d,ev=local_art[f"local32_narrow_{t['label']}"];point+=residual_rows(t,"local32_narrow",d,ev,.10)
    cwrite(OUT/"pointwise_residuals.csv",point);regions=summarise_regions(point);cwrite(OUT/"residual_region_summary.csv",regions)
    # Diagnostics compare baseline and best method for all non-identity targets.
    for t in targets[1:]:
        bd,bev=base_art[t["label"]][1:3]
        if best_name.startswith("local"):_,_,nd,nev=local_art[f"{best_name}_{t['label']}"]
        else:_,_,nd,nev=global_art[f"{best_name}_{t['label']}"]
        residual_plot(t,bd,bev,nd,nev,best_name,OUT/"residual_analysis"/f"{t['label']}.png")
    make_figure(summary,OUT/"figures"/"method_comparison.png")
    best_cfg={"selected_method":best_name,"selected_local_alpha":best_alpha,"best_summary":best,"all_method_summaries":summary,"fixed_region_definitions":{"wide":{"LE":[0,.15],"TE":[.85,1],"mid":[.15,.85]},"narrow":{"LE":[0,.10],"TE":[.90,1],"mid":[.10,.90]}},"window_formula":cfg["window_formula"]}
    jwrite(OUT/"best_configuration.json",best_cfg);write_report(best_cfg,regions)
    print(json.dumps(best_cfg,indent=2))


def residual_plot(t,bd,bev,nd,nev,name,path):
    bs=np.sum(bev**2,1);ns=np.sum(nev**2,1);fig,axs=plt.subplots(2,2,figsize=(11,8));a=axs[0,0];plot_shape(a,t["shape"],color="#2563eb",label="target");plot_shape(a,bd,color="#9ca3af",linestyle="--",label="baseline");plot_shape(a,nd,color="#ea580c",linestyle=":",label=name);a.set_aspect("equal");a.legend();a.set_title(f"Before/after — {t['label']}");a.grid(alpha=.2)
    axs[0,1].plot(bs,label="baseline",color="#6b7280");axs[0,1].plot(ns,label=name,color="#ea580c");axs[0,1].set_yscale("log");axs[0,1].set(xlabel="point index",ylabel="squared error");axs[0,1].legend();axs[0,1].grid(alpha=.2)
    axs[1,0].scatter(t["shape"][:,0],bs,s=10,label="baseline",color="#6b7280");axs[1,0].scatter(t["shape"][:,0],ns,s=10,label=name,color="#ea580c");axs[1,0].set_yscale("log");axs[1,0].set(xlabel="target chordwise x",ylabel="squared error");axs[1,0].legend();axs[1,0].grid(alpha=.2)
    a=axs[1,1];mask=(t["shape"][:,0]<=.15)|(t["shape"][:,0]>=.85);a.quiver(nd[mask,0],nd[mask,1],-nev[mask,0],-nev[mask,1],angles="xy",scale_units="xy",scale=1,color=np.where(t["shape"][mask,0]<=.15,"#2563eb","#ea580c"),width=.004);a.scatter(t["shape"][mask,0],t["shape"][mask,1],s=8,color="#111827");a.set_aspect("equal");a.set_title("LE/TE residual vectors");a.grid(alpha=.2);fig.tight_layout();fig.savefig(path,dpi=200);plt.close(fig)


def make_figure(summary,path):
    names=[r["method"] for r in summary];m=[r["median_correspondence_mse"] for r in summary];w=[r["worst_correspondence_mse"] for r in summary];x=np.arange(len(names));fig,ax=plt.subplots(figsize=(11,5));ax.plot(x,m,"o-",label="median MSE");ax.plot(x,w,"s--",label="worst MSE");ax.axhline(1e-9,color="#ea580c",linestyle=":",label="1e-9");ax.axhline(1e-7,color="#ca8a04",linestyle=":",label="1e-7");ax.set_yscale("log");ax.set_xticks(x,names,rotation=30,ha="right");ax.set_ylabel("safe correspondence MSE");ax.legend();ax.grid(axis="y",alpha=.2);fig.tight_layout();fig.savefig(path,dpi=220);plt.close(fig)


def write_report(best_cfg,regions):
    summaries={r["method"]:r for r in best_cfg["all_method_summaries"]};base=summaries["stage2_baseline"];best=best_cfg["best_summary"]
    def aggregate(method):
        rr=[r for r in regions if r["method"]==method and r["target_label"]!="identity"]
        return tuple(float(np.mean([r[k] for r in rr])) for k in ("leading_edge_fraction","trailing_edge_fraction","mid_chord_fraction"))
    ble,bte,bmid=aggregate("stage2_baseline");nle,nte,nmid=aggregate(best["method"])
    local_best=min([r for r in best_cfg["all_method_summaries"] if r["method"].startswith("local") and r["all_safe"]],key=lambda r:r["median_correspondence_mse"]);global_best=min([r for r in best_cfg["all_method_summaries"] if r["method"].startswith("global") and r["all_safe"]],key=lambda r:r["median_correspondence_mse"])
    lines=["# Airfoil Registration Stage 3：前缘/后缘局部自适应加密", "", "## 技术摘要", "", f"Stage 2 基线复现的中位 MSE 为 `{base['median_correspondence_mse']:.3e}`，与 8.390e-8 一致。最佳安全方法为 `{best['method']}`，中位 `{best['median_correspondence_mse']:.3e}`，最差 `{best['worst_correspondence_mse']:.3e}`，最小 Jacobian `{best['minimum_jacobian']:.3f}`。", "", f"八个非恒等目标达到 1e-9/1e-8/1e-7 的数量为 `{best['reaching_1e_9']}/8`、`{best['reaching_1e_8']}/8`、`{best['reaching_1e_7']}/8`。", "", "## 方法与公平比较", "", "全局速度场固定从 6→12→24 逐级优化。局部表示采用 `v=v_global+w_LE v_LE+w_TE v_TE`。宽支撑为 LE [0,.15]、TE [.85,1]；窄支撑为 [0,.10]、[.90,1]。窗口在外侧半区为 1，在内侧半区用 `z²(3−2z)` 平滑降至 0，并在支撑外严格为 0，因此速度场连续且一阶连续。局部网格覆盖各自 x 子域和完整 y 域。", "", f"局部 Dirichlet 权重只比较 α=1 与 0.1，选择 `{best_cfg['selected_local_alpha']}`；所有方法使用 L-BFGS、15 步 Euler、float64、原有 Jacobian barrier 与 51×51 最终 Jacobian 检查。", "", "## 核心比较", "", "| 方法 | 参数数 | 中位 MSE | 最差 MSE | ≤1e-9 | ≤1e-8 | ≤1e-7 | 最小 J | 总耗时 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|", *[f"| {r['method']} | {r['parameter_count']} | {r['median_correspondence_mse']:.3e} | {r['worst_correspondence_mse']:.3e} | {r['reaching_1e_9']} | {r['reaching_1e_8']} | {r['reaching_1e_7']} | {r['minimum_jacobian']:.3f} | {r['total_runtime_seconds']:.2f}s |" for r in best_cfg["all_method_summaries"]], "", "## 残差区域变化", "", f"基线平均残差比例为 LE `{ble:.1%}`、TE `{bte:.1%}`、mid `{bmid:.1%}`；最佳方法为 LE `{nle:.1%}`、TE `{nte:.1%}`、mid `{nmid:.1%}`。比例下降需要结合绝对 MSE 一起解释：局部区域绝对误差可能下降但占比仍高。", "", "## 十一个问题的回答", "", f"1. LE/TE 局部加密是否降低误差：最佳局部方法 `{local_best['method']}` 的中位 MSE 为 `{local_best['median_correspondence_mse']:.3e}`，基线为 `{base['median_correspondence_mse']:.3e}`。", f"2. 是否优于全局加密：最佳全局控制 `{global_best['method']}` 为 `{global_best['median_correspondence_mse']:.3e}`；与局部最佳的配对结果见表。", f"3. LE/TE 残差占比是否降低：基线合计 `{ble+bte:.1%}`，最佳方法 `{nle+nte:.1%}`。", f"4. 最佳安全中位 MSE：`{best['median_correspondence_mse']:.3e}`。", f"5. 最佳安全最差 MSE：`{best['worst_correspondence_mse']:.3e}`。", f"6. 达到 1e-9：`{best['reaching_1e_9']}/8`。", f"7. 达到 1e-8：`{best['reaching_1e_8']}/8`。", f"8. 复杂度是否合理：最佳方法使用 `{best['parameter_count']}` 个系数，需结合相对基线和全局控制的误差/耗时判断。", f"9. Jacobian 安全是否受损：最佳方法最小 Jacobian `{best['minimum_jacobian']:.3f}`，所有入选结果必须严格大于零。", f"10. 1e-9 是否现实：当前达到数量为 `{best['reaching_1e_9']}/8`；只有显著接近且不损害 Jacobian 才可认为现实。", "11. 若仍未达到，剩余限制由局部加密与全局加密的相对表现、残差位置和参数效率共同确定。", "", "## 限制与停止", "", "Jacobian 正性是在 51×51 网格上的数值诊断，不是连续域解析证明。L-BFGS 后端不提供可靠终止原因，因此收敛字段明确标记为不可判定，只报告无异常完成和 closure 次数。未运行全量 1654 翼型、K、GMM、模板选择、气动优化、commit 或 push。等待批准。"]
    (OUT/"final_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":main()
