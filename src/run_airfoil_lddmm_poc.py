from __future__ import annotations

import csv,json,math,time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

import run_airfoil_registration_improvement_stage2 as s2
import run_airfoil_registration_improvement_stage3 as s3
from airfoil_data import plot_shape
from airfoil_registration import validate_airfoil_shape
from lddmm_2d import (discrete_geodesic_energy, integrate_time_dependent,
                      jacobian_determinants, kinetic_l2_diagnostic,
                      smooth_jacobian_barrier)

matplotlib.use("Agg")
OUT=Path("outputs/airfoil_lddmm_poc");DTYPE=torch.float64;SCHEDULE=(1e-2,1e-4,1e-5)


def jwrite(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,sort_keys=True),encoding="utf-8")


def cwrite(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fields=list(rows[0]) if rows else []
    with path.open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)


def optimise(source,target,fields,bounds,cfg,schedule=SCHEDULE):
    fields=fields.detach().clone().requires_grad_(True);runtime=0;calls=0
    for lam in schedule:
        opt=torch.optim.LBFGS([fields],lr=1,max_iter=cfg["lbfgs_max_iter"],max_eval=int(cfg["lbfgs_max_iter"]*1.25),tolerance_grad=1e-12,tolerance_change=1e-15,line_search_fn="strong_wolfe")
        def closure():
            nonlocal calls
            opt.zero_grad();d=integrate_time_dependent(source,fields,bounds,cfg["substeps_per_slice"]);corr=(d-target).square().sum(1).mean();eg=discrete_geodesic_energy(fields,bounds);jac=jacobian_determinants(fields,bounds,cfg["substeps_per_slice"],cfg["barrier_resolution"]);jb=smooth_jacobian_barrier(jac,cfg["barrier_tau"]);total=corr+lam*eg+cfg["barrier_weight"]*jb;total.backward();calls+=1;return total
        start=time.perf_counter();opt.step(closure);runtime+=time.perf_counter()-start
    return fields.detach(),runtime,calls


def metrics(source,target,fields,bounds,cfg,runtime,calls):
    with torch.no_grad():
        d=integrate_time_dependent(source,fields,bounds,cfg["substeps_per_slice"]);ev=d-target;err=torch.linalg.norm(ev,dim=1);disp=torch.linalg.norm(d-source,dim=1);corr=float(err.square().mean());eg=discrete_geodesic_energy(fields,bounds);kin=kinetic_l2_diagnostic(fields);jac=jacobian_determinants(fields,bounds,cfg["substeps_per_slice"],51).cpu().numpy();jb=float(smooth_jacobian_barrier(torch.tensor(jac),cfg["barrier_tau"]));mag=torch.linalg.norm(fields,dim=3);valid=validate_airfoil_shape(d.numpy())
    return {"correspondence_mse":corr,"correspondence_rmse":math.sqrt(corr),"maximum_pointwise_error":float(err.max()),"raw_geodesic_smoothness_energy":float(eg),"weighted_geodesic_energy":float(SCHEDULE[-1]*eg),"kinetic_l2_diagnostic":float(kin),"raw_jacobian_barrier":jb,"weighted_jacobian_barrier":cfg["barrier_weight"]*jb,"total_objective":corr+float(SCHEDULE[-1]*eg)+cfg["barrier_weight"]*jb,
            "mean_velocity_magnitude":float(mag.mean()),"rms_velocity_magnitude":float(torch.sqrt(mag.square().mean())),"maximum_velocity_magnitude":float(mag.max()),"mean_displacement":float(disp.mean()),"maximum_displacement":float(disp.max()),
            "minimum_jacobian":float(jac.min()),"jacobian_p01":float(np.quantile(jac,.01)),"jacobian_p05":float(np.quantile(jac,.05)),"jacobian_median":float(np.median(jac)),"nonpositive_jacobian_proportion":float(np.mean(jac<=0)),"runtime_seconds":runtime,"closure_evaluations":calls,"parameter_count":int(fields.numel()),"spatial_coefficients_per_slice":int(fields[0].numel()),"time_slices":int(fields.shape[0]),"safe":bool(valid["valid"] and jac.min()>0 and np.isfinite(corr)),"valid_geometry":bool(valid["valid"]),"converged":None,"convergence_status":"L-BFGS completed without exception; backend termination reason unavailable"},d.numpy(),ev.numpy()


def stationary_baselines(shapes,mean,targets,bounds):
    source=torch.tensor(mean,dtype=DTYPE);base=[];art={};bcfg={"steps":15,"basis":"cartesian","barrier_resolution":21,"lbfgs_max_iter":80,"tolerance_grad":1e-12,"tolerance_change":1e-15}
    lcfg={"steps":15,"barrier_resolution":21,"barrier_tau":.05,"barrier_weight":.01,"local_lbfgs_max_iter":60}
    for t in targets:
        m,c24,_,_,_=s2.run_schedule(mean,t["shape"],bounds,bcfg,SCHEDULE,.05,.01,(6,12,24));target=torch.tensor(t["shape"],dtype=DTYPE);sm,coef,d,ev=s3.run_local(source,target,torch.tensor(c24),bounds,lcfg,48,.15,1.,"joint");sm["runtime_seconds"]+=m["runtime_seconds"]
        base.append({"target_label":t["label"],"target_index":t["index"],"filename":t["filename"],"method":"stationary_local48",**sm});art[t["label"]]=(coef,d,ev,sm)
    return base,art


def warm_fields(coef,T,G,bounds):
    g,le,te=(torch.tensor(x,dtype=DTYPE) for x in coef);xs=torch.linspace(bounds[0],bounds[1],G,dtype=DTYPE);ys=torch.linspace(bounds[2],bounds[3],G,dtype=DTYPE);yy,xx=torch.meshgrid(ys,xs,indexing="ij");p=torch.stack((xx.T.flatten(),yy.T.flatten()),1)
    v=s3.hierarchical_velocity(p,g,le,te,bounds,.15).reshape(G,G,2)
    return v.unsqueeze(0).repeat(T,1,1,1)


def row(t,name,init,T,G,m):return {"target_label":t["label"],"target_index":t["index"],"filename":t["filename"],"method":name,"initialization":init,"spatial_resolution":G,"time_slices":T,**m}


def residual_rows(t,method,d,ev):
    sq=np.sum(ev**2,1);total=max(float(sq.sum()),1e-300);rows=[]
    for i,v in enumerate(sq):
        x=float(t["shape"][i,0]);region="LE" if x<=.15 else ("TE" if x>=.85 else "mid")
        rows.append({"target_label":t["label"],"method":method,"point_index":i,"surface":"upper" if i<100 else "lower","x":x,"y":float(t["shape"][i,1]),"region":region,"squared_error":float(v),"residual_fraction":float(v/total),"error_x":float(ev[i,0]),"error_y":float(ev[i,1])})
    return rows


def main():
    for p in ["baseline","t2","t4","t8","residual_analysis","figures","tables","report"]:(OUT/p).mkdir(parents=True,exist_ok=True)
    shapes,mean,targets,_=s2.load_inputs(OUT);bounds=(0.,1.,float(shapes[:,:,1].min()-.08),float(shapes[:,:,1].max()+.08));source=torch.tensor(mean,dtype=DTYPE)
    cfg={"data_path":str(s2.DATA),"direct_mean":str(s2.OUT/"direct_coordinate_mean/coordinates.npy"),"development_subset":str(s2.STAGE1/"development_subset"),"model":"piecewise-constant time-dependent Cartesian velocity; LDDMM-style POC, not RKHS-LDDMM","configurations":[{"T":2,"G":12},{"T":4,"G":12},{"T":4,"G":24},{"T":8,"G":12}],"substeps_per_slice":4,"schedule":list(SCHEDULE),"barrier_tau":.05,"barrier_weight":.01,"barrier_resolution":21,"jacobian_report_resolution":51,"lbfgs_max_iter":60,"dtype":"float64"};jwrite(OUT/"config.json",cfg)
    baseline,stationary_art=stationary_baselines(shapes,mean,targets,bounds);cwrite(OUT/"stationary_baseline.csv",baseline)
    results=[];art={}
    for spec in cfg["configurations"]:
      T,G=spec["T"],spec["G"]
      for t in targets:
        target=torch.tensor(t["shape"],dtype=DTYPE);fields=torch.zeros((T,G,G,2),dtype=DTYPE);f,rt,calls=optimise(source,target,fields,bounds,cfg);m,d,ev=metrics(source,target,f,bounds,cfg,rt,calls);name=f"t{T}_g{G}_zero";results.append(row(t,name,"zero",T,G,m));art[f"{name}_{t['label']}"]=(m,f.numpy(),d,ev)
    # One warm-start comparison on the most expressive moderate spatial model.
    for t in targets:
        target=torch.tensor(t["shape"],dtype=DTYPE);fields=warm_fields(stationary_art[t["label"]][0],4,24,bounds);f,rt,calls=optimise(source,target,fields,bounds,cfg);m,d,ev=metrics(source,target,f,bounds,cfg,rt,calls);name="t4_g24_warm";results.append(row(t,name,"stationary_copy",4,24,m));art[f"{name}_{t['label']}"]=(m,f.numpy(),d,ev)
    # The fixed-field convergence check shows four substeps are not invariant at
    # the requested scale, so re-optimise only the best temporal configuration at 8.
    cfg8={**cfg,"substeps_per_slice":8}
    for t in targets:
        target=torch.tensor(t["shape"],dtype=DTYPE);fields=torch.zeros((4,24,24,2),dtype=DTYPE);f,rt,calls=optimise(source,target,fields,bounds,cfg8);m,d,ev=metrics(source,target,f,bounds,cfg8,rt,calls);name="t4_g24_zero_sub8";results.append(row(t,name,"zero",4,24,m));art[f"{name}_{t['label']}"]=(m,f.numpy(),d,ev)
    cwrite(OUT/"lddmm_results.csv",results)
    methods={"stationary_local48":baseline}
    for name in sorted(set(r["method"] for r in results)):methods[name]=[r for r in results if r["method"]==name]
    summary=[]
    for name,rr in methods.items():
        n=[r for r in rr if r["target_label"]!="identity"];summary.append({"method":name,"parameter_count":int(n[0].get("parameter_count",n[0].get("trainable_coefficients"))),"all_safe":all(r["safe"] for r in n),"median_mse":float(np.median([r["correspondence_mse"] for r in n])),"worst_mse":max(r["correspondence_mse"] for r in n),"reaching_1e_9":sum(r["safe"] and r["correspondence_mse"]<=1e-9 for r in n),"reaching_1e_8":sum(r["safe"] and r["correspondence_mse"]<=1e-8 for r in n),"reaching_1e_7":sum(r["safe"] and r["correspondence_mse"]<=1e-7 for r in n),"minimum_jacobian":min(r["minimum_jacobian"] for r in n),"total_runtime_seconds":sum(r["runtime_seconds"] for r in n),"mse_times_parameters":float(np.median([r["correspondence_mse"] for r in n]))*int(n[0].get("parameter_count",n[0].get("trainable_coefficients")))})
    cwrite(OUT/"temporal_resolution_comparison.csv",summary);cwrite(OUT/"parameter_efficiency.csv",summary)
    safe=[r for r in summary if r["all_safe"] and r["method"]!="stationary_local48"];best=min(safe,key=lambda r:r["median_mse"]);best_name=best["method"]
    point=[]
    for t in targets:
        _,sd,sev,_=stationary_art[t["label"]];point+=residual_rows(t,"stationary_local48",sd,sev)
        m,f,d,ev=art[f"{best_name}_{t['label']}"];point+=residual_rows(t,best_name,d,ev)
        if t["label"]!="identity":residual_plot(t,sd,sev,d,ev,best_name,OUT/"residual_analysis"/f"{t['label']}.png")
    cwrite(OUT/"pointwise_residuals.csv",point);regions=[]
    for method in ("stationary_local48",best_name):
      for t in targets:
        rr=[r for r in point if r["method"]==method and r["target_label"]==t["label"]];tot=sum(r["squared_error"] for r in rr)
        regions.append({"method":method,"target_label":t["label"],"leading_edge_fraction":sum(r["squared_error"] for r in rr if r["region"]=="LE")/max(tot,1e-300),"trailing_edge_fraction":sum(r["squared_error"] for r in rr if r["region"]=="TE")/max(tot,1e-300),"mid_chord_fraction":sum(r["squared_error"] for r in rr if r["region"]=="mid")/max(tot,1e-300)})
    cwrite(OUT/"residual_region_summary.csv",regions)
    # Numerical convergence check: re-integrate best fields with doubled substeps.
    numerical=[]
    for t in targets[1:]:
        m,f,d,ev=art[f"{best_name}_{t['label']}"];target=torch.tensor(t["shape"],dtype=DTYPE);ft=torch.tensor(f,dtype=DTYPE)
        for sub in (4,8):
            cc={**cfg,"substeps_per_slice":sub};mm,_,_=metrics(source,target,ft,bounds,cc,0,0);numerical.append({"target_label":t["label"],"method":best_name,"substeps_per_slice":sub,**mm})
    cwrite(OUT/"tables"/"integration_convergence.csv",numerical)
    best_cfg={"selected_method":best_name,"best_summary":best,"all_summaries":summary,"integration_convergence":numerical,"implementation_class":"LDDMM-style time-dependent proof-of-concept; not classical RKHS-LDDMM"};jwrite(OUT/"best_configuration.json",best_cfg);summary_plot(summary,OUT/"figures"/"method_comparison.png");write_report(best_cfg,regions)
    print(json.dumps(best_cfg,indent=2))


def residual_plot(t,sd,sev,d,ev,name,path):
    ss=np.sum(sev**2,1);ns=np.sum(ev**2,1);fig,axs=plt.subplots(2,2,figsize=(11,8));plot_shape(axs[0,0],t["shape"],color="#2563eb",label="target");plot_shape(axs[0,0],sd,color="#6b7280",linestyle="--",label="stationary");plot_shape(axs[0,0],d,color="#ea580c",linestyle=":",label=name);axs[0,0].set_aspect("equal");axs[0,0].legend();axs[0,0].grid(alpha=.2)
    axs[0,1].plot(ss,label="stationary",color="#6b7280");axs[0,1].plot(ns,label=name,color="#ea580c");axs[0,1].set_yscale("log");axs[0,1].set(xlabel="point index",ylabel="squared error");axs[0,1].legend();axs[0,1].grid(alpha=.2)
    le=t["shape"][:,0]<=.15;te=t["shape"][:,0]>=.85
    for ax,mask,title in [(axs[1,0],le,"Leading edge"),(axs[1,1],te,"Trailing edge")]:ax.plot(t["shape"][mask,0],ss[mask],"o-",label="stationary");ax.plot(t["shape"][mask,0],ns[mask],"o-",label=name);ax.set_yscale("log");ax.set_title(title);ax.set_xlabel("x");ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(path,dpi=200);plt.close(fig)


def summary_plot(summary,path):
    names=[r["method"] for r in summary];med=[r["median_mse"] for r in summary];worst=[r["worst_mse"] for r in summary];x=np.arange(len(names));fig,ax=plt.subplots(figsize=(10,5));ax.plot(x,med,"o-",label="median");ax.plot(x,worst,"s--",label="worst");ax.axhline(1e-9,color="#ea580c",linestyle=":");ax.axhline(1e-8,color="#ca8a04",linestyle=":");ax.set_yscale("log");ax.set_xticks(x,names,rotation=30,ha="right");ax.set_ylabel("safe correspondence MSE");ax.legend();ax.grid(axis="y",alpha=.2);fig.tight_layout();fig.savefig(path,dpi=220);plt.close(fig)


def write_report(best_cfg,regions):
    ss={r["method"]:r for r in best_cfg["all_summaries"]};base=ss["stationary_local48"];best=best_cfg["best_summary"]
    def bands(method):
        q=[r for r in regions if r["method"]==method and r["target_label"]!="identity"];return tuple(float(np.mean([r[k] for r in q])) for k in ("leading_edge_fraction","trailing_edge_fraction","mid_chord_fraction"))
    ble,bte,bmid=bands("stationary_local48");nle,nte,nmid=bands(best["method"]);zero=ss.get("t4_g24_zero");warm=ss.get("t4_g24_warm")
    lines=["# Airfoil LDDMM-style 时间依赖配准概念验证", "", "## 技术摘要", "", f"最佳安全时间依赖方法为 `{best['method']}`：中位 MSE `{best['median_mse']:.3e}`，最差 `{best['worst_mse']:.3e}`，最小 Jacobian `{best['minimum_jacobian']:.3f}`。Stationary local48 控制为中位 `{base['median_mse']:.3e}`、最差 `{base['worst_mse']:.3e}`。", "", f"时间依赖方法达到 1e-9/1e-8/1e-7 的数量为 `{best['reaching_1e_9']}/8`、`{best['reaching_1e_8']}/8`、`{best['reaching_1e_7']}/8`。", "", "## 模型定义", "", "这是分片常数 Cartesian 速度路径：每个时间区间使用独立 `v^(m)`，每片内用 4 个 Euler 子步，`dt=1/(4T)`。数据项为逐点 MSE。所谓 geodesic energy 是 `T^-1 Σ_m R_D(v^(m))` 的透明离散 Dirichlet 代理，并非经典 RKHS V-norm。", "", "总目标为 `D_corr + λ_G E_G + λ_J R_J`，其中 λ_G 使用 `1e-2→1e-4→1e-5`，Jacobian barrier 与 Stage 3 相同。所有项分开保存。", "", "## 直接比较", "", "| 方法 | 参数数 | 中位 MSE | 最差 MSE | ≤1e-9 | ≤1e-8 | ≤1e-7 | 最小 J | 总耗时 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|", *[f"| {r['method']} | {r['parameter_count']} | {r['median_mse']:.3e} | {r['worst_mse']:.3e} | {r['reaching_1e_9']} | {r['reaching_1e_8']} | {r['reaching_1e_7']} | {r['minimum_jacobian']:.3f} | {r['total_runtime_seconds']:.2f}s |" for r in best_cfg["all_summaries"]], "", "## 残差与数值检查", "", f"Stationary 的平均 LE/TE/mid 残差比例为 `{ble:.1%}/{bte:.1%}/{bmid:.1%}`；最佳时间依赖方法为 `{nle:.1%}/{nte:.1%}/{nmid:.1%}`。固定最终参数把每片子步从 4 加倍到 8 的结果保存在 tables/integration_convergence.csv。", "", "## 十二个问题的回答", "", f"1. 时间依赖是否改善：最佳时间依赖中位 `{best['median_mse']:.3e}`，stationary 为 `{base['median_mse']:.3e}`。", f"2. 最佳安全中位 MSE：`{best['median_mse']:.3e}`。", f"3. 最佳安全最差 MSE：`{best['worst_mse']:.3e}`。", f"4. 达到 1e-9：`{best['reaching_1e_9']}/8`。", f"5. 达到 1e-8：`{best['reaching_1e_8']}/8`。", f"6. LE/TE 集中度：stationary 合计 `{ble+bte:.1%}`，时间依赖 `{nle+nte:.1%}`。", f"7. Jacobian 安全：最小 `{best['minimum_jacobian']:.3f}`，所有入选结果必须正 Jacobian且几何有效。", f"8. 代价：最佳方法 `{best['parameter_count']}` 参数、八目标总耗时 `{best['total_runtime_seconds']:.2f}s`；stationary 为 `{base['parameter_count']}` 和 `{base['total_runtime_seconds']:.2f}s`。", f"9. Warm start：T4/G24 zero 中位 `{zero['median_mse']:.3e}`，warm 中位 `{warm['median_mse']:.3e}`。", "10. 是否值得复杂度取决于相对 stationary 的配对改善和参数效率；本报告不以单个易目标作为成功。", "11. 当前只是 LDDMM-style time-dependent POC，不是经典 RKHS-LDDMM。经典实现仍需 RKHS kernel/平滑算子、正确 V-norm，以及 shooting 或 momentum 表示。", "12. 若仍未到 1e-9，剩余瓶颈由空间分辨率、当前 Dirichlet 代理正则、边缘局部表达和时间参数效率共同判断。", "", "## 停止与限制", "", "只使用固定开发集；未运行全量 1654、K、GMM、模板选择、气动优化、commit 或 push。Jacobian 正性是 51×51 数值诊断，不是解析证明。等待批准。"]
    (OUT/"final_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":main()
