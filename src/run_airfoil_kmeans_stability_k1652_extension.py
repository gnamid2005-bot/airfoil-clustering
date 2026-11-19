"""Resume-safe limiting-behaviour extension of direct airfoil K-means to K=1652."""
from __future__ import annotations
import csv, json, math, platform, time
from itertools import combinations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import adjusted_rand_score
from run_airfoil_kmeans_stability_k400_extension import digest, dump, read_csv, run_model, summarize_metric, write_csv

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/"data/airfoils/processed/airfoils_n100_seed0.npz"
OLD0=ROOT/"outputs/airfoil_kmeans_stability"
OLD400=ROOT/"outputs/airfoil_kmeans_stability_k400_extension"
OUT=ROOT/"outputs/airfoil_kmeans_stability_k1652_extension"
HASH="9ec711663c4d3bde214f2ac6359591b644e5d66e849811c9fb728b593a5df2d7"
N=1654; SEEDS=list(range(20)); NEW_KS=list(range(401,1653))

def occupancy(labels,k):
    c=np.bincount(labels,minlength=k); s1=int(np.sum(c==1)); s2=int(np.sum(c==2)); le3=int(np.sum(c<=3)); lt5=int(np.sum(c<5)); lt10=int(np.sum(c<10))
    sam1=int(c[c==1].sum()); sam5=int(c[c<5].sum()); sam10=int(c[c<10].sum()); gt1=int(np.sum(c>1))
    return {"n_nonempty_clusters":int(np.count_nonzero(c)),"cluster_size_sum":int(c.sum()),"singleton_clusters":s1,"clusters_size_2":s2,"clusters_size_le_3":le3,"clusters_size_lt_5":lt5,"clusters_size_lt_10":lt10,"clusters_size_gt_1":gt1,"singleton_cluster_fraction":s1/k,"cluster_fraction_size_lt_5":lt5/k,"cluster_fraction_size_lt_10":lt10/k,"samples_in_singleton_clusters":sam1,"samples_in_clusters_lt_5":sam5,"samples_in_clusters_lt_10":sam10,"sample_fraction_in_singleton_clusters":sam1/N,"sample_fraction_in_clusters_lt_5":sam5/N,"sample_fraction_in_clusters_lt_10":sam10/N,"minimum_cluster_size":int(c.min()),"median_cluster_size":float(np.median(c)),"maximum_cluster_size":int(c.max()),"mean_cluster_size":N/k,"cluster_size_cv":float(np.std(c)/np.mean(c))}

def execute(X):
    ck=OUT/"checkpoints"; ck.mkdir(exist_ok=True); fit=skip=0; started=time.perf_counter()
    for k in NEW_KS:
        p=ck/f"k{k:04d}.npz"
        if p.exists():
            try:
                z=np.load(p,allow_pickle=False)
                if z["labels"].shape==(20,N) and np.array_equal(z["seeds"],SEEDS): skip+=20; continue
            except Exception: pass
        labels=np.empty((20,N),np.int16); inertia=np.empty(20); nit=np.empty(20,np.int16); conv=np.empty(20,bool); warns=[]
        for i,s in enumerate(SEEDS):
            m,l,cw=run_model(X,k,s); labels[i]=l; inertia[i]=m.inertia_; nit[i]=m.n_iter_; conv[i]=not cw and m.n_iter_<m.max_iter; warns.append(" | ".join(cw)); fit+=1
        tmp=p.with_suffix(".tmp.npz"); np.savez_compressed(tmp,labels=labels,inertia=inertia,n_iterations=nit,converged=conv,seeds=np.asarray(SEEDS),warnings=np.asarray(warns)); tmp.replace(p)
        if k==401 or k%25==0 or k==1652: print(json.dumps({"completed_k":k,"fits":fit,"elapsed_seconds":time.perf_counter()-started}),flush=True)
    return fit,skip,time.perf_counter()-started

def load_existing_labels(k):
    if k<=20:
        return np.load(OLD0/"kmeans_labels.npz",allow_pickle=False)["labels"][k-2]
    return np.load(OLD400/"checkpoints"/f"k{k:03d}.npz",allow_pickle=False)["labels"]

def shade(ax,lo,hi,color,alpha=.06): ax.axvspan(lo,hi,color=color,alpha=alpha,lw=0)
def base(ax,xl,yl,title): ax.set(xlabel=xl,ylabel=yl,title=title); ax.grid(alpha=.18); ax.spines[["top","right"]].set_visible(False)
def context(ax,full=True):
    shade(ax,3,5,"#2563EB",.08); shade(ax,6,8,"#F59E0B",.08)
    if full: shade(ax,400,1652,"#6B7280",.035)

def make_plots(stab,iner,cs):
    fd=OUT/"figures"; fd.mkdir(exist_ok=True); blue="#1D4ED8"; orange="#C2410C"; purple="#7C3AED"; teal="#0F766E"; pink="#BE185D"
    sk=np.array([int(r["k"]) for r in stab]); mean=np.array([float(r["mean_pairwise_ari"]) for r in stab]); mn=np.array([float(r["min_pairwise_ari"]) for r in stab]); p10=np.array([float(r["p10_pairwise_ari"]) for r in stab]); p90=np.array([float(r["p90_pairwise_ari"]) for r in stab])
    def stability(lo,hi,name,title):
        m=(sk>=lo)&(sk<=hi); fig,ax=plt.subplots(figsize=(10,5.5)); ax.fill_between(sk[m],p10[m],p90[m],color="#93C5FD",alpha=.38,label="10th–90th percentile"); ax.plot(sk[m],mean[m],color=blue,lw=1.5,label="mean ARI"); context(ax,hi>400); base(ax,"Number of clusters K","Adjusted Rand Index",title); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(fd/name,dpi=220); plt.close(fig)
    stability(2,1652,"stability_full_k2_k1652.png","K-means Stability vs Number of Clusters"); stability(2,30,"stability_early_k2_k30.png","K-means Stability — Early Region"); stability(20,400,"stability_intermediate_k20_k400.png","K-means Stability — Intermediate Region"); stability(400,1652,"stability_high_k400_k1652.png","K-means Stability — High-K Region")
    fig,ax=plt.subplots(figsize=(10,5.5)); ax.plot(sk,mean,label="mean",color=blue); ax.plot(sk,mn,label="minimum",color=orange,alpha=.85); ax.plot(sk,p10,label="10th percentile",color=purple,alpha=.9); context(ax); base(ax,"Number of clusters K","Adjusted Rand Index","Mean / Minimum / 10th-percentile ARI vs K"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(fd/"ari_mean_min_p10_k2_k1652.png",dpi=220); plt.close(fig)
    ik=np.array([int(r["k"]) for r in iner]); im=np.array([float(r["mean_inertia"]) for r in iner]); delta=np.array([np.nan if r["relative_inertia_reduction"]=="" else float(r["relative_inertia_reduction"]) for r in iner])
    for log,name in [(False,"inertia_full_k1_k1652.png"),(True,"inertia_full_logy_k1_k1652.png")]:
        fig,ax=plt.subplots(figsize=(10,5.5)); ax.plot(ik,im,color="#B45309"); context(ax); base(ax,"Number of clusters K","Mean inertia"+(" (log scale)" if log else ""),"K-means Inertia vs Number of Clusters");
        if log: ax.set_yscale("log")
        fig.tight_layout(); fig.savefig(fd/name,dpi=220); plt.close(fig)
    m=ik<=30; fig,ax=plt.subplots(figsize=(9,5.2)); ax.plot(ik[m],im[m],"o-",color="#B45309"); context(ax,False); base(ax,"Number of clusters K","Mean inertia","K-means Inertia — Early Region"); fig.tight_layout(); fig.savefig(fd/"inertia_early_k1_k30.png",dpi=220); plt.close(fig)
    for lo,hi,name in [(2,1652,"relative_inertia_reduction_full.png"),(2,30,"relative_inertia_reduction_early.png")]:
        m=(ik>=lo)&(ik<=hi); fig,ax=plt.subplots(figsize=(10,5.2)); ax.plot(ik[m],delta[m],color=purple,lw=1.3); ax.yaxis.set_major_formatter(lambda x,pos:f"{100*x:.1f}%"); context(ax,hi>400); base(ax,"Number of clusters K","Relative inertia reduction","Relative Inertia Improvement vs K"+(" — Early Region" if hi==30 else "")); fig.tight_layout(); fig.savefig(fd/name,dpi=220); plt.close(fig)
    ck=np.array([int(r["k"]) for r in cs]); sing=np.array([float(r["mean_singleton_cluster_fraction"]) for r in cs]); cl5=np.array([float(r["mean_cluster_fraction_size_lt_5"]) for r in cs]); cl10=np.array([float(r["mean_cluster_fraction_size_lt_10"]) for r in cs]); ss1=np.array([float(r["mean_sample_fraction_in_singleton_clusters"]) for r in cs]); ss5=np.array([float(r["mean_sample_fraction_in_clusters_lt_5"]) for r in cs]); ss10=np.array([float(r["mean_sample_fraction_in_clusters_lt_10"]) for r in cs]); med=np.array([float(r["mean_median_cluster_size"]) for r in cs]); avg=N/ck
    fig,ax=plt.subplots(figsize=(10,5.5)); ax.plot(ck,sing,color=blue); context(ax); base(ax,"Number of clusters K","Mean singleton-cluster fraction","Singleton-cluster Fraction vs K"); ax.yaxis.set_major_formatter(lambda x,pos:f"{100*x:.0f}%"); fig.tight_layout(); fig.savefig(fd/"singleton_cluster_fraction.png",dpi=220); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5.5)); ax.plot(ck,cl5,label="clusters size <5",color=purple); ax.plot(ck,cl10,label="clusters size <10",color=orange); context(ax); base(ax,"Number of clusters K","Mean fraction of clusters","Tiny-cluster Fractions vs K"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(fd/"tiny_cluster_fractions.png",dpi=220); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5.5)); ax.plot(ck,ss1,label="samples in singleton clusters",color=blue); ax.plot(ck,ss5,label="samples in clusters <5",color=purple); ax.plot(ck,ss10,label="samples in clusters <10",color=orange); context(ax); base(ax,"Number of clusters K","Mean fraction of samples","Fraction of Samples in Tiny Clusters vs K"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(fd/"sample_fraction_tiny_clusters.png",dpi=220); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5.5)); ax.plot(ck,avg,label="N/K",color=teal); ax.plot(ck,med,label="mean across seeds of median cluster size",color=pink); context(ax); base(ax,"Number of clusters K","Samples per cluster","Average / Median Cluster Size vs K"); ax.set_yscale("log"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(fd/"average_median_cluster_size.png",dpi=220); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,6)); sc=ax.scatter(sing[1:],mean[1:],c=sk[1:],cmap="viridis",s=12,alpha=.75); base(ax,"Mean singleton-cluster fraction","Mean pairwise ARI","Stability vs Fragmentation"); fig.colorbar(sc,ax=ax,label="K"); fig.tight_layout(); fig.savefig(fd/"stability_vs_singleton_fraction.png",dpi=220); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,6)); sc=ax.scatter(N/sk[1:],mean[1:],c=sk[1:],cmap="viridis",s=12,alpha=.75); ax.set_xscale("log"); base(ax,"Mean cluster size N/K (log scale)","Mean pairwise ARI","ARI vs Mean Cluster Size"); fig.colorbar(sc,ax=ax,label="K"); fig.tight_layout(); fig.savefig(fd/"ari_vs_mean_cluster_size.png",dpi=220); plt.close(fig)
    fig,axs=plt.subplots(4,1,figsize=(11,12),sharex=True); axs[0].plot(sk,mean,color=blue); axs[0].fill_between(sk,p10,p90,color="#93C5FD",alpha=.35); base(axs[0],"","Mean ARI","Stability"); axs[1].plot(ik,im,color="#B45309"); axs[1].set_yscale("log"); base(axs[1],"","Mean inertia (log)","Inertia collapse"); axs[2].plot(ck,sing,color=blue); base(axs[2],"","Singleton fraction","Singleton fragmentation"); axs[3].plot(ck,cl10,color=orange); base(axs[3],"K","Fraction clusters <10","Tiny-cluster fragmentation"); [context(a) for a in axs]; fig.suptitle("K-means K=1–1652 Comprehensive Diagnostic Overview"); fig.tight_layout(); fig.savefig(fd/"comprehensive_diagnostic_overview.png",dpi=220); plt.close(fig)

def threshold_rows(cs):
    specs=[("singleton_cluster_fraction",[.10,.25,.50,.75,.90]),("cluster_fraction_size_lt_10",[.50,.75,.90]),("sample_fraction_in_clusters_lt_10",[.50,.75,.90])]; out=[]
    for metric,ths in specs:
        key="mean_"+metric
        for t in ths:
            hit=next((r for r in cs if float(r[key])>t),None); out.append({"metric":metric,"threshold_strictly_greater_than":t,"first_k":int(hit["k"]) if hit else "","value_at_first_k":float(hit[key]) if hit else ""})
    return out

def main():
    if OUT.exists() and not (OUT/"checkpoints").exists(): raise FileExistsError(f"Refusing unrelated existing output {OUT}")
    OUT.mkdir(exist_ok=True)
    cfg=json.loads((OLD400/"experiment_config.json").read_text()); expected={"algorithm":"lloyd","implementation":"sklearn.cluster.KMeans","initialization":"k-means++","max_iter":1000,"n_init":1,"random_state":"explicit seed 0..19","tolerance":1e-6}
    if digest(INPUT)!=HASH or cfg["input_sha256"]!=HASH or cfg["kmeans"]!=expected: raise ValueError("Input/config mismatch")
    shapes=np.load(INPUT,allow_pickle=False)["shapes"].astype(float); X=np.ascontiguousarray(shapes.reshape(N,-1))
    fitted,skipped,runtime=execute(X)
    new=[]; new_labels={}
    for k in NEW_KS:
        z=np.load(OUT/"checkpoints"/f"k{k:04d}.npz",allow_pickle=False); new_labels[k]=z["labels"]
        for i,s in enumerate(SEEDS): new.append({"k":k,"seed":s,"source":"new","inertia":float(z["inertia"][i]),"n_iterations":int(z["n_iterations"][i]),"converged":bool(z["converged"][i]),"convergence_warning":str(z["warnings"][i]),**occupancy(z["labels"][i],k)})
    write_csv(OUT/"kmeans_new_runs_k401_k1652.csv",new)
    old_pair=read_csv(OLD400/"kmeans_pairwise_ari_k1_k400.csv"); new_pair=[]; new_stab=[]
    for k in NEW_KS:
        vals=[]
        for a,b in combinations(range(20),2):
            v=float(adjusted_rand_score(new_labels[k][a],new_labels[k][b])); vals.append(v); new_pair.append({"k":k,"seed_a":a,"seed_b":b,"ari":v})
        v=np.asarray(vals); new_stab.append({"k":k,"stability_is_trivial":False,"n_pairwise_comparisons":190,"mean_pairwise_ari":float(v.mean()),"std_pairwise_ari":float(v.std()),"median_pairwise_ari":float(np.median(v)),"min_pairwise_ari":float(v.min()),"max_pairwise_ari":float(v.max()),"p10_pairwise_ari":float(np.percentile(v,10)),"p90_pairwise_ari":float(np.percentile(v,90))})
    write_csv(OUT/"kmeans_pairwise_ari_k1_k1652.csv",old_pair+new_pair)
    old_stab=read_csv(OLD400/"kmeans_stability_summary_k1_k400.csv"); stab=old_stab+new_stab; write_csv(OUT/"kmeans_stability_summary_k1_k1652.csv",stab,list(old_stab[0]))
    old_in=read_csv(OLD400/"kmeans_inertia_summary_k1_k400.csv"); inertia=list(old_in); prev=float(old_in[-1]["mean_inertia"])
    for k in NEW_KS:
        rows=[r for r in new if r["k"]==k]; sm=summarize_metric(rows,"inertia","inertia"); cur=sm["mean_inertia"]; absred=prev-cur; rel=absred/prev if abs(prev)>1e-15 else ""; inertia.append({"k":k,"runs":20,**sm,"absolute_inertia_reduction":absred,"relative_inertia_reduction":rel,"normalized_full_range_chord_distance":""}); prev=cur
    write_csv(OUT/"kmeans_inertia_summary_k1_k1652.csv",inertia,list(old_in[0]))
    cs=[]
    for k in range(1,1653):
        if k==1: labels=[np.zeros(N,np.int16)]
        elif k<=400: labels=load_existing_labels(k)
        else: labels=new_labels[k]
        rows=[occupancy(x,k) for x in labels]; cr={"k":k,"runs":len(rows),"n_minus_k":N-k,"mean_cluster_size_theoretical":N/k,"singleton_count_lower_bound":max(0,2*k-N),"singleton_fraction_lower_bound":max(0,2*k-N)/k,"minimum_clusters_size_gt1_necessary":0 if k==N else 1,"maximum_clusters_size_gt1_possible":min(k,N-k)}
        for key in ["singleton_clusters","clusters_size_2","clusters_size_le_3","clusters_size_lt_5","clusters_size_lt_10","clusters_size_gt_1","singleton_cluster_fraction","cluster_fraction_size_lt_5","cluster_fraction_size_lt_10","sample_fraction_in_singleton_clusters","sample_fraction_in_clusters_lt_5","sample_fraction_in_clusters_lt_10","minimum_cluster_size","median_cluster_size","maximum_cluster_size","mean_cluster_size","cluster_size_cv"]: cr.update(summarize_metric(rows,key,key))
        cr["all_requested_clusters_nonempty_all_runs"]=all(r["n_nonempty_clusters"]==k for r in rows); cr["cluster_size_sum_n_all_runs"]=all(r["cluster_size_sum"]==N for r in rows); cs.append(cr)
    write_csv(OUT/"kmeans_cluster_size_summary_k1_k1652.csv",cs)
    thresholds=threshold_rows(cs); write_csv(OUT/"fragmentation_thresholds.csv",thresholds)
    ranges=[(2,20),(21,50),(51,100),(101,200),(201,400),(401,800),(801,1200),(1201,1500),(1501,1652)]; rr=[]
    for lo,hi in ranges:
        ss=[r for r in stab if lo<=int(r["k"])<=hi]; cc=[r for r in cs if lo<=int(r["k"])<=hi]; ii=[r for r in inertia if lo<=int(r["k"])<=hi]; bm=max(ss,key=lambda r:float(r["mean_pairwise_ari"])); bn=max(ss,key=lambda r:float(r["min_pairwise_ari"])); rel=[float(r["relative_inertia_reduction"]) for r in ii if r["relative_inertia_reduction"]!=""]
        rr.append({"range":f"K={lo}-{hi}","maximum_mean_ari":float(bm["mean_pairwise_ari"]),"k_at_maximum_mean_ari":int(bm["k"]),"maximum_minimum_ari":float(bn["min_pairwise_ari"]),"k_at_maximum_minimum_ari":int(bn["k"]),"median_mean_ari":float(np.median([float(r["mean_pairwise_ari"]) for r in ss])),"median_singleton_fraction":float(np.median([r["mean_singleton_cluster_fraction"] for r in cc])),"median_cluster_fraction_lt_10":float(np.median([r["mean_cluster_fraction_size_lt_10"] for r in cc])),"median_sample_fraction_in_clusters_lt_10":float(np.median([r["mean_sample_fraction_in_clusters_lt_10"] for r in cc])),"median_relative_inertia_gain":float(np.median(rel))})
    write_csv(OUT/"range_summary.csv",rr)
    reps=[]
    for k in [3,5,7,400,600,800,1000,1200,1400,1500,1600,1650,1652]:
        s=next(r for r in stab if int(r["k"])==k); c=cs[k-1]; i=next(r for r in inertia if int(r["k"])==k); reps.append({"k":k,"n_minus_k":N-k,"mean_ari":s["mean_pairwise_ari"],"minimum_ari":s["min_pairwise_ari"],"mean_inertia":i["mean_inertia"],"mean_singleton_fraction":c["mean_singleton_cluster_fraction"],"mean_cluster_fraction_lt_5":c["mean_cluster_fraction_size_lt_5"],"mean_cluster_fraction_lt_10":c["mean_cluster_fraction_size_lt_10"],"mean_sample_fraction_in_clusters_lt_10":c["mean_sample_fraction_in_clusters_lt_10"],"mean_cluster_size":N/k,"mean_median_cluster_size":c["mean_median_cluster_size"],"mean_clusters_size_gt_1":c["mean_clusters_size_gt_1"],"singleton_lower_bound":c["singleton_count_lower_bound"]})
    write_csv(OUT/"representative_k_comparison.csv",reps)
    make_plots(stab,inertia,cs)
    old_cs=read_csv(OLD400/"kmeans_cluster_size_summary_k1_k400.csv"); common=[("mean_singleton_clusters","mean_singleton_clusters"),("mean_clusters_size_lt_5","mean_clusters_size_lt_5"),("mean_clusters_size_lt_10","mean_clusters_size_lt_10"),("mean_singleton_cluster_fraction","mean_singleton_cluster_fraction"),("mean_cluster_fraction_size_lt_5","mean_cluster_fraction_size_lt_5"),("mean_cluster_fraction_size_lt_10","mean_cluster_fraction_size_lt_10"),("mean_sample_fraction_in_clusters_lt_10","mean_sample_fraction_in_clusters_size_lt_10"),("mean_minimum_cluster_size","mean_minimum_cluster_size"),("mean_median_cluster_size","mean_median_cluster_size"),("mean_maximum_cluster_size","mean_maximum_cluster_size"),("mean_cluster_size_cv","mean_cluster_size_cv")]
    cs_exact=all(float(cs[int(o["k"])-1][new_key])==float(o[old_key]) for o in old_cs for new_key,old_key in common)
    old_in_exact=all(inertia[i][key]==old_in[i][key] for i in range(400) for key in old_in[i]); old_stab_exact=all(stab[i][key]==old_stab[i][key] for i in range(400) for key in old_stab[i])
    finite_in=bool(np.isfinite([float(r["mean_inertia"]) for r in inertia]).all()); finite_ari=bool(np.isfinite([float(r["ari"]) for r in old_pair+new_pair]).all())
    effective=next((int(r["k"]) for r in inertia if float(r["mean_inertia"])<=1e-6*float(inertia[0]["mean_inertia"])),None)
    validation={"input_hash_before":HASH,"input_hash_after":digest(INPUT),"input_unchanged":digest(INPUT)==HASH,"existing_k1_k400_stability_reused_exactly":old_stab_exact,"existing_k1_k400_inertia_reused_exactly":old_in_exact,"existing_k1_k400_cluster_common_metrics_recomputed_identically":cs_exact,"existing_fits_recomputed":0,"new_fits_expected":25040,"new_fits_executed_this_invocation":fitted,"new_fits_resumed":skipped,"new_k_values_present":sorted(set(r["k"] for r in new))==NEW_KS,"new_run_rows":len(new),"exactly_20_fits_per_new_k":all(sum(r["k"]==k for r in new)==20 for k in NEW_KS),"new_pairwise_ari_rows":len(new_pair),"new_pairwise_ari_expected":237880,"all_pairwise_ari_rows_k2_k1652":len(old_pair+new_pair),"all_pairwise_ari_expected":1651*190,"exactly_190_ari_each_new_k":all(sum(r["k"]==k for r in new_pair)==190 for k in NEW_KS),"all_new_requested_clusters_nonempty":all(r["n_nonempty_clusters"]==r["k"] for r in new),"all_new_cluster_size_sums_n":all(r["cluster_size_sum"]==N for r in new),"all_ari_finite":finite_ari,"all_inertia_finite":finite_in,"convergence_failure_count":sum(not r["converged"] for r in new),"fit_runtime_seconds":runtime,"inertia_effectively_negligible_threshold":"mean <= 1e-6 * K1 mean","first_k_effectively_negligible":effective}
    dump(OUT/"validation.json",validation)
    dump(OUT/"experiment_config.json",{"experiment":"direct-coordinate K-means limiting behaviour K=1..1652","input_path":str(INPUT.relative_to(ROOT)),"input_sha256":HASH,"shape":list(shapes.shape),"flattened_dimension":400,"flatten_order":cfg["flatten_order"],"fixed_x_dimensions_retained":True,"representation_changed":False,"pca":False,"gmm":False,"registration":False,"velocity_fields":False,"existing_results":[str(OLD0.relative_to(ROOT)),str(OLD400.relative_to(ROOT))],"existing_k_values_reused":[1,400],"new_k_values":[401,1652],"seeds":SEEDS,"kmeans":cfg["kmeans"],"resume":{"supported":True,"checkpoint_directory":"checkpoints/","skip_rule":"valid (20,1654) labels and seeds 0..19"},"pre_run_estimate":{"new_k_values":1252,"new_fits":25040,"new_pairwise_ari":237880,"runtime_minutes":"60-120","storage_megabytes":"100-180","peak_memory_gb":"<0.5"},"inertia_relative_reduction_guard":"blank when abs(previous mean inertia) <= 1e-15","software":{"python":platform.python_version(),"numpy":np.__version__,"sklearn":sklearn.__version__,"matplotlib":matplotlib.__version__}})
    print(json.dumps(validation,indent=2),flush=True)
if __name__=="__main__":main()
