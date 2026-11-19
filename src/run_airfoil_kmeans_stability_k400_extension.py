"""Resume-safe K=1..400 extension of the direct-coordinate airfoil K-means study."""

from __future__ import annotations

import csv, hashlib, json, math, platform, time, warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/airfoils/processed/airfoils_n100_seed0.npz"
OLD = ROOT / "outputs/airfoil_kmeans_stability"
OUT = ROOT / "outputs/airfoil_kmeans_stability_k400_extension"
SEEDS = list(range(20))
OLD_KS = list(range(2, 21))
NEW_KS = list(range(21, 401))
ALL_KS = list(range(1, 401))
EXPECTED_HASH = "9ec711663c4d3bde214f2ac6359591b644e5d66e849811c9fb728b593a5df2d7"

def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def read_csv(path):
    with open(path,newline="",encoding="utf-8") as f: return list(csv.DictReader(f))

def write_csv(path, rows, fields=None):
    fields=fields or list(rows[0])
    tmp=path.with_suffix(path.suffix+".tmp")
    with open(tmp,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    tmp.replace(path)

def dump(path,obj):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True),encoding="utf-8"); tmp.replace(path)

def run_model(X,k,seed):
    model=KMeans(n_clusters=k,init="k-means++",n_init=1,max_iter=1000,tol=1e-6,random_state=seed,algorithm="lloyd")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always"); labels=model.fit_predict(X)
    cw=[str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]
    return model,labels,cw

def occupancy(labels,k,n):
    counts=np.bincount(labels,minlength=k)
    single=int(np.sum(counts==1)); lt5=int(np.sum(counts<5)); lt10=int(np.sum(counts<10)); lt1=int(np.sum(counts<.01*n))
    tiny_samples=int(np.sum(counts[counts<10]))
    return {"n_nonempty_clusters":int(np.count_nonzero(counts)),"cluster_size_sum":int(counts.sum()),
      "singleton_clusters":single,"clusters_size_lt_5":lt5,"clusters_size_lt_10":lt10,"clusters_size_lt_1pct_n":lt1,
      "singleton_cluster_fraction":single/k,"cluster_fraction_size_lt_5":lt5/k,"cluster_fraction_size_lt_10":lt10/k,
      "cluster_fraction_size_lt_1pct_n":lt1/k,"samples_in_clusters_size_lt_10":tiny_samples,
      "sample_fraction_in_clusters_size_lt_10":tiny_samples/n,"minimum_cluster_size":int(counts.min()),
      "median_cluster_size":float(np.median(counts)),"maximum_cluster_size":int(counts.max()),
      "cluster_size_cv":float(np.std(counts)/np.mean(counts)),"cluster_sizes":json.dumps(counts.tolist())}

def execute_new(X):
    ck=OUT/"checkpoints"; ck.mkdir(exist_ok=True)
    skipped=0; fitted=0; started=time.perf_counter()
    for k in NEW_KS:
        p=ck/f"k{k:03d}.npz"
        if p.exists():
            try:
                z=np.load(p,allow_pickle=False)
                if z["labels"].shape==(20,len(X)) and np.array_equal(z["seeds"],SEEDS): skipped+=20; continue
            except Exception: pass
        labels=np.empty((20,len(X)),dtype=np.int16); inertia=np.empty(20); nit=np.empty(20,dtype=np.int16); conv=np.empty(20,dtype=bool); warns=[]
        for i,s in enumerate(SEEDS):
            m,l,cw=run_model(X,k,s); labels[i]=l; inertia[i]=m.inertia_; nit[i]=m.n_iter_; conv[i]=not cw and m.n_iter_<m.max_iter; warns.append(" | ".join(cw)); fitted+=1
        tmp=p.with_suffix(".tmp.npz")
        np.savez_compressed(tmp,labels=labels,inertia=inertia,n_iterations=nit,converged=conv,seeds=np.asarray(SEEDS),warnings=np.asarray(warns))
        tmp.replace(p)
        if k%10==0 or k==21: print(json.dumps({"completed_k":k,"new_fits_this_process":fitted,"elapsed_seconds":time.perf_counter()-started}),flush=True)
    return fitted,skipped,time.perf_counter()-started

def summarize_metric(rows,key,prefix):
    a=np.asarray([float(r[key]) for r in rows])
    return {f"mean_{prefix}":float(a.mean()),f"std_{prefix}":float(a.std()),f"median_{prefix}":float(np.median(a)),f"min_{prefix}":float(a.min()),f"max_{prefix}":float(a.max())}

def local_maxima(rows,key):
    v=np.asarray([float(r[key]) for r in rows]); ks=np.asarray([int(r["k"]) for r in rows])
    return [{"k":int(ks[i]),"value":float(v[i])} for i in range(1,len(v)-1) if v[i]>v[i-1] and v[i]>v[i+1]]

def knee(ks,vals):
    x=(ks-ks.min())/(ks.max()-ks.min()); y=(vals-vals.min())/(vals.max()-vals.min())
    d=np.abs((y[-1]-y[0])*x-(x[-1]-x[0])*y+x[-1]*y[0]-y[-1]*x[0])/math.hypot(y[-1]-y[0],x[-1]-x[0])
    return int(ks[np.argmax(d)]),d

def plots(stab,iner,cs):
    fd=OUT/"figures"; fd.mkdir(exist_ok=True)
    blue="#1D4ED8"; orange="#C2410C"; purple="#7C3AED"; teal="#0F766E"; grey="#4B5563"
    def base(ax,xl,yl,title): ax.set(xlabel=xl,ylabel=yl,title=title); ax.grid(alpha=.2); ax.spines[["top","right"]].set_visible(False)
    k=np.asarray([r["k"] for r in stab]); mean=np.asarray([r["mean_pairwise_ari"] for r in stab],float); p10=np.asarray([r["p10_pairwise_ari"] for r in stab],float); p90=np.asarray([r["p90_pairwise_ari"] for r in stab],float); mn=np.asarray([r["min_pairwise_ari"] for r in stab],float)
    for zoom,name in [(False,"stability_k1_k400.png"),(True,"stability_zoom_k2_k30.png")]:
        mask=(k>=2)&(k<=30 if zoom else True); fig,ax=plt.subplots(figsize=(9,5.2)); ax.fill_between(k[mask],p10[mask],p90[mask],color="#93C5FD",alpha=.4,label="10th–90th percentile"); ax.plot(k[mask],mean[mask],color=blue,lw=1.7,label="mean pairwise ARI"); ax.plot(k[mask],mn[mask],color=orange,lw=1,alpha=.8,label="minimum ARI"); base(ax,"Number of clusters K","Adjusted Rand Index","K-means Stability vs Number of Clusters"+(" — Early Region" if zoom else "")); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(fd/name,dpi=220); plt.close(fig)
    ik=np.asarray([r["k"] for r in iner]); im=np.asarray([r["mean_inertia"] for r in iner]); ist=np.asarray([r["std_inertia"] for r in iner])
    for zoom,name in [(False,"inertia_k1_k400.png"),(True,"inertia_zoom_k1_k30.png")]:
        mask=ik<=30 if zoom else np.ones(len(ik),bool); fig,ax=plt.subplots(figsize=(9,5.2)); ax.fill_between(ik[mask],im[mask]-ist[mask],im[mask]+ist[mask],color="#FDE68A",alpha=.5); ax.plot(ik[mask],im[mask],color="#B45309",lw=1.8); base(ax,"Number of clusters K","Mean K-means inertia","K-means Inertia vs Number of Clusters"+(" — Early Region" if zoom else "")); fig.tight_layout(); fig.savefig(fd/name,dpi=220); plt.close(fig)
    delta=np.asarray([np.nan if r["relative_inertia_reduction"]=="" else r["relative_inertia_reduction"] for r in iner],float)
    for zoom,name in [(False,"relative_inertia_reduction_k2_k400.png"),(True,"relative_inertia_reduction_zoom_k2_k30.png")]:
        mask=(ik>=2)&(ik<=30 if zoom else True); fig,ax=plt.subplots(figsize=(9,5.2)); ax.plot(ik[mask],delta[mask],color=purple,lw=1.5); ax.yaxis.set_major_formatter(lambda x,pos:f"{100*x:.1f}%"); base(ax,"Number of clusters K","Relative inertia reduction","Relative Inertia Improvement vs K"+(" — Early Region" if zoom else "")); fig.tight_layout(); fig.savefig(fd/name,dpi=220); plt.close(fig)
    ck=np.asarray([r["k"] for r in cs]); cmean=np.asarray([r["mean_minimum_cluster_size"] for r in cs]); cmin=np.asarray([r["min_minimum_cluster_size"] for r in cs]);
    fig,ax=plt.subplots(figsize=(9,5.2)); ax.plot(ck,cmean,color=teal,label="mean minimum size"); ax.plot(ck,cmin,color=orange,label="smallest observed",alpha=.8); ax.axhline(10,color=grey,ls=":",label="size 10"); ax.set_yscale("symlog",linthresh=1); base(ax,"Number of clusters K","Minimum cluster size (symlog)","Minimum Cluster Size vs K"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(fd/"minimum_cluster_size_k1_k400.png",dpi=220); plt.close(fig)
    sing=np.asarray([r["mean_singleton_cluster_fraction"] for r in cs]); lt5=np.asarray([r["mean_cluster_fraction_size_lt_5"] for r in cs]); lt10=np.asarray([r["mean_cluster_fraction_size_lt_10"] for r in cs]);
    fig,ax=plt.subplots(figsize=(9,5.2)); ax.plot(ck,sing,label="singletons",color=blue); ax.plot(ck,lt5,label="size < 5",color=purple); ax.plot(ck,lt10,label="size < 10",color=orange); base(ax,"Number of clusters K","Fraction of clusters","Singleton / Tiny Cluster Fraction vs K"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(fd/"singleton_tiny_cluster_fraction_k1_k400.png",dpi=220); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,6)); sc=ax.scatter(mean[1:],lt10[1:],c=k[1:],cmap="viridis",s=18,alpha=.8); base(ax,"Mean pairwise ARI","Mean fraction of clusters with size < 10","Stability vs Cluster Pathology"); fig.colorbar(sc,ax=ax,label="K"); fig.tight_layout(); fig.savefig(fd/"stability_vs_cluster_pathology.png",dpi=220); plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(11,10),sharex=True); axs[0].plot(k,mean,color=blue); axs[0].fill_between(k,p10,p90,color="#93C5FD",alpha=.4); base(axs[0],"","Mean ARI","Stability"); axs[1].plot(ik[1:],delta[1:],color=purple); axs[1].yaxis.set_major_formatter(lambda x,pos:f"{100*x:.1f}%"); base(axs[1],"","Relative reduction","Inertia improvement"); axs[2].plot(ck,cmean,color=teal); axs[2].plot(ck,cmin,color=orange,alpha=.8); axs[2].set_yscale("symlog",linthresh=1); base(axs[2],"Number of clusters K","Minimum size","Cluster-size pathology"); fig.suptitle("K-means K=1–400 Diagnostic Overview"); fig.tight_layout(); fig.savefig(fd/"combined_diagnostic_overview.png",dpi=220); plt.close(fig)

def main():
    if OUT.exists() and not (OUT/"checkpoints").exists(): raise FileExistsError(f"Refusing unrelated existing output {OUT}")
    OUT.mkdir(exist_ok=True)
    old_config=json.loads((OLD/"experiment_config.json").read_text()); old_validation=json.loads((OLD/"validation.json").read_text())
    if digest(INPUT)!=EXPECTED_HASH or old_config["input"]["input_sha256"]!=EXPECTED_HASH: raise ValueError("Input hash mismatch")
    if old_config["kmeans"]!={"algorithm":"lloyd","implementation":"sklearn.cluster.KMeans","initialization":"k-means++","max_iter":1000,"n_init":1,"random_state":"explicit seed 0..19","tolerance":1e-6}: raise ValueError("Old KMeans config mismatch")
    shapes=np.load(INPUT,allow_pickle=False)["shapes"].astype(np.float64); X=np.ascontiguousarray(shapes.reshape(len(shapes),-1)); n=len(X)
    old_runs=read_csv(OLD/"kmeans_all_runs.csv"); old_ari=read_csv(OLD/"kmeans_pairwise_ari.csv"); old_labels=np.load(OLD/"kmeans_labels.npz",allow_pickle=False)["labels"]
    if len(old_runs)!=380 or len(old_ari)!=3610 or old_labels.shape!=(19,20,n): raise ValueError("Old results incomplete")
    fitted,skipped,fit_runtime=execute_new(X)
    m1,l1,cw1=run_model(X,1,0); k1_occ=occupancy(l1,1,n)
    all_runs=[]; labels_by_k={}
    for ki,k in enumerate(OLD_KS):
        labels_by_k[k]=old_labels[ki].astype(np.int16)
        for si,s in enumerate(SEEDS):
            source=next(r for r in old_runs if int(r["k"])==k and int(r["seed"])==s); occ=occupancy(labels_by_k[k][si],k,n)
            all_runs.append({"k":k,"seed":s,"source":"reused","inertia":float(source["inertia"]),"n_iterations":int(source["n_iterations"]),"converged":source["converged"]=="True","convergence_warning":source["convergence_warning"],**occ})
    new_rows=[]
    for k in NEW_KS:
        z=np.load(OUT/"checkpoints"/f"k{k:03d}.npz",allow_pickle=False); labels_by_k[k]=z["labels"]
        for i,s in enumerate(SEEDS):
            row={"k":k,"seed":s,"source":"new","inertia":float(z["inertia"][i]),"n_iterations":int(z["n_iterations"][i]),"converged":bool(z["converged"][i]),"convergence_warning":str(z["warnings"][i]),**occupancy(z["labels"][i],k,n)}
            all_runs.append(row); new_rows.append(row)
    fields=list(new_rows[0]); write_csv(OUT/"kmeans_new_runs_k21_k400.csv",new_rows,fields)
    pairwise=[]; stab=[{"k":1,"stability_is_trivial":True,"n_pairwise_comparisons":0,"mean_pairwise_ari":1.0,"std_pairwise_ari":0.0,"median_pairwise_ari":1.0,"min_pairwise_ari":1.0,"max_pairwise_ari":1.0,"p10_pairwise_ari":1.0,"p90_pairwise_ari":1.0}]
    for k in range(2,401):
        vals=[]
        for a,b in combinations(range(20),2):
            ari=float(adjusted_rand_score(labels_by_k[k][a],labels_by_k[k][b])); vals.append(ari); pairwise.append({"k":k,"seed_a":a,"seed_b":b,"ari":ari})
        v=np.asarray(vals); stab.append({"k":k,"stability_is_trivial":False,"n_pairwise_comparisons":190,"mean_pairwise_ari":float(v.mean()),"std_pairwise_ari":float(v.std()),"median_pairwise_ari":float(np.median(v)),"min_pairwise_ari":float(v.min()),"max_pairwise_ari":float(v.max()),"p10_pairwise_ari":float(np.percentile(v,10)),"p90_pairwise_ari":float(np.percentile(v,90))})
    write_csv(OUT/"kmeans_pairwise_ari_k1_k400.csv",pairwise); write_csv(OUT/"kmeans_stability_summary_k1_k400.csv",stab)
    k1run={"k":1,"seed":0,"source":"new_k1_sanity","inertia":float(m1.inertia_),"n_iterations":int(m1.n_iter_),"converged":not cw1,"convergence_warning":" | ".join(cw1),**k1_occ}
    inertia=[]; cs=[]; tiny=[]
    for k in ALL_KS:
        rows=[k1run] if k==1 else [r for r in all_runs if r["k"]==k]
        ir=summarize_metric(rows,"inertia","inertia"); inertia.append({"k":k,"runs":len(rows),**ir,"absolute_inertia_reduction":"","relative_inertia_reduction":""})
        cr={"k":k,"runs":len(rows),"average_samples_per_cluster":n/k}
        for key in ["singleton_clusters","clusters_size_lt_5","clusters_size_lt_10","clusters_size_lt_1pct_n","singleton_cluster_fraction","cluster_fraction_size_lt_5","cluster_fraction_size_lt_10","cluster_fraction_size_lt_1pct_n","sample_fraction_in_clusters_size_lt_10","minimum_cluster_size","median_cluster_size","maximum_cluster_size","cluster_size_cv"]: cr.update(summarize_metric(rows,key,key))
        cr["all_requested_clusters_nonempty_all_runs"]=all(r["n_nonempty_clusters"]==k for r in rows); cr["cluster_size_sum_n_all_runs"]=all(r["cluster_size_sum"]==n for r in rows); cs.append(cr)
        tiny.extend({key:r[key] for key in ["k","seed","singleton_clusters","clusters_size_lt_5","clusters_size_lt_10","clusters_size_lt_1pct_n","singleton_cluster_fraction","cluster_fraction_size_lt_5","cluster_fraction_size_lt_10","cluster_fraction_size_lt_1pct_n","samples_in_clusters_size_lt_10","sample_fraction_in_clusters_size_lt_10","minimum_cluster_size","median_cluster_size","maximum_cluster_size","cluster_size_cv"]} for r in rows)
    for i in range(1,400):
        prev=inertia[i-1]["mean_inertia"]; cur=inertia[i]["mean_inertia"]; inertia[i]["absolute_inertia_reduction"]=prev-cur; inertia[i]["relative_inertia_reduction"]=(prev-cur)/prev
    kk=np.arange(1,401); iv=np.asarray([r["mean_inertia"] for r in inertia]); full_knee,dist=knee(kk,iv); early_knee,_=knee(kk[:30],iv[:30])
    for r,d in zip(inertia,dist):r["normalized_full_range_chord_distance"]=float(d)
    write_csv(OUT/"kmeans_inertia_summary_k1_k400.csv",inertia); write_csv(OUT/"kmeans_cluster_size_summary_k1_k400.csv",cs); write_csv(OUT/"kmeans_tiny_cluster_diagnostics.csv",tiny)
    ranges=[(2,20),(21,50),(51,100),(101,200),(201,400)]; ranges_out=[]
    for lo,hi in ranges:
        sr=[r for r in stab if lo<=r["k"]<=hi]; cr=[r for r in cs if lo<=r["k"]<=hi]; ir=[r for r in inertia if lo<=r["k"]<=hi]
        best_mean=max(sr,key=lambda r:r["mean_pairwise_ari"]); best_min=max(sr,key=lambda r:r["min_pairwise_ari"])
        ranges_out.append({"range":f"K={lo}-{hi}","highest_mean_ari_k":best_mean["k"],"highest_mean_ari":best_mean["mean_pairwise_ari"],"highest_min_ari_k":best_min["k"],"highest_min_ari":best_min["min_pairwise_ari"],"median_of_mean_ari":float(np.median([r["mean_pairwise_ari"] for r in sr])),"median_mean_minimum_cluster_size":float(np.median([r["mean_minimum_cluster_size"] for r in cr])),"median_mean_singleton_cluster_fraction":float(np.median([r["mean_singleton_cluster_fraction"] for r in cr])),"median_mean_cluster_fraction_size_lt_10":float(np.median([r["mean_cluster_fraction_size_lt_10"] for r in cr])),"median_relative_inertia_reduction":float(np.median([r["relative_inertia_reduction"] for r in ir if r["relative_inertia_reduction"]!=""]))})
    write_csv(OUT/"candidate_range_summary.csv",ranges_out)
    plots(stab,inertia,cs)
    old_stab=read_csv(OLD/"kmeans_stability_summary.csv"); old_iner=read_csv(OLD/"kmeans_inertia_summary.csv")
    exact_stability=all(str(next(r for r in stab if r["k"]==int(o["k"]))[key])==o[key] for o in old_stab for key in ["mean_pairwise_ari","std_pairwise_ari","median_pairwise_ari","min_pairwise_ari","max_pairwise_ari","p10_pairwise_ari","p90_pairwise_ari"])
    exact_inertia=all(str(next(r for r in inertia if r["k"]==int(o["k"]))[key])==o[key] for o in old_iner for key in ["mean_inertia","std_inertia","median_inertia","min_inertia","max_inertia"])
    validation={"input_hash_before":EXPECTED_HASH,"input_hash_after":digest(INPUT),"input_unchanged":digest(INPUT)==EXPECTED_HASH,"old_k2_k20_runs_reused":True,"old_runs_recomputed":0,"old_stability_summaries_match_exactly":exact_stability,"old_inertia_summaries_match_exactly":exact_inertia,"new_nontrivial_fits_expected":7600,"new_nontrivial_fits_executed_this_invocation":fitted,"new_nontrivial_fits_resumed":skipped,"k1_sanity_fits":1,"new_run_rows":len(new_rows),"all_k21_k400_present":sorted(set(r["k"] for r in new_rows))==NEW_KS,"exactly_20_runs_each_k_ge_2":all(sum(r["k"]==k for r in all_runs)==20 for k in range(2,401)),"pairwise_ari_rows":len(pairwise),"pairwise_ari_rows_expected":75810,"exactly_190_ari_each_k_ge_2":all(sum(r["k"]==k for r in pairwise)==190 for k in range(2,401)),"all_inertia_finite":bool(np.isfinite([r["inertia"] for r in all_runs]+[m1.inertia_]).all()),"all_ari_finite":bool(np.isfinite([r["ari"] for r in pairwise]).all()),"all_clusterings_exactly_k_nonempty":all(r["n_nonempty_clusters"]==r["k"] for r in all_runs+[k1run]),"all_cluster_size_sums_n":all(r["cluster_size_sum"]==n for r in all_runs+[k1run]),"convergence_failure_count":sum(not r["converged"] for r in all_runs+[k1run]),"fit_runtime_seconds_this_invocation":fit_runtime,"early_knee_k1_k30":early_knee,"full_range_chord_knee_k1_k400":full_knee,"local_maxima_mean_ari":local_maxima(stab[1:],"mean_pairwise_ari"),"local_maxima_min_ari":local_maxima(stab[1:],"min_pairwise_ari"),"local_maxima_p10_ari":local_maxima(stab[1:],"p10_pairwise_ari")}
    dump(OUT/"validation.json",validation)
    dump(OUT/"experiment_config.json",{"experiment":"direct-coordinate K-means stability K=1..400 extension","input_path":str(INPUT.relative_to(ROOT)),"input_sha256":EXPECTED_HASH,"shape":list(shapes.shape),"flattened_dimension":400,"flatten_order":"NumPy C order, interleaved point x,y","fixed_x_dimensions_retained":True,"representation_changed":False,"pca":False,"gmm":False,"registration":False,"velocity_fields":False,"old_results_path":str(OLD.relative_to(ROOT)),"old_k_values_reused":OLD_KS,"new_k_values":NEW_KS,"k1_sanity_seed":0,"seeds":SEEDS,"kmeans":old_config["kmeans"],"resume":{"supported":True,"checkpoint_directory":"checkpoints/","skip_rule":"valid (20,1654) labels and seeds 0..19"},"pre_run_estimate":{"new_nontrivial_fits":7600,"additional_k1_sanity_fits":1,"runtime_minutes":"30-60","storage_megabytes":"30-80"},"ari":{"metric":"adjusted_rand_score","manual_label_alignment":False,"pairs_per_k_ge_2":190,"k1":"trivial and non-informative"},"tiny_cluster_definition":{"singleton":1,"lt5":5,"lt10":10,"lt1pct_n":.01*n},"software":{"python":platform.python_version(),"numpy":np.__version__,"sklearn":sklearn.__version__,"matplotlib":matplotlib.__version__},"diagnostic_knees":{"early_k1_k30":early_knee,"full_range_k1_k400":full_knee}})
    print(json.dumps(validation,indent=2),flush=True)

if __name__=="__main__": main()
