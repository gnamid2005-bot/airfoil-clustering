#!/usr/bin/env python3
"""Strictly reproduce the 400 missing original PCA-SVF GMM assignments.

No registration and no PCA refit are performed. The immutable PCA state and
scores from the original experiment are verified and reused.
"""
from __future__ import annotations

import csv, hashlib, json, os, time, warnings
from collections import Counter
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-gabriel-task2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             normalized_mutual_info_score)
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/gabriel_globalmean_field_pca_gmm"
KULFAN = ROOT / "outputs/gabriel_kulfan8_gmm_k30"
KULFAN_OLD = ROOT / "outputs/gabriel_kulfan8_gmm"
SELECT_META = ROOT / "outputs/gabriel_big_registration_k8_k11/config.json"
DATA = ROOT / "data/airfoils/processed_gabriel/airfoils_pinned_arclength_n200.npz"
OUT = ROOT / "outputs/gabriel_task2_assignment_consistency_refit"
KS, QS = (8, 9, 10, 11), tuple(range(1, 101))
SEED, N_INIT, REG_COVAR, MAX_ITER, TOL = 20260813, 5, 1e-7, 500, 1e-5
SELECTED = {8: 27, 9: 14, 10: 17, 11: 15}
DATA_SHA = "f357610c338b7a820c1f09522052d29951e9e62e41d3faffc61de19621995e33"

def sha_file(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def write_json(p,x):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,sort_keys=True),encoding="utf8")

def write_csv(p,rows,fields=None):
    rows=list(rows); p.parent.mkdir(parents=True,exist_ok=True)
    if not rows: raise ValueError(f"empty rows: {p}")
    fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    with p.open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,separators=(",",":")) if isinstance(v,(list,dict,tuple)) else v for k,v in r.items()})

def read_csv(p):
    with p.open(newline="",encoding="utf8") as f:return list(csv.DictReader(f))

def canonical(labels):
    m={};n=0;out=[]
    for x in map(int,labels):
        if x not in m:m[x]=n;n+=1
        out.append(m[x])
    return tuple(out)

def stable_ranges(M,threshold):
    # Maximal contiguous blocks for which every within-block pair meets threshold.
    out=[]; start=0
    while start<len(M):
        end=start
        while end+1<len(M) and np.all(M[start:end+2,start:end+2]>=threshold):end+=1
        if end>start:out.append([start+1,end+1])
        start=end+1
    return out

def load_and_audit():
    if sha_file(DATA)!=DATA_SHA:raise RuntimeError("dataset SHA mismatch")
    audit=json.loads((SOURCE/"audit.json").read_text()); val=json.loads((SOURCE/"validation.json").read_text())
    if audit["pca_input"]!="REGISTRATION-DERIVED B-SPLINE COEFFICIENTS ONLY (never contour coordinates)":raise RuntimeError("PCA input mismatch")
    expected={"covariance_type":"full","n_init":5,"reg_covar":1e-7,"max_iter":500,"tol":1e-5,"init_params":"kmeans","random_state":SEED}
    if any(audit["gmm"][k]!=v for k,v in expected.items()):raise RuntimeError("GMM config mismatch")
    with np.load(DATA,allow_pickle=False) as z: canonical_names=z["filenames"].astype(str)
    with np.load(SOURCE/"coefficients/registration_coefficients.npz",allow_pickle=False) as z:
        X=z["X_field"].copy();idx=z["target_indices"].copy();names=z["filenames"].astype(str)
    if X.shape!=(1651,1800) or not np.array_equal(names,canonical_names[idx]) or not np.all(np.diff(idx)>0):raise RuntimeError("coefficient identity/order mismatch")
    missing=np.setdiff1d(np.arange(1652),idx)
    if not np.array_equal(missing,[513]) or canonical_names[513]!="fx79w660a.dat":raise RuntimeError("unexpected missing sample")
    with np.load(SOURCE/"pca/pca_q100.npz",allow_pickle=False) as z:pca={k:z[k].copy() for k in z.files}
    train_global=np.load(SOURCE/"splits/train_indices_valid_used.npy");held_global=np.load(SOURCE/"splits/heldout_indices_valid_used.npy")
    if not np.array_equal(train_global,pca["train_target_indices"]) or not np.array_equal(held_global,pca["heldout_target_indices"]):raise RuntimeError("split/PCA mismatch")
    pos={int(v):i for i,v in enumerate(idx)};train=np.array([pos[int(v)] for v in train_global]);held=np.array([pos[int(v)] for v in held_global])
    scores=(X-pca["mean"])@pca["components"].T
    train_diff=float(np.max(np.abs(scores[train]-pca["train_scores"])))
    held_diff=float(np.max(np.abs(scores[held]-pca["heldout_scores"])))
    if train_diff>5e-13 or held_diff>5e-13:raise RuntimeError(f"PCA score mismatch {train_diff} {held_diff}")
    # The original GMM grid consumed these exact cached arrays. Preserve those
    # exact bits for fitting/prediction; the matrix-product audit above verifies
    # that the all-sample reconstruction is otherwise compatible.
    scores[train]=pca["train_scores"];scores[held]=pca["heldout_scores"]
    selected_cfg=json.loads(SELECT_META.read_text())
    if {int(k):int(q) for k,q in selected_cfg["pairs"]}!=SELECTED:raise RuntimeError("selected q mismatch")
    grid={(int(r["K"]),int(r["q"])):r for r in read_csv(SOURCE/"gmm/full_model_selection_grid.csv")}
    pref={int(r["q"]):r for r in read_csv(SOURCE/"gmm/preferred_k_by_q.csv")}
    return X,idx,names,canonical_names,scores,train,held,pca,grid,pref,train_diff,held_diff,audit,val

def fit_all(state):
    X,idx,names,canonical_names,scores,train,held,pca,grid,pref,train_diff,held_diff,audit,val=state
    validation=[]; started=time.perf_counter()
    for K in KS:
        labels=[];weights=[];resp=[];iters=[];converged=[];bic=[];icl=[];trainll=[];heldll=[];warnings_text=[]
        for q in QS:
            tick=time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                gm=GaussianMixture(n_components=K,covariance_type="full",n_init=N_INIT,reg_covar=REG_COVAR,max_iter=MAX_ITER,tol=TOL,init_params="kmeans",random_state=SEED).fit(scores[train,:q])
            tr_resp=gm.predict_proba(scores[train,:q]);all_resp=gm.predict_proba(scores[:,:q]);lab=all_resp.argmax(axis=1)
            entropy=float(-np.sum(tr_resp*np.log(np.clip(tr_resp,np.finfo(float).tiny,1))))
            b=float(gm.bic(scores[train,:q]));ic=b+2*entropy;h=float(gm.score(scores[held,:q]));tl=float(gm.score(scores[train,:q]))
            old=grid[(K,q)];ob=float(old["BIC"]);oi=float(old["ICL"]);oh=float(old["heldout_log_likelihood_per_sample"])
            validation.append({"K":K,"q":q,"original_BIC":ob,"reproduced_BIC":b,"abs_diff_BIC":abs(b-ob),"original_ICL":oi,"reproduced_ICL":ic,"abs_diff_ICL":abs(ic-oi),"original_heldout_LL_per_sample":oh,"reproduced_heldout_LL_per_sample":h,"abs_diff_heldout_LL":abs(h-oh),"converged":bool(gm.converged_),"iterations":int(gm.n_iter_),"runtime_seconds":time.perf_counter()-tick})
            labels.append(lab.astype(np.int16));weights.append(gm.weights_);resp.append(all_resp.astype(np.float32));iters.append(gm.n_iter_);converged.append(gm.converged_);bic.append(b);icl.append(ic);trainll.append(tl);heldll.append(h);warnings_text.append(" | ".join(str(w.message) for w in caught))
            print(f"fit K={K} q={q} BICdiff={abs(b-ob):.3g} ICLdiff={abs(ic-oi):.3g} LLdiff={abs(h-oh):.3g}",flush=True)
        np.savez_compressed(OUT/f"assignments/pca_svf_assignments_K{K:02d}.npz",assignments=np.asarray(labels),responsibilities=np.asarray(resp),max_responsibility=np.max(resp,axis=2),weights=np.asarray(weights),q=np.asarray(QS),K=np.asarray(K),target_indices=idx,filenames=names,random_state=np.asarray(SEED),converged=converged,iterations=iters,BIC=bic,ICL=icl,train_log_likelihood_per_sample=trainll,heldout_log_likelihood_per_sample=heldll,cluster_sizes=np.asarray([np.bincount(x,minlength=K) for x in labels]),warnings=np.asarray(warnings_text))
    write_csv(OUT/"refit_validation/model_metric_reproduction.csv",validation)
    maxima={k:max(r[k] for r in validation) for k in ("abs_diff_BIC","abs_diff_ICL","abs_diff_heldout_LL")}
    # Original script itself used rtol=2e-12, atol=2e-7 downstream; apply the same justified gate.
    ok=all(np.isclose(r["reproduced_BIC"],r["original_BIC"],rtol=2e-12,atol=2e-7) and np.isclose(r["reproduced_ICL"],r["original_ICL"],rtol=2e-12,atol=2e-7) and np.isclose(r["reproduced_heldout_LL_per_sample"],r["original_heldout_LL_per_sample"],rtol=2e-12,atol=2e-7) for r in validation)
    if not ok:raise RuntimeError(f"metric reproduction gate failed: {maxima}")
    return time.perf_counter()-started,validation,maxima

def repair_single_sensitive_cell_and_resume(state):
    """Replace the sole diagnostic-local-optimum cell using exact cached scores."""
    X,idx,names,canonical_names,scores,train,held,pca,grid,pref,train_diff,held_diff,audit,val=state
    K,q=11,1;tick=time.perf_counter()
    gm=GaussianMixture(n_components=K,covariance_type="full",n_init=N_INIT,reg_covar=REG_COVAR,max_iter=MAX_ITER,tol=TOL,init_params="kmeans",random_state=SEED).fit(scores[train,:q])
    tr=gm.predict_proba(scores[train,:q]);allr=gm.predict_proba(scores[:,:q]);entropy=float(-np.sum(tr*np.log(np.clip(tr,np.finfo(float).tiny,1))))
    b=float(gm.bic(scores[train,:q]));ic=b+2*entropy;h=float(gm.score(scores[held,:q]));tl=float(gm.score(scores[train,:q]));old=grid[(K,q)]
    zpath=OUT/f"assignments/pca_svf_assignments_K{K:02d}.npz";z=np.load(zpath,allow_pickle=False);d={k:z[k].copy() for k in z.files};z.close()
    d["assignments"][0]=allr.argmax(1);d["responsibilities"][0]=allr.astype(np.float32);d["max_responsibility"][0]=allr.max(1);d["weights"][0]=gm.weights_;d["converged"][0]=gm.converged_;d["iterations"][0]=gm.n_iter_;d["BIC"][0]=b;d["ICL"][0]=ic;d["train_log_likelihood_per_sample"][0]=tl;d["heldout_log_likelihood_per_sample"][0]=h;d["cluster_sizes"][0]=np.bincount(allr.argmax(1),minlength=K)
    np.savez_compressed(zpath,**d)
    rows=read_csv(OUT/"refit_validation/model_metric_reproduction.csv")
    for r in rows:
        if int(r["K"])==K and int(r["q"])==q:r.update({"reproduced_BIC":b,"abs_diff_BIC":abs(b-float(old["BIC"])),"reproduced_ICL":ic,"abs_diff_ICL":abs(ic-float(old["ICL"])),"reproduced_heldout_LL_per_sample":h,"abs_diff_heldout_LL":abs(h-float(old["heldout_log_likelihood_per_sample"])),"converged":bool(gm.converged_),"iterations":int(gm.n_iter_),"runtime_seconds":time.perf_counter()-tick})
    write_csv(OUT/"refit_validation/model_metric_reproduction.csv",rows)
    numeric=[]
    for r in rows:
        rr=dict(r)
        for k in ("original_BIC","reproduced_BIC","abs_diff_BIC","original_ICL","reproduced_ICL","abs_diff_ICL","original_heldout_LL_per_sample","reproduced_heldout_LL_per_sample","abs_diff_heldout_LL","runtime_seconds"):rr[k]=float(rr[k])
        rr["K"],rr["q"],rr["iterations"]=int(rr["K"]),int(rr["q"]),int(rr["iterations"]);rr["converged"]=str(rr["converged"]).lower()=="true";numeric.append(rr)
    maxima={k:max(r[k] for r in numeric) for k in ("abs_diff_BIC","abs_diff_ICL","abs_diff_heldout_LL")}
    ok=all(np.isclose(r["reproduced_BIC"],r["original_BIC"],rtol=2e-12,atol=2e-7) and np.isclose(r["reproduced_ICL"],r["original_ICL"],rtol=2e-12,atol=2e-7) and np.isclose(r["reproduced_heldout_LL_per_sample"],r["original_heldout_LL_per_sample"],rtol=2e-12,atol=2e-7) for r in numeric)
    if not ok:raise RuntimeError(f"repair gate failed {maxima}")
    return float(sum(r["runtime_seconds"] for r in numeric)),numeric,maxima

def analyse(state,fit_runtime,validation,maxima):
    X,idx,names,canonical_names,scores,train,held,pca,grid,pref,train_diff,held_diff,audit,val=state
    kulfan_rows=[];stability_rows=[];q_rows=[];adj=[];selref=[];cross=[];selcomp=[];sample_rows=[];contingencies={}
    pair_mats={};labs_by_k={};kul_by_k={};summary_by_k={}
    for K in KS:
        z=np.load(OUT/f"assignments/pca_svf_assignments_K{K:02d}.npz",allow_pickle=False); labs=z["assignments"].astype(int);labs_by_k[K]=labs
        M=np.eye(100)
        for a in range(100):
            for b in range(a+1,100):M[a,b]=M[b,a]=adjusted_rand_score(labs[a],labs[b])
        pair_mats[K]=M;write_csv(OUT/f"pca_svf/pairwise_ari_K{K:02d}.csv",[{"q":q,**{f"q{j:03d}":M[q-1,j-1] for j in QS}} for q in QS])
        pair=M[np.triu_indices(100,1)];means=(M.sum(1)-1)/99;qmed=int(np.argmax(means)+1);sq=SELECTED[K];ref=M[sq-1]
        sr={"K":K,"mean_pairwise_ARI":float(pair.mean()),"median_pairwise_ARI":float(np.median(pair)),"minimum_pairwise_ARI":float(pair.min()),"maximum_pairwise_ARI":float(pair.max()),"q_medoid":qmed,"q_medoid_mean_ARI":float(means[qmed-1]),"selected_q":sq,"selected_q_mean_ARI_to_other_q":float((ref.sum()-1)/99),"selected_q_median_ARI_to_other_q":float(np.median(np.delete(ref,sq-1))),"selected_q_minimum_ARI":float(np.min(np.delete(ref,sq-1))),"selected_q_maximum_ARI":float(np.max(np.delete(ref,sq-1))),"stable_ranges_all_pairwise_ge_0_9":stable_ranges(M,.9),"stable_ranges_all_pairwise_ge_0_8":stable_ranges(M,.8)}
        stability_rows.append(sr);summary_by_k[K]=sr
        for q in QS:
            others=np.delete(M[q-1],q-1);q_rows.append({"K":K,"q":q,"mean_ARI_to_other_q":float(others.mean()),"median_ARI_to_other_q":float(np.median(others)),"minimum_ARI_to_other_q":float(others.min()),"maximum_ARI_to_other_q":float(others.max()),"is_q_medoid":q==qmed,"is_selected_q":q==sq})
            selref.append({"K":K,"q":q,"selected_q":sq,"ARI_to_selected_q":float(ref[q-1])})
            for lag in (1,2,5):
                if q+lag<=100:adj.append({"K":K,"q":q,"q_lag":q+lag,"lag":lag,"ARI":float(M[q-1,q+lag-1]),"BIC_preferred_K_at_q":int(pref[q]["preferred_k_bic"]),"BIC_preferred_K_at_q_lag":int(pref[q+lag]["preferred_k_bic"]),"ICL_preferred_K_at_q":int(pref[q]["preferred_k_icl"]),"ICL_preferred_K_at_q_lag":int(pref[q+lag]["preferred_k_icl"]),"BIC_preferred_K_changed":int(pref[q]["preferred_k_bic"])!=int(pref[q+lag]["preferred_k_bic"]),"ICL_preferred_K_changed":int(pref[q]["preferred_k_icl"])!=int(pref[q+lag]["preferred_k_icl"])})
        stab=np.load(KULFAN/f"checkpoints/stability_K{K:02d}.npz",allow_pickle=False)["labels"]
        stab_summary=next(r for r in read_csv(KULFAN/"tables/seed_stability_k1_k30.csv") if int(r["K"])==K)
        prod=(KULFAN_OLD if K<=9 else KULFAN)/f"gmm/assignments/K{K:02d}_assignments.npy";kul=np.load(prod).astype(int)
        if len(kul)!=1652 or not np.array_equal(canonical_names[idx],names):raise RuntimeError("common identity failure")
        kc=kul[idx];kul_by_k[K]=kc
        kulfan_rows.extend({"K":K,"target_index":int(t),"filename":str(n),"label":int(l)} for t,n,l in zip(idx,names,kc))
        for q in QS:cross.append({"K":K,"q":q,"selected_q":sq,"ARI":adjusted_rand_score(labs[q-1],kc),"AMI":adjusted_mutual_info_score(labs[q-1],kc),"NMI":normalized_mutual_info_score(labs[q-1],kc)})
        cr=[r for r in cross if r["K"]==K];best=max(cr,key=lambda r:r["ARI"]);at=cr[sq-1]
        A=labs[sq-1];C=np.zeros((K,K),int)
        for a,b in zip(A,kc):C[a,b]+=1
        rr,cc=linear_sum_assignment(-C);mapping={int(a):int(b) for a,b in zip(rr,cc)};matched=np.array([mapping[int(a)]==int(b) for a,b in zip(A,kc)])
        reordered=C[:,cc];row_norm=np.divide(reordered,reordered.sum(1,keepdims=True),out=np.zeros_like(reordered,dtype=float),where=reordered.sum(1,keepdims=True)>0)
        contingencies[K]=(C,reordered,row_norm,cc)
        np.savetxt(OUT/f"comparison/selected_q_contingency_K{K:02d}.csv",C,delimiter=",",fmt="%d",header=",".join(f"Kulfan_{j}" for j in range(K)),comments="")
        selcomp.append({"K":K,"selected_q":sq,"ARI":at["ARI"],"AMI":at["AMI"],"NMI":at["NMI"],"maximum_SVF_vs_Kulfan_ARI":best["ARI"],"q_of_maximum":best["q"],"difference_max_minus_selected":best["ARI"]-at["ARI"],"mean_ARI_across_q":float(np.mean([r["ARI"] for r in cr])),"median_ARI_across_q":float(np.median([r["ARI"] for r in cr])),"matched_count":int(matched.sum()),"matched_fraction":float(matched.mean()),"unmatched_fraction":float(1-matched.mean()),"hungarian_mapping":mapping,"matched_cluster_contributions":{int(a):int(C[a,mapping[a]]) for a in mapping},"kulfan_mean_seed_ARI":float(stab_summary["mean_pairwise_ARI"]),"kulfan_modal_frequency":int(stab_summary["modal_exact_partition_frequency"]),"kulfan_ARI_medoid_seed":int(stab_summary["ARI_medoid_seed"])})
        maxresp=z["max_responsibility"][sq-1]
        for i,(t,n,a,b,m,conf) in enumerate(zip(idx,names,A,kc,matched,maxresp)):sample_rows.append({"K":K,"selected_q":sq,"target_index":int(t),"filename":str(n),"svf_label":int(a),"kulfan_label":int(b),"mapped_kulfan_label_for_svf":mapping[int(a)],"matched":bool(m),"svf_max_responsibility":float(conf)})
    write_csv(OUT/"pca_svf/adjacent_q_ari.csv",adj);write_csv(OUT/"pca_svf/selected_q_reference_ari.csv",selref);write_csv(OUT/"pca_svf/q_stability_summary.csv",q_rows);write_csv(OUT/"pca_svf/fixed_k_summary.csv",stability_rows)
    write_csv(OUT/"kulfan/kulfan_reference_assignments.csv",kulfan_rows);write_csv(OUT/"kulfan/kulfan_stability_summary.csv",[{"K":K,**{k:v for k,v in next(r for r in read_csv(KULFAN/"tables/seed_stability_k1_k30.csv") if int(r["K"])==K).items() if k!="K"}} for K in KS])
    write_csv(OUT/"comparison/svf_vs_kulfan_ari.csv",cross);write_csv(OUT/"comparison/selected_q_comparison.csv",selcomp);write_csv(OUT/"comparison/sample_level_agreement.csv",sample_rows)
    make_plots(pair_mats,adj,selref,cross,contingencies,stability_rows,selcomp)
    runtime=sum(float(r["runtime_seconds"]) for r in validation)
    config={"task":"Gabriel Task 2 strict assignment reproduction","source":str(SOURCE.relative_to(ROOT)),"Kulfan_source":str(KULFAN.relative_to(ROOT)),"K":list(KS),"q":[1,100],"GMM_cells_refitted":400,"population":{"PCA_SVF_N":1651,"Kulfan_N":1652,"comparison_N":1651,"removed_from_Kulfan":{"target_index":513,"filename":"fx79w660a.dat"}},"PCA":{"reused_saved_state":True,"refit":False,"fit_scope":"training only","solver":"full","centered":True,"whiten":False,"n_components":100,"max_train_score_abs_diff":train_diff,"max_heldout_score_abs_diff":held_diff},"GMM":{"covariance_type":"full","n_init":5,"reg_covar":1e-7,"max_iter":500,"tol":1e-5,"init_params":"kmeans","random_state":SEED,"BIC":"sklearn GaussianMixture.bic on training scores","ICL":"BIC + 2 * positive training posterior classification entropy","heldout_LL":"GaussianMixture.score: mean per held-out sample"},"selected_q":SELECTED,"metric_reproduction_tolerance":{"rtol":2e-12,"atol":2e-7},"fit_runtime_wall_seconds":fit_runtime,"sum_cell_runtime_seconds":runtime,"registration_run":False,"original_outputs_modified":False,"big_registration_assignments_used":False,"commit_or_push":False};write_json(OUT/"config.json",config)
    validation_json={"status":"PASS","GMM_cells_expected":400,"GMM_cells_completed":len(validation),"all_converged":all(r["converged"] for r in validation),"maximum_metric_absolute_differences":maxima,"metric_gate_passed":True,"common_sample_count":1651,"common_filenames_exact_order":True,"excluded_Kulfan_index":513,"excluded_Kulfan_filename":"fx79w660a.dat","registration_run":False,"PCA_refit":False,"original_outputs_modified":False,"big_registration_assignment_source_used":False};write_json(OUT/"validation.json",validation_json)
    make_report(config,stability_rows,selcomp,adj)

def make_plots(mats,adj,selref,cross,conts,stability,selcomp):
    figdir=OUT/"figures";figdir.mkdir(parents=True,exist_ok=True)
    for K in KS:
        sq=SELECTED[K];M=mats[K]
        fig,ax=plt.subplots(figsize=(8,7));im=ax.imshow(M,vmin=0,vmax=1,origin="lower",extent=(.5,100.5,.5,100.5),cmap="viridis",aspect="auto");ax.axvline(sq,color="red",ls="--",lw=1);ax.axhline(sq,color="red",ls="--",lw=1);ax.set(xlabel="PCA dimension q",ylabel="PCA dimension q",title=f"PCA-SVF pairwise ARI, fixed K={K}");fig.colorbar(im,ax=ax,label="ARI");fig.tight_layout();fig.savefig(figdir/f"pca_svf_pairwise_ari_heatmap_K{K:02d}.png",dpi=240);plt.close(fig)
        for rows,ykey,title,name in [([r for r in adj if r["K"]==K and r["lag"]==1],"ARI",f"Adjacent-q PCA-SVF consistency, K={K}",f"adjacent_q_ari_K{K:02d}.png"),([r for r in selref if r["K"]==K],"ARI_to_selected_q",f"ARI to Gabriel selected q={sq}, K={K}",f"ari_to_selected_q_K{K:02d}.png"),([r for r in cross if r["K"]==K],"ARI",f"PCA-SVF vs direct Kulfan, K={K}",f"svf_vs_kulfan_ari_K{K:02d}.png")]:
            fig,ax=plt.subplots(figsize=(9,4.8));ax.plot([r["q"] for r in rows],[r[ykey] for r in rows],color="#2563eb",lw=1.5);ax.axvline(sq,color="#dc2626",ls="--",label=f"selected q={sq}");ax.set(xlabel="PCA dimension q",ylabel="ARI",title=title,ylim=(-.02,1.02));ax.grid(alpha=.22);ax.legend();fig.tight_layout();fig.savefig(figdir/name,dpi=240);plt.close(fig)
        C,R,N,cc=conts[K];fig,axs=plt.subplots(1,3,figsize=(15,4.5));
        for ax,A,t,cmap in zip(axs,(C,R,N),("Raw contingency","Hungarian reordered","Row-normalised overlap"),("Blues","Blues","magma")):
            im=ax.imshow(A,cmap=cmap,aspect="auto");ax.set(title=t,xlabel="Kulfan cluster",ylabel="SVF cluster");fig.colorbar(im,ax=ax,fraction=.046)
        fig.suptitle(f"Selected-q cluster comparison K={K}, q={sq}");fig.tight_layout();fig.savefig(figdir/f"selected_q_contingency_K{K:02d}.png",dpi=240);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));ax.boxplot([mats[K][np.triu_indices(100,1)] for K in KS],tick_labels=[str(k) for k in KS]);ax.set(xlabel="Fixed K",ylabel="Pairwise ARI across q",title="Fixed-K PCA-SVF assignment stability");ax.grid(axis="y",alpha=.22);fig.tight_layout();fig.savefig(figdir/"fixed_k_stability_summary.png",dpi=240);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));
    for K in KS:
        rr=[r for r in cross if r["K"]==K];ax.plot(QS,[r["ARI"] for r in rr],label=f"K={K}")
    ax.set(xlabel="PCA dimension q",ylabel="ARI vs Kulfan",title="PCA-SVF vs Kulfan across q");ax.grid(alpha=.22);ax.legend();fig.tight_layout();fig.savefig(figdir/"svf_vs_kulfan_summary.png",dpi=240);plt.close(fig)

def make_report(config,stability,selcomp,adj):
    lines=["# Gabriel Task 2：聚类 assignment 一致性", "", "严格复现了原始 PCA-SVF 方法的 400 个目标 GMM 单元；未运行配准，未重拟合 PCA。SVF-vs-Kulfan comparisons were performed on the 1,651 airfoils common to both experiments.","","## 数值复现","",f"- 拟合 wall time：{config['fit_runtime_wall_seconds']:.3f} s",f"- 逐单元累计时间：{config['sum_cell_runtime_seconds']:.3f} s",f"- 数值容差：rtol={config['metric_reproduction_tolerance']['rtol']}, atol={config['metric_reproduction_tolerance']['atol']}","","## 结果","","| K | mean pairwise ARI | median | q medoid | selected q | selected mean ARI | Kulfan seed mean ARI | max cross ARI (q) | selected cross ARI | Hungarian matched |","|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s,c in zip(stability,selcomp):lines.append(f"| {s['K']} | {s['mean_pairwise_ARI']:.4f} | {s['median_pairwise_ARI']:.4f} | {s['q_medoid']} | {s['selected_q']} | {s['selected_q_mean_ARI_to_other_q']:.4f} | {c['kulfan_mean_seed_ARI']:.4f} | {c['maximum_SVF_vs_Kulfan_ARI']:.4f} ({c['q_of_maximum']}) | {c['ARI']:.4f} | {c['matched_fraction']:.2%} |")
    lines += ["","## 稳定区（描述性阈值）",""]
    for s in stability:lines.append(f"- K={s['K']}: all-pairwise ARI≥0.9: `{s['stable_ranges_all_pairwise_ge_0_9']}`; ≥0.8: `{s['stable_ranges_all_pairwise_ge_0_8']}`.")
    lines += ["","ARI 是主指标；Hungarian 匹配仅用于簇级解释。BIC/ICL 与 adjacent-q ARI 的逐 q 联合记录见 `pca_svf/adjacent_q_ari.csv`。","","未运行 registration，未使用 big-registration assignments，未修改原始科学输出，未 commit/push。"]
    (OUT/"report").mkdir(parents=True,exist_ok=True);(OUT/"report/final_report.md").write_text("\n".join(lines)+"\n",encoding="utf8")

def main():
    for d in ("refit_validation","assignments","pca_svf","kulfan","comparison","figures","report"):(OUT/d).mkdir(parents=True,exist_ok=True)
    state=load_and_audit()
    existing=all((OUT/f"assignments/pca_svf_assignments_K{K:02d}.npz").exists() for K in KS) and (OUT/"refit_validation/model_metric_reproduction.csv").exists()
    if existing:fit_runtime,validation,maxima=repair_single_sensitive_cell_and_resume(state)
    else:fit_runtime,validation,maxima=fit_all(state)
    analyse(state,fit_runtime,validation,maxima)

if __name__=="__main__":main()
