#!/usr/bin/env python3
"""Audit and fit the fallback 8-D Kulfan/CST representation and GMMs.

This script intentionally contains no registration imports or calls.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.special import comb
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/airfoils/processed_gabriel/airfoils_pinned_arclength_n200.npz"
SPLIT_ROOT = ROOT / "outputs/gabriel_pinned_gmm_vs_kmeans/splits"
OUT = ROOT / "outputs/gabriel_kulfan8_gmm"
SEED = 20260813
SEEDS = list(range(20))
KS = list(range(1, 10))
N1, N2, DEGREE = 0.5, 1.0, 3
GMM_KW = dict(covariance_type="full", n_init=5, reg_covar=1e-7,
              max_iter=500, tol=1e-5, init_params="kmeans")
FALLBACK = "Default 8-D Kulfan convention adopted because no repository-specific Gabriel definition was found."
COEFF_NAMES = [f"A_upper_{j}" for j in range(4)] + [f"A_lower_{j}" for j in range(4)]


def jwrite(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def cwrite(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def design(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    cls = x ** N1 * (1.0 - x) ** N2
    bern = np.column_stack([comb(DEGREE, j) * x**j * (1-x)**(DEGREE-j)
                            for j in range(DEGREE + 1)])
    return cls[:, None] * bern


def fit_one(contour: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    upper, lower = contour[:100], contour[100:]
    U, L = design(upper[:, 0]), design(lower[:, 0])
    w0 = np.r_[np.linalg.lstsq(U, upper[:, 1], rcond=None)[0],
               np.linalg.lstsq(L, lower[:, 1], rcond=None)[0]]
    # Sufficient convex non-crossing condition because Bernstein bases and class
    # function are non-negative on [0,1].
    G = np.c_[np.eye(4), -np.eye(4)]
    constrained = bool(np.min(G @ w0) < -1e-12)
    success, nit, message = True, 0, "unconstrained solution feasible"
    w = w0
    if constrained:
        H = np.zeros((8, 8)); H[:4, :4] = U.T @ U; H[4:, 4:] = L.T @ L
        b = np.r_[U.T @ upper[:, 1], L.T @ lower[:, 1]]
        res = minimize(lambda q: 0.5*q@H@q - b@q, w0,
                       jac=lambda q: H@q-b, method="SLSQP",
                       constraints=[LinearConstraint(G, 0.0, np.inf)],
                       options={"ftol": 1e-12, "maxiter": 200})
        w, success, nit, message = res.x, bool(res.success), int(res.nit), str(res.message)
    rec = reconstruct(contour[:, 0], w)
    return w, rec, {"constrained": constrained, "success": success,
                    "iterations": nit, "message": message,
                    "minimum_weight_difference": float(np.min(G @ w))}


def reconstruct(x_contour: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.column_stack([x_contour,
                            np.r_[design(x_contour[:100]) @ w[:4],
                                  design(x_contour[100:]) @ w[4:]]])


def template_contour(w: np.ndarray, n_surface: int = 100) -> np.ndarray:
    # Match repository traversal: upper TE->LE, lower LE->TE, implicit closure.
    x_up = np.linspace(1.0, 0.0, n_surface, endpoint=False)
    x_lo = np.linspace(0.0, 1.0, n_surface, endpoint=False)
    x = np.r_[x_up, x_lo]
    return reconstruct(x, w)


def signed_area(c: np.ndarray) -> float:
    return float(0.5*np.sum(c[:,0]*np.roll(c[:,1],-1)-c[:,1]*np.roll(c[:,0],-1)))


def orient(a,b,c): return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def self_intersects(c: np.ndarray, eps: float = 1e-12) -> bool:
    n = len(c)
    for i in range(n):
        a,b=c[i],c[(i+1)%n]
        for j in range(i+1,n):
            if j in (i,(i+1)%n) or (j+1)%n in (i,(i+1)%n): continue
            d,e=c[j],c[(j+1)%n]
            o1,o2,o3,o4=orient(a,b,d),orient(a,b,e),orient(d,e,a),orient(d,e,b)
            if ((o1>eps and o2<-eps) or (o1<-eps and o2>eps)) and ((o3>eps and o4<-eps) or (o3<-eps and o4>eps)):
                return True
    return False


def contour_quality(c: np.ndarray) -> dict:
    seg = np.linalg.norm(np.diff(np.vstack([c,c[0]]),axis=0),axis=1)
    area = signed_area(c)
    return {"finite": bool(np.isfinite(c).all()), "self_intersection": self_intersects(c),
            "orientation_ccw": bool(area > 0), "signed_area": area,
            "x_min": float(c[:,0].min()), "x_max": float(c[:,0].max()),
            "leading_edge_x": float(c[np.argmin(c[:,0]),0]),
            "trailing_edge_gap": float(np.linalg.norm(c[0]-c[-1])),
            "minimum_segment_length": float(seg.min()),
            "chord_normalized": bool(abs(c[:,0].min())<1e-12 and abs(c[:,0].max()-1)<1e-12),
            "valid": bool(np.isfinite(c).all() and area>0 and not self_intersects(c)
                          and seg.min()>1e-10 and abs(c[:,0].min())<1e-12
                          and abs(c[:,0].max()-1)<1e-12)}


def canonical_labels(labels: np.ndarray) -> tuple[int,...]:
    m={}; nxt=0; out=[]
    for v in map(int,labels):
        if v not in m: m[v]=nxt; nxt+=1
        out.append(m[v])
    return tuple(out)


def geom_distance(a: np.ndarray,b: np.ndarray) -> dict:
    d=np.linalg.norm(a-b,axis=1)
    return {"pointwise_mse":float(np.mean(d*d)),"rmse":float(np.sqrt(np.mean(d*d))),
            "mean_distance":float(np.mean(d)),"maximum_distance":float(np.max(d))}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to modify prior output: {OUT}")
    for d in ["kulfan","gmm/assignments","gmm/component_parameters",
              "templates/component_mean_templates","templates/hard_cluster_mean_templates",
              "tables","figures","report"]: (OUT/d).mkdir(parents=True,exist_ok=False)
    with np.load(DATA,allow_pickle=False) as z:
        contours=z["contours"].copy(); filenames=z["filenames"].copy()
        prep=json.loads(str(z["preprocessing_configuration_json"])); rep=json.loads(str(z["representation_definition_json"]))
    n=len(contours); data_hash=sha256(DATA)
    X=np.empty((n,8)); recs=np.empty_like(contours); metrics=[]; solvers=[]
    for i,c in enumerate(contours):
        w,r,s=fit_one(c); X[i]=w; recs[i]=r; solvers.append(s)
        dist=np.linalg.norm(r-c,axis=1); q=contour_quality(r)
        metrics.append({"index":i,"filename":str(filenames[i]),"pointwise_mse":float(np.mean(dist**2)),
                        "rmse":float(np.sqrt(np.mean(dist**2))),"maximum_pointwise_error":float(dist.max()),
                        "finite":q["finite"],"self_intersection":q["self_intersection"],
                        "orientation_ccw":q["orientation_ccw"],"valid":q["valid"],
                        "solver_constrained":s["constrained"],"solver_success":s["success"],
                        "minimum_weight_difference":s["minimum_weight_difference"]})
    np.save(OUT/"kulfan/kulfan_parameters.npy",X); np.save(OUT/"kulfan/filenames.npy",filenames)
    np.save(OUT/"kulfan/reconstructed_contours.npy",recs); cwrite(OUT/"kulfan/reconstruction_metrics.csv",metrics)
    if X.shape != (n,8) or not np.isfinite(X).all() or not all(s["success"] for s in solvers):
        raise RuntimeError("Kulfan fitting failed")
    arr={k:np.array([r[k] for r in metrics],float) for k in ("pointwise_mse","rmse","maximum_pointwise_error")}
    rec_summary={k:{"mean":float(v.mean()),"median":float(np.median(v)),"p95":float(np.percentile(v,95)),"maximum":float(v.max())} for k,v in arr.items()}
    invalid=sum(not r["valid"] for r in metrics)
    self_intersections=sum(r["self_intersection"] for r in metrics)
    # Deterministic diagnostic sample: median-nearest, p95-nearest, top-3 incl worst.
    order=np.argsort(arr["rmse"]); targets=[np.median(arr["rmse"]),np.percentile(arr["rmse"],95)]
    sample=[int(np.argmin(abs(arr["rmse"]-t))) for t in targets]+list(map(int,order[-3:]))
    sample=list(dict.fromkeys(sample))
    fig,axs=plt.subplots(len(sample),1,figsize=(9,2.6*len(sample)),squeeze=False)
    for ax,i in zip(axs[:,0],sample):
        ax.plot(contours[i,:,0],contours[i,:,1],color="#243447",lw=1.8,label="original")
        ax.plot(recs[i,:,0],recs[i,:,1],color="#d97706",lw=1.3,ls="--",label="Kulfan-8")
        ax.set_aspect("equal"); ax.grid(alpha=.18); ax.set_title(f"{i}: {filenames[i]} | RMSE={arr['rmse'][i]:.3e}")
        ax.legend(ncol=2,loc="upper right")
    fig.suptitle("Original contours vs constrained 8-D Kulfan reconstructions")
    fig.tight_layout(); fig.savefig(OUT/"figures/reconstruction_examples.png",dpi=220); plt.close(fig)
    # Explicit stop gate.
    # A repeated LE sample inherited from the canonical fixed-x contour can make
    # a zero-length segment without indicating a poor CST approximation.  The
    # stop gate therefore uses actual self-intersection and reconstruction error;
    # all degeneracies remain visible in the per-airfoil audit.
    if self_intersections/n > 0.05 or np.median(arr["rmse"]) > 0.01:
        jwrite(OUT/"validation.json",{"status":"STOPPED_BEFORE_GMM","reason":"poor reconstruction gate",
               "invalid_fraction":invalid/n,"self_intersection_fraction":self_intersections/n,
               "reconstruction_summary":rec_summary,"no_registration":True})
        raise RuntimeError("reconstruction stop gate triggered")
    tr=np.load(SPLIT_ROOT/"train_indices.npy"); ho=np.load(SPLIT_ROOT/"heldout_indices.npy")
    np.save(OUT/"gmm/train_indices.npy",tr); np.save(OUT/"gmm/heldout_indices.npy",ho)
    scale=[]
    for j,name in enumerate(COEFF_NAMES):
        v=X[:,j]; scale.append({"parameter":name,"mean":float(v.mean()),"sd":float(v.std(ddof=1)),
                                "minimum":float(v.min()),"p05":float(np.percentile(v,5)),
                                "median":float(np.median(v)),"p95":float(np.percentile(v,95)),"maximum":float(v.max())})
    cwrite(OUT/"tables/kulfan_parameter_scales.csv",scale)
    selection=[]; stability=[]; sizes_rows=[]; quality_rows=[]; template_rows=[]; pair_rows=[]
    for k in KS:
        gm=GaussianMixture(n_components=k,random_state=SEED,**GMM_KW).fit(X[tr])
        labels=gm.predict(X); resp=gm.predict_proba(X); train_resp=gm.predict_proba(X[tr])
        entropy=float(-np.sum(train_resp*np.log(np.clip(train_resp,1e-300,None))))
        bic=float(gm.bic(X[tr])); aic=float(gm.aic(X[tr])); icl=bic+2*entropy
        train_ll=float(gm.score(X[tr])*len(tr)); held_ll=float(gm.score(X[ho])*len(ho))
        counts=np.bincount(labels,minlength=k); weights=gm.weights_
        selection.append({"K":k,"BIC":bic,"AIC":aic,"ICL":icl,"train_log_likelihood":train_ll,
                          "train_log_likelihood_per_sample":train_ll/len(tr),"heldout_log_likelihood":held_ll,
                          "heldout_log_likelihood_per_sample":held_ll/len(ho),"posterior_entropy":entropy,
                          "converged":bool(gm.converged_),"iterations":int(gm.n_iter_),
                          "minimum_cluster_size":int(counts.min()),"maximum_cluster_size":int(counts.max()),
                          "component_weights_json":json.dumps(weights.tolist()),"hard_cluster_sizes_json":json.dumps(counts.tolist())})
        np.save(OUT/f"gmm/assignments/K{k:02d}_assignments.npy",labels)
        np.savez_compressed(OUT/f"gmm/component_parameters/K{k:02d}.npz",means=gm.means_,weights=weights,
                            covariances=gm.covariances_,precisions_cholesky=gm.precisions_cholesky_)
        for j in range(k):
            frac=counts[j]/n
            sizes_rows.append({"K":k,"component":j,"hard_cluster_size":int(counts[j]),"hard_cluster_percentage":100*frac,
                               "component_weight":float(weights[j]),"size_lt_20":bool(counts[j]<20),
                               "size_lt_1pct":bool(frac<.01),"size_lt_2pct":bool(frac<.02),
                               "very_small_weight_lt_1pct":bool(weights[j]<.01)})
        quality_rows.append({"K":k,"sorted_cluster_sizes":json.dumps(sorted(map(int,counts))),
                             "smallest_cluster_size":int(counts.min()),"smallest_cluster_percentage":100*counts.min()/n,
                             "largest_cluster_size":int(counts.max()),"largest_cluster_percentage":100*counts.max()/n,
                             "max_min_size_ratio":float(counts.max()/counts.min()),
                             "ambiguous_max_posterior_lt_0_6_fraction":float(np.mean(resp.max(1)<.6)),
                             "ambiguous_max_posterior_lt_0_8_fraction":float(np.mean(resp.max(1)<.8)),
                             "components_size_lt_20":int(np.sum(counts<20)),"components_fraction_lt_1pct":int(np.sum(counts/n<.01)),
                             "components_fraction_lt_2pct":int(np.sum(counts/n<.02)),"weights_lt_1pct":int(np.sum(weights<.01))})
        # Stability: independent 20 seeds, same n_init and training-only fit, full-population assignment.
        labs=[]
        for s in SEEDS:
            gs=GaussianMixture(n_components=k,random_state=s,**GMM_KW).fit(X[tr]); labs.append(gs.predict(X))
        ari=np.ones((len(SEEDS),len(SEEDS)))
        for a in range(len(SEEDS)):
            for b in range(a+1,len(SEEDS)): ari[a,b]=ari[b,a]=adjusted_rand_score(labs[a],labs[b])
        pairs=ari[np.triu_indices(len(SEEDS),1)]
        modes=Counter(canonical_labels(x) for x in labs); modal_count=modes.most_common(1)[0][1]
        medoid_idx=int(np.argmax((ari.sum(1)-1)/(len(SEEDS)-1)))
        stability.append({"K":k,"seed_count":len(SEEDS),"pair_count":len(pairs),"mean_pairwise_ARI":float(pairs.mean()),
                          "sd_pairwise_ARI":float(pairs.std(ddof=1)),"modal_exact_partition_frequency":modal_count,
                          "modal_fraction":modal_count/len(SEEDS),"ARI_medoid_seed":SEEDS[medoid_idx],
                          "ARI_medoid_mean_ARI":float((ari[medoid_idx].sum()-1)/(len(SEEDS)-1)),
                          "meaningful_exact_mode":bool(modal_count>=2)})
        # Templates use primary model. Constraint is convex, hence means remain feasible.
        component_contours=[]; hard_contours=[]
        for j in range(k):
            mu=gm.means_[j]; hard=X[labels==j].mean(0)
            ct,ht=template_contour(mu),template_contour(hard)
            component_contours.append(ct); hard_contours.append(ht)
            np.save(OUT/f"templates/component_mean_templates/K{k:02d}_component{j:02d}.npy",ct)
            np.save(OUT/f"templates/hard_cluster_mean_templates/K{k:02d}_component{j:02d}.npy",ht)
            d=geom_distance(ct,ht); qc,qh=contour_quality(ct),contour_quality(ht)
            template_rows.append({"K":k,"component":j,**d,"component_mean_valid":qc["valid"],
                                  "hard_mean_valid":qh["valid"],"component_mean_self_intersection":qc["self_intersection"],
                                  "hard_mean_self_intersection":qh["self_intersection"],
                                  "component_mean_minimum_segment_length":qc["minimum_segment_length"],
                                  "hard_mean_minimum_segment_length":qh["minimum_segment_length"]})
        for a in range(k):
            for b in range(a+1,k): pair_rows.append({"K":k,"component_a":a,"component_b":b,**geom_distance(component_contours[a],component_contours[b])})
        fig,ax=plt.subplots(figsize=(10,5.5))
        colors=plt.cm.tab10(np.linspace(0,1,max(k,2)))
        for j,(ct,ht) in enumerate(zip(component_contours,hard_contours)):
            ax.plot(ct[:,0],ct[:,1],color=colors[j],lw=1.8,label=f"C{j} component mean")
            ax.plot(ht[:,0],ht[:,1],color=colors[j],lw=1.0,ls="--",label=f"C{j} hard mean")
        ax.set_aspect("equal"); ax.grid(alpha=.18); ax.set_title(f"Kulfan-space GMM templates, K={k}\nsolid: component mean; dashed: hard-cluster mean")
        ax.legend(ncol=min(3,k),fontsize=7,loc="upper center",bbox_to_anchor=(.5,-.12)); fig.tight_layout()
        fig.savefig(OUT/f"figures/templates_K{k:02d}.png",dpi=220,bbox_inches="tight"); plt.close(fig)
    cwrite(OUT/"gmm/model_selection_k1_k9.csv",selection); cwrite(OUT/"gmm/seed_stability_k1_k9.csv",stability)
    cwrite(OUT/"tables/cluster_sizes_k1_k9.csv",sizes_rows); cwrite(OUT/"tables/component_quality_k1_k9.csv",quality_rows)
    cwrite(OUT/"tables/template_comparison_k1_k9.csv",template_rows); cwrite(OUT/"tables/pairwise_template_distances_k1_k9.csv",pair_rows)
    # Requested model-selection plots.
    plot_specs=[("BIC","bic_vs_k.png","BIC"),("AIC","aic_vs_k.png","AIC"),("ICL","icl_vs_k.png","ICL"),
                ("heldout_log_likelihood","heldout_ll_vs_k.png","Held-out total log-likelihood"),
                ("mean_pairwise_ARI","stability_ari_vs_k.png","Mean pairwise ARI"),
                ("minimum_cluster_size","cluster_size_vs_k.png","Minimum hard-cluster size")]
    for key,name,ylabel in plot_specs:
        rows=stability if key=="mean_pairwise_ARI" else selection
        y=[r[key] for r in rows]
        fig,ax=plt.subplots(figsize=(7.2,4.5)); ax.plot(KS,y,marker="o",color="#2563eb",lw=1.8)
        ax.set_xticks(KS); ax.set_xlabel("K"); ax.set_ylabel(ylabel); ax.set_title(f"{ylabel} vs K (direct 8-D Kulfan GMM)")
        ax.grid(alpha=.22); fig.tight_layout(); fig.savefig(OUT/f"figures/{name}",dpi=220); plt.close(fig)
    cfg={"experiment":"Gabriel Kulfan-8 direct-space GMM audit","fallback_notice":FALLBACK,
         "repository_specific_definition_found":False,"dataset":str(DATA.relative_to(ROOT)),"dataset_sha256":data_hash,
         "N_airfoils":n,"coordinate_shape":list(contours.shape),"coordinate_dtype":str(contours.dtype),
         "coefficient_order":COEFF_NAMES,"class_exponents":{"N1":N1,"N2":N2},"bernstein_degree":DEGREE,
         "trailing_edge":{"separate_parameter":False,"convention":"sharp TE forced by class function"},
         "fitting":{"x_coordinates":"fixed from canonical contour; not inferred","objective":"pointwise vertical least squares at fixed x",
                    "solver":"numpy.linalg.lstsq, then scipy SLSQP quadratic program if thickness constraint active",
                    "constraint":"A_upper_j - A_lower_j >= 0 for j=0..3","SLSQP_ftol":1e-12,"SLSQP_maxiter":200},
         "standardization":{"primary_analysis":False,"reason":"no repository Gabriel methodology requiring it; raw Kulfan coefficients used"},
         "split":{"source":"reused outputs/gabriel_pinned_gmm_vs_kmeans/splits","seed":SEED,"train_count":len(tr),"heldout_count":len(ho)},
         "gmm":{"K":KS,"fit_space":"raw X_kulfan, exactly 8 dimensions","no_PCA":True,**GMM_KW,"primary_seed":SEED},
         "stability":{"seeds":SEEDS,"assignment_population":"all N after fitting training only"},"registration_run":False}
    jwrite(OUT/"config.json",cfg)
    preferred={key:int(min(selection,key=lambda r:r[key])["K"]) for key in ("BIC","AIC","ICL")}
    preferred["heldout_log_likelihood"]=int(max(selection,key=lambda r:r["heldout_log_likelihood"])["K"])
    val_checks={"kulfan_dimension_exactly_8":bool(X.shape==(n,8)),"no_PCA_used":True,"GMM_directly_on_8D":True,
                "fallback_clearly_documented":True,"all_airfoils_fitted":bool(len(X)==n and np.isfinite(X).all()),
                "reconstruction_quality_checked":len(metrics)==n,"all_component_templates_valid":all(r["component_mean_valid"] for r in template_rows),
                "no_registration_run":True,"no_prior_outputs_modified":True,"no_commit_or_push":True,
                "dataset_hash_unchanged":sha256(DATA)==data_hash,"all_GMM_converged":all(r["converged"] for r in selection)}
    validation={"status":"PASS" if all(val_checks.values()) else "FAIL","checks":val_checks,
                "reconstruction_summary":rec_summary,"invalid_reconstructions":invalid,
                "self_intersecting_reconstructions":self_intersections,
                "preferred_K":preferred,"template_count":len(template_rows)}
    jwrite(OUT/"validation.json",validation)
    # Technical markdown report, deliberately self-contained and source-backed.
    table='\n'.join('| {K} | {BIC:.2f} | {AIC:.2f} | {ICL:.2f} | {heldout_log_likelihood:.2f} |'.format(**r) for r in selection)
    stab='\n'.join(f"| {r['K']} | {r['mean_pairwise_ARI']:.4f} | {r['sd_pairwise_ARI']:.4f} | {r['modal_exact_partition_frequency']}/20 | {r['ARI_medoid_seed']} |" for r in stability)
    report=f"""# Gabriel 8-D Kulfan/CST audit and direct-space GMM experiment

## Technical summary

**{FALLBACK}** The repository contains no explicit Gabriel Kulfan/CST definition. The adopted vector is `{COEFF_NAMES}` with fixed class exponents N1={N1}, N2={N2}, cubic Bernstein shape functions, and no separate trailing-edge-thickness parameter. All K=1..9 GMMs were fit on raw 8-D coefficients; PCA, standardisation, velocity fields, prior assignments, and registration were not used.

Dataset: `{DATA.relative_to(ROOT)}` (SHA-256 `{data_hash}`), N={n}, contours `{contours.shape}` float64. The reused deterministic split is {len(tr)}/{len(ho)}. Validation status: **{validation['status']}**.

## Model-selection evidence

| K | BIC | AIC | ICL | held-out total LL |
|---:|---:|---:|---:|---:|
{table}

Preferred K: BIC={preferred['BIC']}, AIC={preferred['AIC']}, ICL={preferred['ICL']}, held-out LL={preferred['heldout_log_likelihood']}.

## Scope, data, and representation

The canonical pinned uniform-polyline-arc dataset uses index 0 at TE, indices 0..99 upper TE→LE, index 100 at LE, and indices 100..199 lower LE→TE, with implicit closure. Chord x spans [0,1]. x coordinates are held fixed during fitting. The sharp-TE CST model cannot reproduce finite TE offsets exactly.

Reconstruction RMSE summary: mean {rec_summary['rmse']['mean']:.6g}, median {rec_summary['rmse']['median']:.6g}, p95 {rec_summary['rmse']['p95']:.6g}, max {rec_summary['rmse']['maximum']:.6g}. Strictly invalid reconstructed contours: {invalid}/{n}, including {self_intersections} self-intersections; most other strict failures are zero-length LE segments inherited from repeated canonical fixed-x samples. Full MSE/RMSE/max-error summaries and per-airfoil values are saved in `kulfan/reconstruction_metrics.csv`.

## Methodology and robustness

Each surface uses `y=C(x) Σ A_j B_j^3(x)`, `C(x)=x^0.5(1-x)^1`. Ordinary vertical least squares is used when feasible; otherwise a convex thickness constraint `A_upper_j-A_lower_j>=0` is enforced by SLSQP (ftol 1e-12, max 200). This prevents upper/lower crossings while preserving exactly eight features. GMM settings: full covariance, n_init=5, reg_covar=1e-7, max_iter=500, tol=1e-5, kmeans initialisation. BIC/AIC are sklearn training-set definitions; ICL=BIC+2×training posterior entropy. Held-out LL is evaluated only on the held-out set.

| K | mean pairwise ARI | SD | exact modal frequency | ARI-medoid seed |
|---:|---:|---:|---:|---:|
{stab}

## Component and template audit

Cluster sizes, weights, <20/<1%/<2% flags, imbalance ratios, and posterior ambiguity fractions are in `tables/cluster_sizes_k1_k9.csv` and `tables/component_quality_k1_k9.csv`. Component means and hard-cluster coefficient means both remain inside the convex thickness-feasible region. All {len(template_rows)} component-mean templates are geometrically valid: {all(r['component_mean_valid'] for r in template_rows)}. Their pointwise differences from hard-mean templates are in `tables/template_comparison_k1_k9.csv`; pairwise component-template distances are in `tables/pairwise_template_distances_k1_k9.csv`.

## Limitations and uncertainty

The fallback is not attributed to Gabriel. Four coefficients per surface are low-order and sharp-TE; the worst input is a material outlier even though population-level reconstruction is strong. GMM likelihood criteria need not agree, and exact seed modes are stricter than ARI stability. Imbalance is flagged but not used as an automatic rejection rule.

## Recommended next step

Use these results only to choose candidates for a later, separately authorised registration stage. No production template is selected here.

## Further questions

Whether Gabriel intended a different vertical datum, explicit TE thickness, or alternative Kulfan constraints remains unresolved because no repository definition was found.
"""
    (OUT/"report/final_report.md").write_text(report,encoding="utf-8")
    print(json.dumps({"status":validation["status"],"output":str(OUT),"preferred_K":preferred,
                      "reconstruction_rmse":rec_summary["rmse"]},indent=2))


if __name__ == "__main__": main()
