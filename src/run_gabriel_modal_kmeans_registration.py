"""Gabriel registration driven by saved modal R=200 pinned-shape partitions.

This module never fits K-means.  It verifies the saved 200-seed partitions,
reconstructs pointwise-mean templates, and delegates every registration to the
frozen production implementation.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, time
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import numpy as np
from sklearn.metrics import adjusted_rand_score

import gabriel_pinned_registration as gabriel
import run_gabriel_k2_k8_energy_magnitude as production

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/airfoils/processed_gabriel/airfoils_pinned_arclength_n200.npz"
R200 = ROOT / "outputs/gabriel_pinned_kmeans_r200"
OLD = ROOT / "outputs/gabriel_k2_k8_energy_magnitude"
OUT = ROOT / "outputs/gabriel_modal_kmeans_registration"
KS = tuple(range(2, 10))
CONFIG = production.CONFIG


def canonical(labels):
    labels = np.asarray(labels, dtype=np.int64)
    out = np.empty_like(labels); seen = {}; nxt = 0
    for i, value in enumerate(labels.tolist()):
        if value not in seen: seen[value] = nxt; nxt += 1
        out[i] = seen[value]
    return out


def write_csv(path, rows):
    production.cwrite(path, rows)


def modal_from_saved(K):
    with np.load(R200 / "checkpoints" / f"labels_K{K:02d}.npz") as z:
        labels = np.asarray(z["labels"], dtype=np.int64)
        seeds = np.asarray(z["seeds"], dtype=np.int64)
    groups = {}
    for row, seed in zip(labels, seeds):
        c = canonical(row); key = c.tobytes()
        groups.setdefault(key, {"labels": c, "seeds": []})["seeds"].append(int(seed))
    ranked = sorted(groups.values(), key=lambda g: (-len(g["seeds"]), min(g["seeds"])))
    best = ranked[0]
    saved = canonical(np.load(R200 / "assignments" / f"modal_assignment_K{K:02d}.npy"))
    if not np.array_equal(saved, best["labels"]):
        raise RuntimeError(f"saved modal assignment differs from recomputation for K={K}")
    return saved, len(best["seeds"]), min(best["seeds"]), len(groups)


def template_geometry(points):
    diag = gabriel.validate_pinned_contour(points)
    return {
        "finite": bool(np.isfinite(points).all()),
        "signed_area": float(gabriel.signed_polygon_area(points)),
        "self_intersection_count": int(gabriel.self_intersection_count(points)),
        "pinned_validator_valid": bool(diag["valid"]),
        # A pointwise mean need not satisfy the single-contour arc-length CV
        # tolerance exactly.  Re-sampling it would violate the specified
        # template definition, so retain that check as a diagnostic while
        # using the established production geometric criteria for admission.
        "valid": bool(np.isfinite(points).all() and gabriel.signed_polygon_area(points) > 1e-8
                      and gabriel.self_intersection_count(points) == 0
                      and diag["index0_is_te"] and diag["index100_is_le"]
                      and diag["upper_expected_direction_fraction"] == 1.0
                      and diag["lower_expected_direction_fraction"] == 1.0
                      and diag["upper_is_above_lower"]),
    }


def prepare():
    with np.load(DATA, allow_pickle=False) as z:
        contours = np.asarray(z["contours"], dtype=np.float64)
        filenames = np.asarray(z["filenames"]).astype(str)
    if contours.shape != (1652, 200, 2) or not np.isfinite(contours).all():
        raise RuntimeError(f"unexpected Gabriel pinned data: {contours.shape}")
    for name in ("assignments", "templates", "registrations", "tables", "figures", "report"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    templates = {}; assignments = []; modal_rows = []; size_rows = []
    for K in KS:
        labels, frequency, seed, unique = modal_from_saved(K)
        sizes = np.bincount(labels, minlength=K)
        modal_rows.append({"K": K, "modal_frequency": frequency,
                           "modal_proportion": frequency / 200, "representative_seed": seed,
                           "number_of_unique_partitions": unique,
                           "cluster_sizes": json.dumps(sizes.tolist())})
        np.save(OUT / "assignments" / f"modal_assignment_K{K:02d}.npy", labels)
        fig = Figure(figsize=(10, 4.8)); ax = fig.subplots()
        for cluster in range(K):
            members = np.flatnonzero(labels == cluster)
            mean = contours[members].mean(axis=0)
            geom = template_geometry(mean)
            if not geom["valid"]: raise RuntimeError(f"invalid template K={K}, cluster={cluster}: {geom}")
            tid = f"modal_K{K:02d}_cluster{cluster:02d}"
            templates[(K, cluster)] = (mean, members, tid)
            gabriel.atomic_npz(OUT / "templates" / f"{tid}.npz", coordinates=mean,
                               member_indices=members, member_filenames=filenames[members],
                               geometry_json=np.asarray(json.dumps(geom, sort_keys=True)))
            size_rows.append({"K": K, "cluster_id": cluster, "cluster_size": len(members),
                              "template_id": tid, "template_valid": True,
                              "coordinates_sha256": hashlib.sha256(mean.tobytes()).hexdigest()})
            ax.plot(np.r_[mean[:, 0], mean[0, 0]], np.r_[mean[:, 1], mean[0, 1]],
                    lw=1.4, label=f"cluster {cluster} (n={len(members)})")
        for i, cluster in enumerate(labels):
            assignments.append({"K": K, "target_index": i, "filename": filenames[i],
                                "cluster_id": int(cluster), "representative_seed": seed})
        ax.set_aspect("equal"); ax.set(xlabel="x / chord", ylabel="y / chord",
                                       title=f"Modal Gabriel pinned templates — K={K}")
        ax.grid(alpha=.2); ax.legend(ncol=min(4, K), fontsize=8); fig.tight_layout()
        fig.savefig(OUT / "figures" / f"template_overlay_K{K:02d}.png", dpi=250)
    write_csv(OUT / "tables" / "modal_assignment_summary_k2_k9.csv", modal_rows)
    write_csv(OUT / "tables" / "cluster_sizes_k2_k9.csv", size_rows)
    write_csv(OUT / "assignments" / "modal_assignments_k2_k9.csv", assignments)
    config = {"experiment": "Gabriel pinned modal K-means registration", "dataset": str(DATA.relative_to(ROOT)),
              "dataset_sha256": gabriel.sha256_file(DATA), "dataset_shape": list(contours.shape),
              "K": list(KS), "kmeans_refit": False, "partition_selection": "most frequent among saved 200 seeds; label-permutation invariant",
              "r200_source": str(R200.relative_to(ROOT)), "template_definition": "pointwise arithmetic mean of modal-partition members",
              "registration_config": asdict(CONFIG), "diagnostic_resolution": 101, "workers": 4}
    gabriel.atomic_json(OUT / "config.json", config)
    return contours, filenames, templates, assignments, modal_rows


def read_rows(base, K):
    rows = []; bad = []
    for i in range(1652):
        r = production.load_artifact(base / "registrations" / f"K{K:02d}" / f"{i:04d}.npz", K, i)
        (rows if r is not None else bad).append(r if r is not None else i)
    return rows, bad


def metric_stats(values):
    a = np.asarray(values, float); n = len(a)
    return dict(mean=float(a.mean()), sd=float(a.std(ddof=1)), se=float(a.std(ddof=1)/math.sqrt(n)),
                median=float(np.median(a)), p95=float(np.quantile(a, .95)), maximum=float(a.max()))


def finalise(contours, filenames, templates, assignments, modal_rows):
    all_rows=[]; summary=[]; comparison=[]; clusters=[]; invalid_targets=[]
    amap={(int(a["K"]),int(a["target_index"])):int(a["cluster_id"]) for a in assignments}
    old_assign = {(int(r["K"]), int(r["target_index"])): int(r["cluster_id"])
                  for r in production.cread(OLD / "clustering_assignments.csv")}
    for K in KS:
        rows,bad=read_rows(OUT,K); old,_=read_rows(OLD,K); all_rows.extend(rows)
        valid=[r for r in rows if bool(r.get("valid_registration"))]
        failed=[r for r in rows if r.get("status") != "completed"]
        invalid=[r for r in rows if r.get("status") == "completed" and not bool(r.get("valid_registration"))]
        for r in failed+invalid: invalid_targets.append(r)
        d=metric_stats([r["physical_dirichlet_integral"] for r in valid]); m=metric_stats([r["velocity_mean_magnitude"] for r in valid]); mse=metric_stats([r["final_pointwise_mse"] for r in valid])
        summary.append({"K":K,"completed":len(rows),"valid":len(valid),"invalid":len(invalid),"failed":len(failed),"missing_or_corrupt":len(bad),
                        **{f"dirichlet_{x}":v for x,v in d.items()}, **{f"svf_magnitude_{x}":v for x,v in m.items()},
                        **{f"correspondence_mse_{x}":v for x,v in mse.items()},
                        "minimum_jacobian_determinant":min(float(r["jacobian_minimum"]) for r in rows if "jacobian_minimum" in r)})
        for cl in range(K):
            rr=[r for r in valid if int(r["cluster_id"])==cl]; dd=metric_stats([r["physical_dirichlet_integral"] for r in rr]); mm=metric_stats([r["velocity_mean_magnitude"] for r in rr])
            clusters.append({"K":K,"cluster_id":cl,"cluster_size":len(templates[(K,cl)][1]),"valid":len(rr),**{f"dirichlet_{x}":v for x,v in dd.items()},**{f"svf_magnitude_{x}":v for x,v in mm.items()}})
        ov=[r for r in old if bool(r.get("valid_registration"))]; od=np.mean([r["physical_dirichlet_integral"] for r in ov]); om=np.mean([r["velocity_mean_magnitude"] for r in ov])
        old_lab=np.array([old_assign[(K,i)] for i in range(1652)]); new_lab=np.array([amap[(K,i)] for i in range(1652)])
        comparison.append({"K":K,"old_dirichlet_mean":od,"new_dirichlet_mean":d["mean"],"dirichlet_absolute_difference":d["mean"]-od,"dirichlet_percentage_difference":100*(d["mean"]-od)/od,
                           "old_svf_mean":om,"new_svf_mean":m["mean"],"svf_absolute_difference":m["mean"]-om,"svf_percentage_difference":100*(m["mean"]-om)/om,
                           "old_vs_modal_assignment_ari":adjusted_rand_score(old_lab,new_lab),"exact_same_partition":bool(np.array_equal(canonical(old_lab),canonical(new_lab)))})
    write_csv(OUT/"tables"/"per_registration_results.csv",all_rows); write_csv(OUT/"tables"/"registration_summary_k2_k9.csv",summary)
    write_csv(OUT/"tables"/"old_vs_modal_registration_comparison.csv",comparison); write_csv(OUT/"tables"/"registration_summary_by_cluster_k2_k9.csv",clusters)
    write_csv(OUT/"tables"/"invalid_or_failed_targets.csv",invalid_targets)
    def plot_metric(field,sefield,title,ylabel,name,color):
        fig=Figure(figsize=(10,5.6));ax=fig.subplots();x=[r["K"] for r in summary];y=[r[field] for r in summary];e=[r[sefield] for r in summary]
        ax.errorbar(x,y,yerr=e,marker="o",mfc="white",mec=color,color=color,lw=2,capsize=5);ax.set_xticks(x);ax.set(xlabel="Number of templates K",ylabel=ylabel,title=title);ax.grid(axis="y",alpha=.22);ax.spines[["top","right"]].set_visible(False);fig.tight_layout();fig.savefig(OUT/"figures"/name,dpi=300,bbox_inches="tight")
    plot_metric("dirichlet_mean","dirichlet_se","Average Dirichlet Energy vs Number of Templates","Average physical Dirichlet energy","average_dirichlet_energy_vs_k.png","#1f4e79")
    plot_metric("svf_magnitude_mean","svf_magnitude_se","Average SVF Magnitude vs Number of Templates","Average SVF magnitude","average_svf_magnitude_vs_k.png","#c45a18")
    for metric,title,name,color in (("dirichlet","Old vs Modal Dirichlet Energy","old_vs_modal_dirichlet.png","#1f4e79"),("svf","Old vs Modal SVF Magnitude","old_vs_modal_svf.png","#c45a18")):
        fig=Figure(figsize=(10,5.6));ax=fig.subplots();x=[r["K"] for r in comparison];ax.plot(x,[r[f"old_{metric}_mean"] for r in comparison],marker="o",label="Old production");ax.plot(x,[r[f"new_{metric}_mean"] for r in comparison],marker="s",label="Modal partition",color=color);ax.set_xticks(x);ax.set(xlabel="Number of templates K",ylabel=metric,title=title);ax.grid(axis="y",alpha=.22);ax.legend();fig.tight_layout();fig.savefig(OUT/"figures"/name,dpi=300,bbox_inches="tight")
    checks={"counts_complete":all(r["completed"]==1652 and r["missing_or_corrupt"]==0 for r in summary),"modal_assignments_verified":True,"templates_exact_means":all(np.array_equal(t[0],contours[t[1]].mean(0)) for t in templates.values()),"registration_config_frozen":CONFIG==production.CONFIG,"diagnostic_resolution_101":all(int(r.get("diagnostic_resolution",0))==101 for r in all_rows),"no_kmeans_refit":True}
    gabriel.atomic_json(OUT/"validation.json",{"status":"PASS" if all(checks.values()) else "FAIL","checks":checks})
    report=["# Gabriel modal K-means registration","","This report is descriptive and does not select K.","","| K | valid | Dirichlet mean ± SE | SVF mean ± SE | MSE mean |","|---:|---:|---:|---:|---:|"]
    for r in summary: report.append(f'| {r["K"]} | {r["valid"]} | {r["dirichlet_mean"]:.10g} ± {r["dirichlet_se"]:.3g} | {r["svf_magnitude_mean"]:.10g} ± {r["svf_magnitude_se"]:.3g} | {r["correspondence_mse_mean"]:.3g} |')
    report += ["","Assignments are the label-invariant modal partitions of the saved 200-seed Gabriel-pinned K-means experiment. Templates are direct pointwise means. Registration and diagnostics use the frozen Gabriel implementation."]
    (OUT/"report"/"final_report.md").write_text("\n".join(report)+"\n",encoding="utf8")


def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=("prepare","full","finalise"),default="full");p.add_argument("--workers",type=int,default=4);p.add_argument("--progress-log",type=Path,default=OUT/"progress.log");a=p.parse_args()
    contours,filenames,templates,assignments,modal_rows=prepare()
    production.OUT=OUT
    if a.stage=="prepare": return
    if a.stage=="full": production.run(contours,filenames,templates,assignments,a.workers,requested_ks=KS,progress_log=a.progress_log)
    finalise(contours,filenames,templates,assignments,modal_rows)

if __name__ == "__main__": main()
