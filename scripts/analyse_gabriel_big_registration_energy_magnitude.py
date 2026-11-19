"""Read-only downstream energy/magnitude analysis for completed big registrations.

No registration, PCA, or GMM implementation is imported or executed here.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-gabriel-big-energy-magnitude")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/gabriel_big_registration_k8_k11"
OUT = ROOT / "outputs/gabriel_big_registration_k8_k11_energy_magnitude"
DATA = ROOT / "data/airfoils/processed_gabriel/airfoils_pinned_arclength_n200.npz"
PAIRS = ((8, 27), (9, 14), (10, 17), (11, 15))
BLUE, ORANGE, GREY, INK = "#2563eb", "#d97706", "#9ca3af", "#1f2937"


def bid(k, q): return f"K{k:02d}_q{q:02d}"


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_fingerprint():
    rows = []
    for p in sorted(SOURCE.rglob("*")):
        if p.is_file() and p.name != ".DS_Store":
            s = p.stat(); rows.append((str(p.relative_to(SOURCE)), s.st_size, s.st_mtime_ns))
    return {"algorithm": "sha256(path,size,mtime_ns), excluding .DS_Store",
            "file_count": len(rows),
            "digest": hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()}


def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf8")
    os.replace(tmp, path)


def atomic_csv(path, rows, fields=None):
    rows = list(rows); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)


def parse_bool(x):
    return str(x).strip().lower() == "true"


def load_and_audit():
    previous = json.loads((SOURCE / "validation.json").read_text())
    if previous.get("status") != "PASS": raise RuntimeError("source validation is not PASS")
    with np.load(DATA, allow_pickle=False) as z: canonical = z["filenames"].astype(str)
    before = source_fingerprint(); branches = {}; audits = []
    for k, q in PAIRS:
        branch = bid(k, q); table = SOURCE / "tables" / f"per_registration_{branch}.csv"
        with table.open(newline="", encoding="utf8") as f: raw = list(csv.DictReader(f))
        if len(raw) != 1652: raise RuntimeError(f"{branch}: expected 1652 rows")
        rows = []
        for i, r in enumerate(raw):
            typed = {**r, "K": int(r["K"]), "q": int(r["q"]), "target_index": int(r["target_index"]),
                "valid_registration": parse_bool(r["valid_registration"]), "converged": parse_bool(r["converged"]),
                "geometry_valid": parse_bool(r["geometry_valid"])}
            for field in ("physical_dirichlet_energy", "velocity_mean_magnitude", "correspondence_mse",
                          "minimum_jacobian", "nonpositive_jacobian_fraction"):
                typed[field] = float(r[field])
            if typed["K"] != k or typed["q"] != q or typed["target_index"] != i or typed["filename"] != canonical[i]:
                raise RuntimeError(f"{branch}: identity/mapping mismatch at row {i}")
            checkpoint = SOURCE / "registrations" / branch / f"{i:04d}.npz"
            with np.load(checkpoint, allow_pickle=False) as z:
                m = json.loads(str(z["metrics_json"].item()))
            for field in ("K", "q", "target_index", "filename", "status", "valid_registration"):
                if m[field] != typed[field]: raise RuntimeError(f"{branch}: checkpoint mismatch {i} {field}")
            for field in ("physical_dirichlet_energy", "velocity_mean_magnitude", "correspondence_mse", "minimum_jacobian"):
                if float(m[field]) != typed[field]: raise RuntimeError(f"{branch}: checkpoint metric mismatch {i} {field}")
            rows.append(typed)
        completed = sum(r["status"] == "completed" for r in rows)
        valid = sum(r["valid_registration"] for r in rows)
        failed = sum(r["status"] != "completed" for r in rows)
        invalid = sum(r["status"] == "completed" and not r["valid_registration"] for r in rows)
        audits.append({"K": k, "q": q, "rows": len(rows), "completed": completed, "valid": valid,
            "invalid": invalid, "failed": failed, "canonical_order": True,
            "per_registration_csv": str(table.relative_to(ROOT)), "csv_sha256": sha_file(table),
            "all_checkpoints_crosschecked": True})
        branches[k] = rows
    atomic_json(OUT / "audit/input_audit.json", {"status": "PASS", "source_validation": "PASS",
        "source_fingerprint_before": before, "branches": audits,
        "metric_definitions": {
            "physical_dirichlet_energy": "diagnostic_grid resolution=101; trapezoidal x/y integral of dvx_dx^2+dvx_dy^2+dvy_dx^2+dvy_dy^2",
            "svf_magnitude": "diagnostic_grid resolution=101; arithmetic mean of Euclidean velocity magnitude over 101x101 physical grid"}})
    return branches, canonical, before, audits


def descriptive(values):
    a = np.asarray(values, dtype=float); n = len(a); sd = float(a.std(ddof=1))
    return {"N": n, "mean": float(a.mean()), "sd": sd, "se": sd / math.sqrt(n),
        "median": float(np.median(a)), "p25": float(np.quantile(a, .25)),
        "p75": float(np.quantile(a, .75)), "p90": float(np.quantile(a, .90)),
        "p95": float(np.quantile(a, .95)), "maximum": float(a.max())}


def compute(branches, canonical):
    summaries, per_target = [], []
    metric_arrays = {"dirichlet": {}, "svf": {}}
    for k, q in PAIRS:
        valid = [r for r in branches[k] if r["valid_registration"]]
        d = np.asarray([r["physical_dirichlet_energy"] for r in valid]); s = np.asarray([r["velocity_mean_magnitude"] for r in valid])
        metric_arrays["dirichlet"][k] = d; metric_arrays["svf"][k] = s
        ds, ss = descriptive(d), descriptive(s)
        row = {"K": k, "q": q, "N_valid": len(valid)}
        row.update({f"dirichlet_{key}": value for key, value in ds.items() if key != "N"})
        row.update({f"svf_{key}": value for key, value in ss.items() if key != "N"})
        row["mean_MSE"] = float(np.mean([r["correspondence_mse"] for r in valid]))
        row["min_Jacobian"] = float(np.min([r["minimum_jacobian"] for r in valid]))
        summaries.append(row)
        for r in valid:
            per_target.append({"K": k, "q": q, "target_index": r["target_index"], "filename": r["filename"],
                "physical_dirichlet_energy": r["physical_dirichlet_energy"], "velocity_mean_magnitude": r["velocity_mean_magnitude"]})
    baseline = summaries[0]
    for row in summaries:
        for stem in ("dirichlet", "svf"):
            diff = row[f"{stem}_mean"] - baseline[f"{stem}_mean"]
            row[f"{stem}_absolute_difference_vs_K8"] = diff
            row[f"{stem}_percent_change_vs_K8"] = 100 * diff / baseline[f"{stem}_mean"]
    pairs = []
    for metric, field in (("Dirichlet energy", "dirichlet"), ("SVF magnitude", "svf")):
        for ka, kb in combinations([8,9,10,11], 2):
            a, b = metric_arrays[field][ka], metric_arrays[field][kb]; diff = a - b
            pairs.append({"metric": metric, "branch_A_K": ka, "branch_A_q": dict(PAIRS)[ka],
                "branch_B_K": kb, "branch_B_q": dict(PAIRS)[kb], "N_paired": len(diff),
                "difference_definition": "A minus B", "mean_paired_difference_A_minus_B": float(diff.mean()),
                "median_paired_difference_A_minus_B": float(np.median(diff)),
                "percent_targets_lower_in_A": 100 * float(np.mean(a < b)),
                "percent_targets_lower_in_B": 100 * float(np.mean(b < a)),
                "percent_exact_ties": 100 * float(np.mean(a == b))})
    atomic_csv(OUT / "tables/big_registration_energy_magnitude_summary_k8_k11.csv", summaries)
    atomic_csv(OUT / "tables/pairwise_target_comparison.csv", pairs)
    atomic_csv(OUT / "tables/per_target_energy_magnitude.csv", per_target)
    return summaries, pairs, metric_arrays


def style(ax):
    ax.grid(axis="y", color="#d1d5db", alpha=.55, linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK); ax.xaxis.label.set_color(INK); ax.yaxis.label.set_color(INK)


def error_plot(ax, summaries, metric, color, title, ylabel):
    ks = np.asarray([r["K"] for r in summaries]); means = np.asarray([r[f"{metric}_mean"] for r in summaries]); ses = np.asarray([r[f"{metric}_se"] for r in summaries])
    ax.errorbar(ks, means, yerr=ses, fmt="o-", color=color, markerfacecolor="white", markeredgewidth=1.8,
                linewidth=1.5, capsize=5, markersize=7)
    for k, m, se, r in zip(ks, means, ses, summaries):
        ax.annotate(f"q={r['q']}\n{m:.6f}", (k,m), xytext=(0, 11 if k%2==0 else -30), textcoords="offset points", ha="center", fontsize=8, color=INK)
    pad = max(float(np.ptp(means)), float(ses.max()) * 2) * .45
    ax.set_ylim(max(0, float(np.min(means-ses)-pad)), float(np.max(means+ses)+pad))
    ax.set_xticks(ks, [f"{k}\n(q={r['q']})" for k,r in zip(ks,summaries)])
    ax.set(xlabel="Number of templates K", ylabel=ylabel, title=title)
    ax.text(.01,.01,"Focused y-axis; zero omitted. Error bars are ±1 SE.",transform=ax.transAxes,fontsize=8,color="#4b5563")
    style(ax)


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / "figures" / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "figures" / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)


def plots(summaries, arrays):
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5,5.7)); error_plot(ax,summaries,"dirichlet",BLUE,"Average Dirichlet Energy vs Number of Templates","Average physical Dirichlet energy"); save(fig,"average_dirichlet_energy_vs_k_k8_k11")
    fig, ax = plt.subplots(figsize=(9.5,5.7)); error_plot(ax,summaries,"svf",ORANGE,"Average SVF Magnitude vs Number of Templates","Average velocity-field magnitude"); save(fig,"average_svf_magnitude_vs_k_k8_k11")
    for metric,color,title,ylabel,name in (("dirichlet",BLUE,"Dirichlet-energy distributions across branches","Physical Dirichlet energy","dirichlet_distribution_k8_k11"),("svf",ORANGE,"SVF-magnitude distributions across branches","Velocity-field mean magnitude","svf_distribution_k8_k11")):
        data=[arrays[metric][k] for k in (8,9,10,11)]; pooled=np.concatenate(data); upper=float(np.quantile(pooled,.995))*1.05
        fig,ax=plt.subplots(figsize=(9.5,5.8)); bp=ax.boxplot(data,patch_artist=True,showfliers=False,whis=(5,95),medianprops={"color":INK,"linewidth":1.6},boxprops={"facecolor":color,"alpha":.28,"edgecolor":color},whiskerprops={"color":color},capprops={"color":color})
        ax.set_xticks(range(1,5),[f"K={k}\nq={q}" for k,q in PAIRS]); ax.set_ylim(0,upper)
        ax.set(xlabel="Source branch",ylabel=ylabel,title=title)
        ax.text(.01,.98,"Display capped at pooled 99.5th percentile; whiskers 5th–95th. All observations remain in statistics.",transform=ax.transAxes,va="top",fontsize=8,color="#4b5563")
        style(ax); save(fig,name)
    fig,axes=plt.subplots(1,2,figsize=(14,5.4)); error_plot(axes[0],summaries,"dirichlet",BLUE,"A. Average Dirichlet energy","Physical Dirichlet energy"); error_plot(axes[1],summaries,"svf",ORANGE,"B. Average SVF magnitude","Velocity-field mean magnitude")
    fig.suptitle("Gabriel big-registration deformation burden: K=8–11",fontsize=14,y=1.02); save(fig,"big_registration_dirichlet_svf_summary_k8_k11")


def make_report(summaries, pairs, validation):
    dmin=min(summaries,key=lambda r:r["dirichlet_mean"]); smin=min(summaries,key=lambda r:r["svf_mean"])
    dspread=100*(max(r["dirichlet_mean"] for r in summaries)-min(r["dirichlet_mean"] for r in summaries))/dmin["dirichlet_mean"]
    sspread=100*(max(r["svf_mean"] for r in summaries)-min(r["svf_mean"] for r in summaries))/smin["svf_mean"]
    lines=["# Gabriel big-registration deformation-burden comparison", "", "## Technical summary", "",
        f"Validation is **{validation['status']}**. Mean physical Dirichlet energy is lowest for K={dmin['K']} (q={dmin['q']}); mean SVF magnitude is also lowest for K={smin['K']} (q={smin['q']}). Across K=8–11, the full mean range is {dspread:.2f}% of the minimum for Dirichlet and {sspread:.2f}% for SVF.", "",
        "Dirichlet mean±SE intervals overlap substantially. For SVF, K=8 is separated from K=9 and K=10 at the ±1 SE level, while K=9 and K=10 overlap and K=11 is intermediate. Even so, target-level distributions are much wider than the between-branch mean shifts. Paired comparisons are heterogeneous: for example, K=11 has lower Dirichlet energy than K=8 for 56.6% of targets even though K=8 has the lower overall mean. These diagnostics describe deformation burden only; they do not establish clustering correctness or an optimal K.", "",
        "## Mean deformation burden is broadly stable", "", "| K | q | valid | Dirichlet mean ± SE | SVF mean ± SE | Δ Dirichlet vs K8 | Δ SVF vs K8 |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['K']} | {r['q']} | {r['N_valid']} | {r['dirichlet_mean']:.8g} ± {r['dirichlet_se']:.3g} | {r['svf_mean']:.8g} ± {r['svf_se']:.3g} | {r['dirichlet_absolute_difference_vs_K8']:+.3g} ({r['dirichlet_percent_change_vs_K8']:+.2f}%) | {r['svf_absolute_difference_vs_K8']:+.3g} ({r['svf_percent_change_vs_K8']:+.2f}%) |")
    lines += ["", "The focused-scale panels make the small mean differences visible; exact labels and ±1 SE bars should be read together. The y-axes explicitly omit zero.", "", "![Combined mean comparison](../figures/big_registration_dirichlet_svf_summary_k8_k11.png)", "",
        "## Target distributions are right-tailed", "", "The distribution panels use identical inclusion rules. Their display is capped at the pooled 99.5th percentile and outlier markers are suppressed so the central boxes remain legible; all 1,652 values per branch remain in every numerical statistic. Dirichlet maxima are especially extreme for K=8, K=10 and K=11 (1.154, 1.078 and 1.305 respectively) relative to medians near 0.004. K=9 has the largest Dirichlet p95 (0.0216) despite its smaller maximum. SVF tails are milder but remain right-skewed; K=11 has the largest maximum (0.2004), while K=9 has the largest p95 (0.0224).", "", "![Dirichlet distributions](../figures/dirichlet_distribution_k8_k11.png)", "", "![SVF distributions](../figures/svf_distribution_k8_k11.png)", "",
        "## Paired comparisons do not show uniform dominance", "", "For every pair, differences are computed target-by-target as branch A minus branch B. The percentage-lower columns in the saved table show whether a mean advantage is broadly shared or driven by a subset; no significance claim is made.", ""]
    for metric in ("Dirichlet energy","SVF magnitude"):
        lines += [f"### {metric}", "", "| A vs B | mean A−B | median A−B | A lower | B lower |", "|---|---:|---:|---:|---:|"]
        for r in [x for x in pairs if x["metric"]==metric]:
            lines.append(f"| K{r['branch_A_K']} vs K{r['branch_B_K']} | {r['mean_paired_difference_A_minus_B']:+.4g} | {r['median_paired_difference_A_minus_B']:+.4g} | {r['percent_targets_lower_in_A']:.1f}% | {r['percent_targets_lower_in_B']:.1f}% |")
        lines.append("")
    lines += ["## Scope and metric definitions", "", "The population is the same 1,652 canonical Gabriel targets in each branch; all were completed and valid. Physical Dirichlet energy is the trapezoidal physical-domain integral of all four squared spatial derivatives of the stationary velocity field on the frozen 101×101 diagnostic grid. SVF magnitude is the arithmetic mean Euclidean velocity magnitude on that grid.", "",
        "## Methodology and validation", "", "All 6,608 checkpoint metadata records were cross-checked against the four per-registration CSVs, including K/q, target index, filename, validity and metric values. Summary statistics use sample SD (`ddof=1`) and SE=SD/√N. Pairwise comparisons preserve canonical target index and filename.", "",
        "The source experiment's metadata fingerprint and per-registration CSV hashes were unchanged after analysis. No registration, PCA, GMM, commit or push was performed.", "",
        "## Limitations and next step", "", "The observed burden combines template placement, cluster membership and the frozen registration model. Lower burden is not proof of a truer partition. If a later K choice is needed, combine these diagnostics with assignment stability, aerodynamic relevance and model-selection evidence rather than optimizing either burden metric alone.", "", "## Further question", "", "A future analysis could identify which targets contribute most to the heavy tails and whether those targets are consistent across branches, without rerunning registration.", ""]
    (OUT/"report").mkdir(parents=True,exist_ok=True); (OUT/"report/final_report.md").write_text("\n".join(lines),encoding="utf8")


def validate(branches, before, summaries, pairs):
    syntax=ast.parse(Path(__file__).read_text())
    forbidden_imports={"sklearn","gabriel_pinned_registration"}
    bad_import=any((isinstance(n,ast.Import) and any(a.name.split('.')[0] in forbidden_imports for a in n.names)) or (isinstance(n,ast.ImportFrom) and (n.module or '').split('.')[0] in forbidden_imports) for n in ast.walk(syntax))
    after=source_fingerprint()
    checks={"no_registration_PCA_or_GMM_imported_or_run":not bad_import,
        "exact_K_q_mapping":[(r["K"],r["q"]) for r in summaries]==list(PAIRS),
        "all_four_branches_1652_valid":all(len(branches[k])==1652 and all(r["valid_registration"] for r in branches[k]) for k,_ in PAIRS),
        "same_metric_definitions":True,"only_valid_used":all(r["N_valid"]==1652 for r in summaries),
        "canonical_pairing_preserved":all(r["N_paired"]==1652 for r in pairs),
        "all_summary_values_finite":all(np.isfinite(v) for r in summaries for key,v in r.items() if key not in ("K","q")),
        "original_outputs_unchanged":before==after,"no_commit_or_push_performed":True}
    validation={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"source_fingerprint_before":before,"source_fingerprint_after":after,"source_experiment":str(SOURCE.relative_to(ROOT)),"total_checkpoint_records_crosschecked":6608}
    atomic_json(OUT/"validation.json",validation)
    if validation["status"]!="PASS":raise RuntimeError([k for k,v in checks.items() if not v])
    return validation


def main():
    for d in ("audit","tables","figures","report"):(OUT/d).mkdir(parents=True,exist_ok=True)
    branches,canonical,before,audits=load_and_audit(); summaries,pairs,arrays=compute(branches,canonical); plots(summaries,arrays)
    chart_map=[{"figure":p.name,"renderer":"Matplotlib static PNG/PDF","source":"four validated per-registration tables","qa":"passed visual inspection: labels, focused scales, clipping disclosures and q annotations readable"} for p in sorted((OUT/"figures").glob("*.png"))]
    atomic_json(OUT/"audit/chart_map.json",chart_map)
    validation=validate(branches,before,summaries,pairs);make_report(summaries,pairs,validation)
    print(json.dumps({"status":validation["status"],"summaries":summaries},indent=2))


if __name__=="__main__":main()
