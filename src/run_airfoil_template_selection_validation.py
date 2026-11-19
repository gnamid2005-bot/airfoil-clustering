from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from airfoil_data import plot_shape
from airfoil_registration import (
    camber_thickness_from_shapes,
    gmm_posteriors_for_shapes,
    load_airfoil_dataset,
    reconstruct_gmm_templates,
    reconstruct_k1_template,
    register_airfoils,
    registration_bounds,
    save_csv,
    save_loss_curve,
    save_registration_figure,
    validate_airfoil_shape,
    write_json,
)


matplotlib.use("Agg")


DEFAULT_DATA_PATH = Path("data/airfoils/processed/airfoils_n100_seed0.npz")
DEFAULT_GMM_DIR = Path("outputs/airfoil_gmm_selection")
DEFAULT_OUTPUT_DIR = Path("outputs/airfoil_registration_validation")
PCA_Q20 = Path("pca") / "pca_q20.npz"


CANDIDATES = {
    1: {"label": "K=1 global mean", "q": 20, "covariance_type": "global", "seed": 0},
    2: {"label": "K=2 q20 full", "q": 20, "covariance_type": "full", "seed": 0},
    3: {"label": "K=3 q20 full", "q": 20, "covariance_type": "full", "seed": 0},
    6: {"label": "K=6 q20 full", "q": 20, "covariance_type": "full", "seed": 0},
    9: {"label": "K=9 q20 spherical", "q": 20, "covariance_type": "spherical", "seed": 0},
}


def parse_k_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_result_path(gmm_dir: Path, k: int) -> Path:
    spec = CANDIDATES[k]
    return gmm_dir / "runs" / f"q{spec['q']}" / f"k{k}" / spec["covariance_type"] / f"seed{spec['seed']}" / "result.npz"


def load_candidate_templates(dataset: dict, gmm_dir: Path, train_indices: np.ndarray, k_values: list[int]) -> dict[int, dict]:
    candidates = {}
    pca_path = gmm_dir / PCA_Q20
    for k in k_values:
        spec = dict(CANDIDATES[k])
        if k == 1:
            templates = np.expand_dims(reconstruct_k1_template(dataset, train_indices), axis=0)
            weights = np.ones(1)
            gmm_path = None
        else:
            gmm_path = candidate_result_path(gmm_dir, k)
            templates, metadata = reconstruct_gmm_templates(dataset, pca_path, gmm_path)
            weights = metadata["weights"]
        candidates[k] = {
            "spec": spec,
            "templates": templates,
            "weights": weights,
            "gmm_path": str(gmm_path) if gmm_path else "",
            "validity": [validate_airfoil_shape(shape) for shape in templates],
        }
    return candidates


def distance_to_global_mean(dataset: dict, train_indices: np.ndarray) -> np.ndarray:
    values = camber_thickness_from_shapes(dataset["shapes"])
    global_mean = values[train_indices].mean(axis=0, keepdims=True)
    return np.sqrt(np.sum((values - global_mean) ** 2, axis=1))


def select_targets(dataset: dict, gmm_dir: Path, mode: str) -> list[int]:
    registration_subset = np.load(gmm_dir / "registration_subset_indices.npy").astype(int)
    extreme = set(np.load(gmm_dir / "extreme_group_heldout_indices.npy").astype(int).tolist())
    train_indices = np.load(gmm_dir / "train_indices.npy").astype(int)
    distances = distance_to_global_mean(dataset, train_indices)
    if mode == "full":
        missing_extreme = sorted(extreme - set(registration_subset.tolist()))
        if missing_extreme:
            raise ValueError(f"Registration subset is missing extreme held-out indices: {missing_extreme}")
        return registration_subset.tolist()
    ordinary = [idx for idx in registration_subset.tolist() if idx not in extreme]
    ordinary_index = min(ordinary, key=lambda idx: distances[idx])
    extreme_index = min(extreme, key=lambda idx: distances[idx])
    return [int(ordinary_index), int(extreme_index)]


def target_categories(dataset: dict, gmm_dir: Path, target_indices: list[int]) -> dict[int, str]:
    extreme = set(np.load(gmm_dir / "extreme_group_heldout_indices.npy").astype(int).tolist())
    train_indices = np.load(gmm_dir / "train_indices.npy").astype(int)
    distances = distance_to_global_mean(dataset, train_indices)
    non_extreme = [idx for idx in target_indices if idx not in extreme]
    if not non_extreme:
        return {idx: "extreme" for idx in target_indices}
    q1, q2 = np.quantile(distances[non_extreme], [1.0 / 3.0, 2.0 / 3.0])
    categories = {}
    for idx in target_indices:
        if idx in extreme:
            categories[idx] = "extreme"
        elif distances[idx] <= q1:
            categories[idx] = "close"
        elif distances[idx] <= q2:
            categories[idx] = "medium"
        else:
            categories[idx] = "far"
    return categories


def load_or_run_registration(
    source: np.ndarray,
    target: np.ndarray,
    config: dict,
    seed: int,
    run_dir: Path,
    figure_prefix: Path,
    title: str,
    force: bool,
) -> dict:
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and not force:
        metrics = read_json(summary_path)
        return {"metrics": metrics, "skipped": True}
    run_dir.mkdir(parents=True, exist_ok=True)
    registered = register_airfoils(source, target, config, seed)
    np.save(run_dir / "deformed.npy", registered["deformed"])
    np.save(run_dir / "coeffs.npy", registered["coeffs"])
    save_csv(run_dir / "history.csv", registered["history"])
    write_json(summary_path, registered["metrics"])
    save_registration_figure(source, target, registered["deformed"], figure_prefix.with_suffix(".png"), title)
    save_loss_curve(registered["history"], figure_prefix.with_name(figure_prefix.name + "_loss").with_suffix(".png"), title + " loss")
    return {"metrics": registered["metrics"], "skipped": False}


def finite_valid(row: dict) -> bool:
    return row.get("failure_reason", "") == "" and bool(row.get("final_airfoil_valid", False))


def standard_error(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def summarise_selected(rows: list[dict], target_rows: list[dict], k_values: list[int]) -> list[dict]:
    target_group = {row["target_index"]: row["target_category"] for row in target_rows}
    summaries = []
    for k in k_values:
        selected = [row for row in rows if row["k"] == k and row["selected_template"]]
        attempted = [row for row in rows if row["k"] == k]
        group_pairs = [("all", selected)]
        for group in ["close", "medium", "far", "extreme"]:
            group_pairs.append((group, [row for row in selected if target_group[row["target_index"]] == group]))
        for label, subset in group_pairs:
            subset_targets = {row["target_index"] for row in subset}
            attempted_subset = [row for row in attempted if row["target_index"] in subset_targets]
            losses = [float(row["total_loss"]) for row in subset if finite_valid(row)]
            failures = [row for row in subset if not finite_valid(row)]
            attachments = [float(row["final_data_attachment"]) for row in subset if finite_valid(row)]
            raw_regs = [float(row["raw_regulariser"]) for row in subset if finite_valid(row)]
            regs = [float(row["weighted_regulariser"]) for row in subset if finite_valid(row)]
            disps = [float(row["mean_displacement_norm"]) for row in subset if finite_valid(row)]
            max_disps = [float(row["maximum_displacement_norm"]) for row in subset if finite_valid(row)]
            runtimes = [float(row["runtime_seconds"]) for row in attempted_subset]
            jmins = [float(row["minimum_jacobian"]) for row in subset if finite_valid(row)]
            unsafe = [
                row
                for row in subset
                if (not finite_valid(row))
                or float(row["nonpositive_jacobian_proportion"]) > 0.0
                or float(row["minimum_jacobian"]) <= 0.0
            ]
            agreements = [
                row
                for row in subset
                if row["k"] > 1 and row.get("minimum_energy_template_agrees_with_posterior") in {True, False}
            ]
            if losses:
                summary = {
                    "subset": label,
                    "k": k,
                    "n_targets": len(subset),
                    "attempted_registrations": len(attempted_subset),
                    "n_valid": len(losses),
                    "selected_failure_count": len(failures),
                    "failure_rate": len(failures) / max(1, len(subset)),
                    "mean_total_loss": float(np.mean(losses)),
                    "standard_error": standard_error(losses),
                    "median_total_loss": float(np.median(losses)),
                    "p90_total_loss": float(np.quantile(losses, 0.90)),
                    "p95_total_loss": float(np.quantile(losses, 0.95)),
                    "mean_attachment": float(np.mean(attachments)),
                    "mean_raw_regularisation": float(np.mean(raw_regs)),
                    "mean_weighted_regularisation": float(np.mean(regs)),
                    "mean_displacement": float(np.mean(disps)),
                    "mean_maximum_displacement": float(np.mean(max_disps)),
                    "p95_maximum_displacement": float(np.quantile(max_disps, 0.95)),
                    "mean_runtime_seconds": float(np.sum(runtimes) / max(1, len(subset))) if runtimes else 0.0,
                    "total_runtime_seconds": float(np.sum(runtimes)),
                    "minimum_jacobian": float(np.min(jmins)),
                    "median_minimum_jacobian": float(np.median(jmins)),
                    "p05_minimum_jacobian": float(np.quantile(jmins, 0.05)),
                    "unsafe_selected_mapping_count": len(unsafe),
                    "unsafe_selected_mapping_proportion": len(unsafe) / max(1, len(subset)),
                    "posterior_energy_agreement_rate": (
                        float(np.mean([bool(row["minimum_energy_template_agrees_with_posterior"]) for row in agreements]))
                        if agreements
                        else ""
                    ),
                }
            else:
                summary = {
                    "subset": label,
                    "k": k,
                    "n_targets": len(subset),
                    "attempted_registrations": len(attempted_subset),
                    "n_valid": 0,
                    "selected_failure_count": len(failures),
                    "failure_rate": 1.0,
                }
            summaries.append(summary)
    return summaries


def paired_comparisons(rows: list[dict], target_rows: list[dict], k_values: list[int]) -> list[dict]:
    target_extreme = {row["target_index"]: bool(row["previous_extreme_heldout"]) for row in target_rows}
    selected = [row for row in rows if row["selected_template"]]
    by_target_k = {(row["target_index"], row["k"]): row for row in selected}
    runtime_by_target_k = {}
    for row in rows:
        key = (row["target_index"], row["k"])
        runtime_by_target_k[key] = runtime_by_target_k.get(key, 0.0) + float(row["runtime_seconds"])
    comparisons = []
    for target in sorted({row["target_index"] for row in selected}):
        baseline = by_target_k.get((target, 1))
        if baseline is None:
            continue
        for k in k_values:
            if k == 1:
                continue
            row = by_target_k.get((target, k))
            if row is None:
                continue
            comparisons.append(
                {
                    "target_index": target,
                    "filename": row["filename"],
                    "previous_extreme_heldout": target_extreme[target],
                    "target_category": row["target_category"],
                    "k": k,
                    "baseline_valid": finite_valid(baseline),
                    "candidate_valid": finite_valid(row),
                    "delta_attachment": float(row["final_data_attachment"]) - float(baseline["final_data_attachment"]),
                    "delta_regularisation": float(row["weighted_regulariser"]) - float(baseline["weighted_regulariser"]),
                    "delta_weighted_regularisation": float(row["weighted_regulariser"]) - float(baseline["weighted_regulariser"]),
                    "delta_total_loss": float(row["total_loss"]) - float(baseline["total_loss"]),
                    "delta_displacement": float(row["mean_displacement_norm"]) - float(baseline["mean_displacement_norm"]),
                    "delta_runtime": runtime_by_target_k[(target, k)] - runtime_by_target_k[(target, 1)],
                    "improves_total_loss": float(row["total_loss"]) < float(baseline["total_loss"]),
                }
            )
    return comparisons


def bootstrap_deltas(comparisons: list[dict], k_values: list[int], seed: int, n_bootstrap: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    output = []
    for k in k_values:
        if k == 1:
            continue
        subset = [row for row in comparisons if row["k"] == k and row["baseline_valid"] and row["candidate_valid"]]
        if not subset:
            continue
        n = len(subset)
        for key in ["delta_total_loss", "delta_attachment", "delta_weighted_regularisation", "delta_displacement"]:
            observed = float(np.mean([row[key] for row in subset]))
            samples = []
            for _ in range(n_bootstrap):
                draw = rng.integers(0, n, size=n)
                samples.append(float(np.mean([subset[i][key] for i in draw])))
            output.append(
                {
                    "k": k,
                    "metric": key,
                    "n_targets": n,
                    "observed_mean": observed,
                    "ci95_low": float(np.quantile(samples, 0.025)),
                    "ci95_high": float(np.quantile(samples, 0.975)),
                }
            )
    return output


def paired_delta_summary(comparisons: list[dict], k_values: list[int]) -> list[dict]:
    output = []
    for k in k_values:
        if k == 1:
            continue
        for subset_name, subset in [
            ("all", [row for row in comparisons if row["k"] == k and row["baseline_valid"] and row["candidate_valid"]]),
            ("extreme", [row for row in comparisons if row["k"] == k and row["baseline_valid"] and row["candidate_valid"] and row["target_category"] == "extreme"]),
        ]:
            if not subset:
                continue
            deltas = [row["delta_total_loss"] for row in subset]
            unchanged_tol = 1e-8
            output.append(
                {
                    "k": k,
                    "subset": subset_name,
                    "n_targets": len(subset),
                    "mean_delta_total_loss": float(np.mean(deltas)),
                    "median_delta_total_loss": float(np.median(deltas)),
                    "mean_delta_attachment": float(np.mean([row["delta_attachment"] for row in subset])),
                    "mean_delta_weighted_regularisation": float(np.mean([row["delta_weighted_regularisation"] for row in subset])),
                    "mean_delta_displacement": float(np.mean([row["delta_displacement"] for row in subset])),
                    "mean_delta_runtime": float(np.mean([row["delta_runtime"] for row in subset])),
                    "proportion_improved": float(np.mean([delta < -unchanged_tol for delta in deltas])),
                    "proportion_worsened": float(np.mean([delta > unchanged_tol for delta in deltas])),
                    "proportion_approximately_unchanged": float(np.mean([abs(delta) <= unchanged_tol for delta in deltas])),
                }
            )
    return output


def select_k_and_pareto(summary_rows: list[dict], template_meta: dict, k_values: list[int]) -> tuple[dict, list[dict]]:
    all_rows = {row["k"]: row for row in summary_rows if row["subset"] == "all" and row.get("n_valid", 0)}
    extreme_rows = {row["k"]: row for row in summary_rows if row["subset"] == "extreme" and row.get("n_valid", 0)}
    k_min = min(all_rows, key=lambda k: all_rows[k]["mean_total_loss"])
    threshold = all_rows[k_min]["mean_total_loss"] + all_rows[k_min]["standard_error"]
    baseline_failure = all_rows[1]["failure_rate"]
    baseline_unsafe = all_rows[1]["unsafe_selected_mapping_proportion"]
    baseline_extreme_p95 = extreme_rows.get(1, {}).get("p95_total_loss", float("inf"))
    eligible = []
    for k in sorted(all_rows):
        templates_valid = all(row.get("valid", False) for row in template_meta[str(k)]["validity"])
        extreme_ok = k not in extreme_rows or extreme_rows[k]["p95_total_loss"] <= baseline_extreme_p95 * 1.25
        if (
            all_rows[k]["mean_total_loss"] <= threshold
            and all_rows[k]["failure_rate"] <= baseline_failure
            and all_rows[k]["unsafe_selected_mapping_proportion"] <= baseline_unsafe
            and extreme_ok
            and templates_valid
        ):
            eligible.append(k)
    k_star = min(eligible) if eligible else k_min
    pareto_rows = []
    for k in sorted(all_rows):
        row = all_rows[k]
        dominated = False
        for other_k, other in all_rows.items():
            if other_k == k:
                continue
            no_more_templates = other_k <= k
            no_more_runtime = other["mean_runtime_seconds"] <= row["mean_runtime_seconds"] * (1.0 + 1e-12)
            at_least_as_good = (
                other["mean_total_loss"] <= row["mean_total_loss"] * (1.0 + 1e-12)
                and other["p95_total_loss"] <= row["p95_total_loss"] * (1.0 + 1e-12)
                and other["failure_rate"] <= row["failure_rate"]
            )
            strictly_better = (
                other_k < k
                or other["mean_runtime_seconds"] < row["mean_runtime_seconds"]
                or other["mean_total_loss"] < row["mean_total_loss"]
                or other["p95_total_loss"] < row["p95_total_loss"]
            )
            dominated = dominated or (no_more_templates and no_more_runtime and at_least_as_good and strictly_better)
        pareto_rows.append(
            {
                "k": k,
                "n_templates": k,
                "mean_total_loss": row["mean_total_loss"],
                "p95_total_loss": row["p95_total_loss"],
                "failure_rate": row["failure_rate"],
                "mean_runtime_seconds": row["mean_runtime_seconds"],
                "extreme_mean_total_loss": extreme_rows.get(k, {}).get("mean_total_loss", ""),
                "extreme_p95_total_loss": extreme_rows.get(k, {}).get("p95_total_loss", ""),
                "dominated": dominated,
                "pareto_frontier": not dominated,
            }
        )
    selection = {
        "k_minimum_mean_cost": k_min,
        "minimum_mean_cost": all_rows[k_min]["mean_total_loss"],
        "se_at_minimum": all_rows[k_min]["standard_error"],
        "one_standard_error_threshold": threshold,
        "eligible_k_values": eligible,
        "k_star_one_standard_error": k_star,
        "baseline_failure_rate": baseline_failure,
        "baseline_unsafe_selected_mapping_proportion": baseline_unsafe,
        "baseline_extreme_p95_total_loss": baseline_extreme_p95,
    }
    return selection, pareto_rows


def save_template_figure(candidates: dict[int, dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, candidate in candidates.items():
        for template_index, shape in enumerate(candidate["templates"]):
            linewidth = 2.0 if k == 1 else 1.1
            plot_shape(ax, shape, linewidth=linewidth, label=f"K={k} t{template_index}")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Global mean and candidate templates")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close(fig)


def save_summary_figures(summary_rows: list[dict], comparisons: list[dict], agreement_rows: list[dict], output_dir: Path) -> None:
    figures = output_dir / "figures"
    all_rows = sorted([row for row in summary_rows if row["subset"] == "all" and row.get("n_valid", 0) > 0], key=lambda row: row["k"])
    extreme_rows = sorted([row for row in summary_rows if row["subset"] == "extreme" and row.get("n_valid", 0) > 0], key=lambda row: row["k"])
    if not all_rows:
        return
    k = [row["k"] for row in all_rows]

    def line_plot(y_key: str, ylabel: str, filename: str, with_error: bool = False) -> None:
        fig, ax = plt.subplots(figsize=(6, 4))
        y = [row[y_key] for row in all_rows]
        if with_error:
            ax.errorbar(k, y, yerr=[row["standard_error"] for row in all_rows], marker="o", capsize=4)
        else:
            ax.plot(k, y, marker="o")
        ax.set_xlabel("number of templates K")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        figures.mkdir(parents=True, exist_ok=True)
        plt.savefig(figures / filename, dpi=200)
        plt.close(fig)

    line_plot("mean_total_loss", "mean registration cost", "mean_cost_with_se.png", True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k, [row["median_total_loss"] for row in all_rows], marker="o", label="median")
    ax.plot(k, [row["p90_total_loss"] for row in all_rows], marker="o", label="90th percentile")
    ax.plot(k, [row["p95_total_loss"] for row in all_rows], marker="o", label="95th percentile")
    ax.set_xlabel("number of templates K")
    ax.set_ylabel("registration cost")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures / "cost_quantiles.png", dpi=200)
    plt.close(fig)
    line_plot("failure_rate", "failure rate", "failure_rate.png")
    line_plot("mean_runtime_seconds", "runtime seconds", "runtime.png")
    line_plot("mean_displacement", "mean displacement", "mean_displacement.png")
    line_plot("mean_weighted_regularisation", "mean weighted regularisation", "mean_weighted_regularisation.png")

    if extreme_rows:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([row["k"] for row in extreme_rows], [row["mean_total_loss"] for row in extreme_rows], marker="o", label="extreme mean")
        ax.plot([row["k"] for row in extreme_rows], [row["p95_total_loss"] for row in extreme_rows], marker="o", label="extreme p95")
        ax.set_xlabel("number of templates K")
        ax.set_ylabel("extreme-target cost")
        ax.grid(True, alpha=0.25)
        ax.legend()
        plt.tight_layout()
        plt.savefig(figures / "extreme_performance.png", dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(k, [row["mean_total_loss"] for row in all_rows])
    for row in all_rows:
        ax.annotate(f"K={row['k']}", (row["k"], row["mean_total_loss"]))
    ax.set_xlabel("number of templates")
    ax.set_ylabel("mean registration cost")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures / "pareto_templates_vs_cost.png", dpi=200)
    plt.close(fig)

    agreement_by_k = []
    for kk in sorted({row["k"] for row in agreement_rows}):
        subset = [row for row in agreement_rows if row["k"] == kk]
        agreement_by_k.append((kk, float(np.mean([row["agrees_with_posterior"] for row in subset]))))
    if agreement_by_k:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([item[0] for item in agreement_by_k], [item[1] for item in agreement_by_k])
        ax.set_xlabel("number of templates K")
        ax.set_ylabel("agreement proportion")
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(figures / "posterior_energy_agreement.png", dpi=200)
        plt.close(fig)


def write_preflight_report(output_dir: Path, summary_rows: list[dict], comparison_rows: list[dict], rows: list[dict]) -> None:
    failures = [row for row in rows if row.get("failure_reason")]
    selected = [row for row in rows if row["selected_template"]]
    lines = [
        "# Airfoil Template Selection Preflight",
        "",
        "This preflight evaluates every template for K=1, K=2, K=6, and K=9 on one ordinary held-out target and one extreme held-out target.",
        "",
        "## Summary",
        "",
    ]
    for row in summary_rows:
        if row["subset"] == "all":
            lines.append(
                f"- K={row['k']}: n_valid={row.get('n_valid', 0)}/{row['n_targets']}, "
                f"mean_total={row.get('mean_total_loss', float('nan')):.6g}, "
                f"failure_rate={row.get('failure_rate', float('nan')):.3g}"
            )
    lines.extend(["", "## Failures", ""])
    if failures:
        for row in failures:
            lines.append(
                f"- target {row['target_index']} {row['filename']}, K={row['k']}, template={row['template_index']}: "
                f"{row['failure_reason']}, min_jac={float(row['minimum_jacobian']):.6g}"
            )
    else:
        lines.append("- None.")
    lines.extend(["", "## Selected Templates", ""])
    for row in selected:
        lines.append(
            f"- target {row['target_index']} {row['filename']}, K={row['k']}: "
            f"template {row['template_index']}, total={float(row['total_loss']):.6g}, "
            f"min_jac={float(row['minimum_jacobian']):.6g}"
        )
    lines.extend(["", "## Paired Deltas Versus K=1", ""])
    for row in comparison_rows:
        lines.append(
            f"- target {row['target_index']}, K={row['k']}: delta_total={row['delta_total_loss']:.6g}, "
            f"delta_attachment={row['delta_attachment']:.6g}, delta_regularisation={row['delta_regularisation']:.6g}"
        )
    (output_dir / "preflight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_report(
    output_dir: Path,
    selection: dict,
    summary_rows: list[dict],
    delta_summary_rows: list[dict],
    bootstrap_rows: list[dict],
    pareto_rows: list[dict],
) -> None:
    all_rows = sorted([row for row in summary_rows if row["subset"] == "all" and row.get("n_valid", 0)], key=lambda row: row["k"])
    extreme_rows = sorted([row for row in summary_rows if row["subset"] == "extreme" and row.get("n_valid", 0)], key=lambda row: row["k"])
    lines = [
        "# Airfoil Template Registration Validation",
        "",
        "This report compares saved PCA-GMM template sets using the fixed 2D SVF registration settings. No PCA or GMM model was refit for this study.",
        "",
        "## K Selection",
        "",
        f"- Minimum mean-cost K: {selection['k_minimum_mean_cost']}",
        f"- One-standard-error K*: {selection['k_star_one_standard_error']}",
        f"- One-SE threshold: {selection['one_standard_error_threshold']:.6g}",
        f"- Eligible K values: {selection['eligible_k_values']}",
        "",
        "## Per-K Summary",
        "",
    ]
    for row in all_rows:
        lines.append(
            f"- K={row['k']}: mean={row['mean_total_loss']:.6g}, SE={row['standard_error']:.6g}, "
            f"median={row['median_total_loss']:.6g}, p95={row['p95_total_loss']:.6g}, "
            f"failure_rate={row['failure_rate']:.3g}, unsafe={row['unsafe_selected_mapping_count']}/{row['n_targets']}"
        )
    lines.extend(["", "## Extreme Subset", ""])
    for row in extreme_rows:
        lines.append(
            f"- K={row['k']}: mean={row['mean_total_loss']:.6g}, p95={row['p95_total_loss']:.6g}, "
            f"mean_reg={row['mean_weighted_regularisation']:.6g}, mean_disp={row['mean_displacement']:.6g}"
        )
    lines.extend(["", "## Paired Deltas Versus K=1", ""])
    for row in delta_summary_rows:
        if row["subset"] == "all":
            lines.append(
                f"- K={row['k']}: mean_delta={row['mean_delta_total_loss']:.6g}, "
                f"median_delta={row['median_delta_total_loss']:.6g}, improved={row['proportion_improved']:.3g}, "
                f"worsened={row['proportion_worsened']:.3g}"
            )
    lines.extend(["", "## Bootstrap Intervals", ""])
    for row in bootstrap_rows:
        if row["metric"] == "delta_total_loss":
            lines.append(
                f"- K={row['k']} delta_total_loss: mean={row['observed_mean']:.6g}, "
                f"95% CI=({row['ci95_low']:.6g}, {row['ci95_high']:.6g})"
            )
    lines.extend(["", "## Pareto Summary", ""])
    for row in pareto_rows:
        label = "frontier" if row["pareto_frontier"] else "dominated"
        lines.append(
            f"- K={row['k']}: {label}, mean={row['mean_total_loss']:.6g}, "
            f"p95={row['p95_total_loss']:.6g}, runtime={row['mean_runtime_seconds']:.3g}s"
        )
    lines.extend(
        [
            "",
            "## Interpretation Note",
            "",
            "The selected K is application-driven: it depends on held-out registration energy, validity, and extreme-target behaviour. It should not be interpreted as a universally true number of airfoil families.",
        ]
    )
    (output_dir / "final_k_selection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Held-out application-driven airfoil template registration validation.")
    parser.add_argument("--mode", choices=["preflight", "full"], default="preflight")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--gmm-dir", default=str(DEFAULT_GMM_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--k", default="1,2,6,9")
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=15)
    parser.add_argument("--num-iters", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--lambda-reg", type=float, default=1e-3)
    parser.add_argument("--early-stop-tol", type=float, default=1e-10)
    parser.add_argument("--early-stop-patience", type=int, default=40)
    parser.add_argument("--jacobian-resolution", type=int, default=41)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k_values = parse_k_list(args.k)
    unknown = [k for k in k_values if k not in CANDIDATES]
    if unknown:
        raise ValueError(f"Unsupported K values: {unknown}")
    output_root = Path(args.output_dir)
    output_dir = output_root / args.mode
    gmm_dir = Path(args.gmm_dir)
    for child in ["runs", "targets", "templates", "figures", "selected_templates", "representative_examples"]:
        (output_dir / child).mkdir(parents=True, exist_ok=True)

    dataset = load_airfoil_dataset(Path(args.data_path))
    train_indices = np.load(gmm_dir / "train_indices.npy").astype(int)
    target_indices = select_targets(dataset, gmm_dir, args.mode)
    extreme_heldout = set(np.load(gmm_dir / "extreme_group_heldout_indices.npy").astype(int).tolist())
    distances = distance_to_global_mean(dataset, train_indices)
    categories = target_categories(dataset, gmm_dir, target_indices)
    candidates = load_candidate_templates(dataset, gmm_dir, train_indices, k_values)
    all_template_shapes = [shape for candidate in candidates.values() for shape in candidate["templates"]]
    bounds = registration_bounds(all_template_shapes + [dataset["shapes"].reshape(-1, 2)], margin=0.08)
    config = {
        "grid_size": args.grid_size,
        "num_steps": args.num_steps,
        "num_iters": args.num_iters,
        "lr": args.lr,
        "lambda_reg": args.lambda_reg,
        "early_stop_tol": args.early_stop_tol,
        "early_stop_patience": args.early_stop_patience,
        "jacobian_resolution": args.jacobian_resolution,
        "jacobian_nonpositive_tol": -1e-6,
        "severe_negative_thickness": 1e-3,
        "bounds": bounds,
        "device": "cpu",
        "optimizer": "Adam",
        "data_attachment": "corresponding_point_mse",
        "mode": args.mode,
        "k_values": k_values,
        "seed": args.seed,
    }
    write_json(output_dir / "config.json", config)

    template_meta = {}
    template_plot_entries = []
    for k, candidate in candidates.items():
        np.savez(
            output_dir / "templates" / f"k{k}_templates.npz",
            templates=candidate["templates"],
            weights=candidate["weights"],
            validity=np.array([json.dumps(row, sort_keys=True) for row in candidate["validity"]]),
            gmm_path=candidate["gmm_path"],
        )
        np.savez(
            output_dir / "selected_templates" / f"k{k}_templates.npz",
            templates=candidate["templates"],
            weights=candidate["weights"],
            gmm_path=candidate["gmm_path"],
        )
        template_meta[str(k)] = {
            "spec": candidate["spec"],
            "weights": candidate["weights"].tolist(),
            "gmm_path": candidate["gmm_path"],
            "validity": candidate["validity"],
        }
        for template_index, shape in enumerate(candidate["templates"]):
            template_plot_entries.append((f"K={k} t{template_index}", shape))
    write_json(output_dir / "templates" / "template_metadata.json", template_meta)
    save_template_figure(candidates, output_dir / "figures" / "candidate_templates.png")

    target_rows = []
    for index in target_indices:
        target_rows.append(
            {
                "target_index": int(index),
                "filename": dataset["filenames"][index],
                "airfoil_name": dataset["names"][index],
                "distance_to_global_mean": float(distances[index]),
                "previous_extreme_heldout": int(index) in extreme_heldout,
                "target_category": categories[int(index)],
            }
        )
        np.save(output_dir / "targets" / f"target_{index}.npy", dataset["shapes"][index])
    save_csv(output_dir / "targets" / f"{args.mode}_targets.csv", target_rows)
    save_csv(output_dir / "representative_examples" / "target_index.csv", target_rows)

    posterior_cache = {}
    for k, candidate in candidates.items():
        if k == 1:
            posterior_cache[k] = np.ones((len(target_indices), 1))
        else:
            posterior_cache[k] = gmm_posteriors_for_shapes(
                dataset["shapes"][target_indices],
                gmm_dir / PCA_Q20,
                Path(candidate["gmm_path"]),
                candidate["spec"]["covariance_type"],
            )

    rows = []
    for local_target, target_index in enumerate(target_indices):
        target = dataset["shapes"][target_index]
        filename = dataset["filenames"][target_index]
        for k, candidate in candidates.items():
            posteriors = posterior_cache[k][local_target]
            highest_posterior_template = int(np.argmax(posteriors))
            for template_index, source in enumerate(candidate["templates"]):
                run_id = f"target{target_index}_k{k}_template{template_index}"
                result = load_or_run_registration(
                    source,
                    target,
                    config,
                    args.seed,
                    output_dir / "runs" / run_id,
                    output_dir / "figures" / run_id,
                    f"{filename}: K={k} template {template_index}",
                    args.force,
                )
                metrics = result["metrics"]
                rows.append(
                    {
                        "target_index": int(target_index),
                        "filename": filename,
                        "airfoil_name": dataset["names"][target_index],
                        "previous_extreme_heldout": int(target_index) in extreme_heldout,
                        "target_category": categories[int(target_index)],
                        "k": k,
                        "template_index": template_index,
                        "posterior_probability": float(posteriors[template_index]),
                        "maximum_posterior_template": highest_posterior_template,
                        "is_maximum_posterior_template": template_index == highest_posterior_template,
                        "skipped_existing": result["skipped"],
                        **metrics,
                    }
                )

    for row in rows:
        row["selected_template"] = False
        row["minimum_energy_template_agrees_with_posterior"] = False
    agreement_rows = []
    for target_index in target_indices:
        for k in k_values:
            candidates_for_pair = [row for row in rows if row["target_index"] == int(target_index) and row["k"] == k]
            valid = [row for row in candidates_for_pair if finite_valid(row)]
            best = min(valid, key=lambda row: float(row["total_loss"]), default=None)
            if best is None:
                best = min(candidates_for_pair, key=lambda row: float(row["total_loss"]))
            best["selected_template"] = True
            agrees = int(best["template_index"]) == int(best["maximum_posterior_template"])
            for row in candidates_for_pair:
                row["minimum_energy_template_agrees_with_posterior"] = agrees
            if k > 1:
                agreement_rows.append(
                    {
                        "target_index": int(target_index),
                        "filename": best["filename"],
                        "previous_extreme_heldout": best["previous_extreme_heldout"],
                        "k": k,
                        "selected_template": int(best["template_index"]),
                        "maximum_posterior_template": int(best["maximum_posterior_template"]),
                        "agrees_with_posterior": agrees,
                        "selected_posterior_probability": float(best["posterior_probability"]),
                    }
                )

    summary_rows = summarise_selected(rows, target_rows, k_values)
    comparison_rows = paired_comparisons(rows, target_rows, k_values)
    bootstrap_rows = bootstrap_deltas(comparison_rows, k_values, args.seed, args.bootstrap_samples)
    delta_summary_rows = paired_delta_summary(comparison_rows, k_values)
    final_selection, pareto_rows = select_k_and_pareto(summary_rows, template_meta, k_values)

    save_csv(output_dir / "registration_results.csv", rows)
    save_csv(output_dir / "per_k_summary.csv", summary_rows)
    save_csv(output_dir / "paired_comparison_vs_k1.csv", comparison_rows)
    save_csv(output_dir / "paired_bootstrap_ci.csv", bootstrap_rows)
    save_csv(output_dir / "paired_delta_summary.csv", delta_summary_rows)
    save_csv(output_dir / "posterior_energy_agreement.csv", agreement_rows)
    save_csv(output_dir / "pareto_summary.csv", pareto_rows)
    save_summary_figures(summary_rows, comparison_rows, agreement_rows, output_dir)
    selected_rows = [row for row in rows if row["selected_template"]]
    save_csv(output_dir / "selected_registration_results.csv", selected_rows)
    if args.mode == "preflight":
        write_preflight_report(output_dir, summary_rows, comparison_rows, rows)
    if args.mode == "full":
        save_csv(output_dir / "full_registration_results.csv", rows)
        save_csv(output_dir / "registration_summary_by_k.csv", [row for row in summary_rows if row["subset"] == "all"])
        save_csv(output_dir / "registration_summary_by_target_group.csv", summary_rows)
        save_csv(output_dir / "paired_comparisons_vs_k1.csv", comparison_rows)
        save_csv(output_dir / "bootstrap_intervals.csv", bootstrap_rows)
        save_csv(output_dir / "extreme_subset_summary.csv", [row for row in summary_rows if row["subset"] == "extreme"])
        write_json(output_dir / "final_k_selection.json", final_selection)
        write_final_report(output_dir, final_selection, summary_rows, delta_summary_rows, bootstrap_rows, pareto_rows)

    print(
        json.dumps(
            {
                "mode": args.mode,
                "target_count": len(target_indices),
                "k_values": k_values,
                "attempted_registrations": len(rows),
                "failure_count": sum(1 for row in rows if row.get("failure_reason")),
                "selected_failure_count": sum(1 for row in rows if row["selected_template"] and row.get("failure_reason")),
                "output_dir": str(output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
