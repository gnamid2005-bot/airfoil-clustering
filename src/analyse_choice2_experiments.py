import argparse
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from choice1_utils import flatten_dict, read_json, write_csv
from run_choice2_experiments import DEFAULT_ROOT, PILOT_GRID_SIZES


matplotlib.use("Agg")


METRICS = [
    "registration.final_symmetric_chamfer",
    "registration.final_source_to_target_chamfer_component",
    "registration.final_target_to_source_chamfer_component",
    "registration.final_total_loss",
    "registration.final_data_attachment",
    "registration.final_raw_regulariser",
    "registration.final_weighted_regulariser",
    "registration.runtime_seconds",
    "registration.time_per_iteration_seconds",
    "actual_learnable_parameter_count",
    "deformation_quality.mean_velocity_magnitude",
    "deformation_quality.maximum_velocity_magnitude",
    "deformation_quality.mean_velocity_gradient_magnitude",
    "deformation_quality.maximum_velocity_gradient_magnitude",
    "deformation_quality.jacobian_determinant.minimum",
    "deformation_quality.jacobian_determinant.mean",
    "deformation_quality.jacobian_determinant.maximum",
    "deformation_quality.jacobian_determinant.proportion_less_equal_zero",
    "deformation_quality.jacobian_determinant.proportion_below_0_1",
    "memory.peak_memory_mb",
]


def load_runs(root: Path) -> list[dict]:
    rows = []
    for metrics_path in sorted(root.glob("grid_size_*/seed_*/metrics.json")):
        try:
            metrics = read_json(metrics_path)
            config = read_json(metrics_path.parent / "config.json")
        except Exception as exc:
            rows.append(
                {
                    "status": "failed_to_read",
                    "run_dir": str(metrics_path.parent),
                    "error": str(exc),
                }
            )
            continue
        rows.append(
            {
                "run_dir": str(metrics_path.parent),
                **flatten_dict(config),
                **flatten_dict(metrics),
            }
        )
    return rows


def successful(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("status") == "completed"]


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else math.nan


def std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def make_summary(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in successful(rows):
        grouped[int(row["grid_size"])].append(row)

    out = []
    for grid_size, group in sorted(grouped.items()):
        summary = {
            "grid_size": grid_size,
            "successful_runs": len(group),
        }
        for metric in METRICS:
            values = [
                float(row[metric])
                for row in group
                if metric in row and row[metric] not in (None, "")
            ]
            summary[f"{metric}.mean"] = mean(values)
            summary[f"{metric}.std"] = std(values)
            summary[f"{metric}.min"] = min(values) if values else math.nan
            summary[f"{metric}.max"] = max(values) if values else math.nan
        out.append(summary)
    return out


def paired_adjacent_comparisons(rows: list[dict]) -> list[dict]:
    by_key = {
        (int(row["grid_size"]), int(row["seed"])): row
        for row in successful(rows)
    }
    out = []
    for lower, upper in zip(PILOT_GRID_SIZES, PILOT_GRID_SIZES[1:]):
        seed_rows = []
        for seed in sorted({int(row["seed"]) for row in successful(rows)}):
            left = by_key.get((lower, seed))
            right = by_key.get((upper, seed))
            if left is None or right is None:
                continue
            left_chamfer = float(left["registration.final_symmetric_chamfer"])
            right_chamfer = float(right["registration.final_symmetric_chamfer"])
            left_reg = float(left["registration.final_raw_regulariser"])
            right_reg = float(right["registration.final_raw_regulariser"])
            left_time = float(left["registration.runtime_seconds"])
            right_time = float(right["registration.runtime_seconds"])
            left_params = float(left["actual_learnable_parameter_count"])
            right_params = float(right["actual_learnable_parameter_count"])
            seed_rows.append(
                {
                    "from_grid_size": lower,
                    "to_grid_size": upper,
                    "seed": seed,
                    "chamfer_change": right_chamfer - left_chamfer,
                    "percent_chamfer_change": (
                        100.0 * (right_chamfer - left_chamfer) / left_chamfer
                        if left_chamfer != 0
                        else math.nan
                    ),
                    "raw_regulariser_change": right_reg - left_reg,
                    "runtime_multiplier": right_time / left_time if left_time else math.nan,
                    "parameter_count_multiplier": right_params / left_params if left_params else math.nan,
                    "higher_grid_improves_chamfer": right_chamfer < left_chamfer,
                }
            )
        agree = (
            len(seed_rows) > 0
            and (
                all(row["higher_grid_improves_chamfer"] for row in seed_rows)
                or not any(row["higher_grid_improves_chamfer"] for row in seed_rows)
            )
        )
        for row in seed_rows:
            row["both_seeds_agree_direction"] = agree
            out.append(row)
    return out


def plot_metric(rows: list[dict], metric: str, path: Path, title: str, ylabel: str) -> None:
    grouped = defaultdict(list)
    for row in successful(rows):
        if metric in row and row[metric] not in (None, ""):
            grouped[int(row["grid_size"])].append(float(row[metric]))
    if not grouped:
        return

    xs = sorted(grouped)
    ys = [mean(grouped[x]) for x in xs]
    yerr = [std(grouped[x]) for x in xs]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3)
    ax.set_xlabel("Grid size per dimension")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def make_summary_figures(rows: list[dict], root: Path) -> None:
    figures = root / "summary_figures"
    plot_metric(
        rows,
        "registration.final_symmetric_chamfer",
        figures / "final_chamfer_vs_grid_size.png",
        "Final Chamfer versus grid size",
        "Final Chamfer",
    )
    plot_metric(
        rows,
        "registration.final_raw_regulariser",
        figures / "raw_regulariser_vs_grid_size.png",
        "Raw regulariser versus grid size",
        "Raw regulariser",
    )
    plot_metric(
        rows,
        "registration.runtime_seconds",
        figures / "runtime_vs_grid_size.png",
        "Runtime versus grid size",
        "Runtime (seconds)",
    )
    plot_metric(
        rows,
        "actual_learnable_parameter_count",
        figures / "parameter_count_vs_grid_size.png",
        "Parameter count versus grid size",
        "Learnable scalar parameters",
    )
    plot_metric(
        rows,
        "deformation_quality.mean_velocity_gradient_magnitude",
        figures / "velocity_gradient_vs_grid_size.png",
        "Velocity-gradient magnitude versus grid size",
        "Mean gradient magnitude",
    )
    plot_metric(
        rows,
        "deformation_quality.jacobian_determinant.minimum",
        figures / "jacobian_minimum_vs_grid_size.png",
        "Minimum Jacobian determinant versus grid size",
        "Minimum determinant",
    )
    make_chamfer_runtime_plot(rows, figures / "chamfer_vs_runtime.png")


def make_chamfer_runtime_plot(rows: list[dict], path: Path) -> None:
    ok = successful(rows)
    if not ok:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in ok:
        ax.scatter(
            float(row["registration.runtime_seconds"]),
            float(row["registration.final_symmetric_chamfer"]),
            label=f"g={row['grid_size']} seed={row['seed']}",
        )
    ax.set_xlabel("Runtime (seconds)")
    ax.set_ylabel("Final Chamfer")
    ax.set_title("Chamfer versus runtime")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def markdown_table(rows: list[dict], columns: list[str], limit: int = 20) -> str:
    if not rows:
        return "_No data available._"
    rows = rows[:limit]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *body])


def make_report(root: Path, rows: list[dict], summary: list[dict], paired: list[dict]) -> None:
    completed = len(successful(rows))
    failed = len([row for row in rows if row.get("status") not in ("completed", None)])
    report = [
        "# Choice 2 Grid-Resolution Pilot Report",
        "",
        "## Experimental Question",
        "How does the cubic B-spline control-grid resolution affect registration accuracy, smoothness, runtime, and deformation quality?",
        "",
        "## Controlled Variables",
        "All runs fix cube-to-bunny registration, 5000 source and target points, curvature-aware target sampling with strength 4.0, seeds 0 and 1, Adam, learning rate 0.005, 800 iterations, 20 Euler steps, and the existing lambda schedule. Only grid size changes.",
        "",
        "## Completion",
        f"Completed runs: {completed}. Failed or unreadable runs: {failed}.",
        "",
        "## Summary by Grid Size",
        markdown_table(
            summary,
            [
                "grid_size",
                "successful_runs",
                "registration.final_symmetric_chamfer.mean",
                "registration.final_raw_regulariser.mean",
                "registration.runtime_seconds.mean",
                "actual_learnable_parameter_count.mean",
                "deformation_quality.jacobian_determinant.minimum.mean",
            ],
        ),
        "",
        "## Adjacent Grid Comparisons",
        markdown_table(
            paired,
            [
                "from_grid_size",
                "to_grid_size",
                "seed",
                "percent_chamfer_change",
                "raw_regulariser_change",
                "runtime_multiplier",
                "parameter_count_multiplier",
                "both_seeds_agree_direction",
            ],
        ),
        "",
        "## Interpretation Prompts",
        "- Is 8^3 too coarse?",
        "- Does 10^3 adequately represent the deformation?",
        "- Does 12^3 or 16^3 improve local detail?",
        "- Does 24^3 provide further benefit or mainly cost?",
        "- Do Jacobian diagnostics suggest folding risk?",
        "",
        "## Caution",
        "This is a pilot-scale report with two seeds. Use cautious language and prefer conclusions that agree across both seeds.",
    ]
    (root / "choice2_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyse Choice 2 grid-size experiment outputs."
    )
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    rows = load_runs(root)
    summary = make_summary(rows)
    paired = paired_adjacent_comparisons(rows)

    write_csv(root / "choice2_results.csv", rows)
    write_csv(root / "choice2_summary.csv", summary)
    write_csv(root / "choice2_paired_comparisons.csv", paired)
    make_summary_figures(rows, root)
    make_report(root, rows, summary, paired)

    print(f"runs read: {len(rows)}")
    print(f"completed runs: {len(successful(rows))}")
    print(f"wrote: {root / 'choice2_report.md'}")


if __name__ == "__main__":
    main()

