import argparse
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from choice1_utils import (
    DEFAULT_ROOT,
    flatten_dict,
    read_json,
    write_csv,
)


matplotlib.use("Agg")


METRIC_KEYS = [
    "registration.final_symmetric_chamfer",
    "registration.final_data_attachment",
    "registration.final_raw_regulariser",
    "registration.final_weighted_regulariser",
    "registration.runtime_seconds",
    "mesh_coverage.reference_to_target_distance.mean",
    "sampling_distribution.curvature.high_curvature_point_proportion",
]


def load_runs(root: Path) -> list[dict]:
    rows = []
    for metrics_path in sorted(root.glob("point_count_*/seed_*/*/metrics.json")):
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

        row = {
            "run_dir": str(metrics_path.parent),
            **flatten_dict(config),
            **flatten_dict(metrics),
        }
        rows.append(row)
    return rows


def successful(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("status") == "completed"]


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else math.nan


def std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def group_key(row: dict) -> tuple:
    return (
        row.get("point_count"),
        row.get("sampling_method"),
        row.get("curvature_strength"),
    )


def make_summary_rows(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in successful(rows):
        grouped[group_key(row)].append(row)

    summary = []
    for key, group in sorted(grouped.items()):
        point_count, method, strength = key
        out = {
            "point_count": point_count,
            "sampling_method": method,
            "curvature_strength": strength,
            "successful_runs": len(group),
        }
        for metric in METRIC_KEYS:
            values = [
                float(row[metric])
                for row in group
                if metric in row and row[metric] not in (None, "")
            ]
            out[f"{metric}.mean"] = mean(values)
            out[f"{metric}.std"] = std(values)
            out[f"{metric}.min"] = min(values) if values else math.nan
            out[f"{metric}.max"] = max(values) if values else math.nan
        summary.append(out)
    return summary


def compatible_for_sampling_pair(uniform: dict, curvature: dict) -> bool:
    keys = [
        "point_count",
        "seed",
        "source",
        "target",
        "grid_size",
        "num_steps",
        "num_iters",
        "lr",
        "optimizer",
        "lambda_schedule",
    ]
    return all(uniform.get(key) == curvature.get(key) for key in keys)


def paired_sampling_differences(rows: list[dict]) -> list[dict]:
    ok = successful(rows)
    by_key = defaultdict(dict)
    for row in ok:
        key = (row.get("point_count"), row.get("seed"))
        if row.get("sampling_method") == "uniform":
            by_key[key]["uniform"] = row
        else:
            by_key[key].setdefault("curvature", []).append(row)

    paired = []
    for key, group in sorted(by_key.items()):
        uniform = group.get("uniform")
        if uniform is None:
            continue
        for curvature in group.get("curvature", []):
            if not compatible_for_sampling_pair(uniform, curvature):
                continue
            u_data = float(uniform["registration.final_data_attachment"])
            c_data = float(curvature["registration.final_data_attachment"])
            u_reg = float(uniform["registration.final_raw_regulariser"])
            c_reg = float(curvature["registration.final_raw_regulariser"])
            paired.append(
                {
                    "point_count": key[0],
                    "seed": key[1],
                    "curvature_strength": curvature.get("curvature_strength"),
                    "curvature_minus_uniform_data_attachment": c_data - u_data,
                    "percent_difference_data_attachment_vs_uniform": (
                        100.0 * (c_data - u_data) / u_data
                        if u_data != 0
                        else math.nan
                    ),
                    "curvature_minus_uniform_raw_regulariser": c_reg - u_reg,
                    "percent_difference_raw_regulariser_vs_uniform": (
                        100.0 * (c_reg - u_reg) / u_reg
                        if u_reg != 0
                        else math.nan
                    ),
                    "compatible_training_settings": True,
                }
            )
    return paired


def point_count_comparisons(rows: list[dict]) -> list[dict]:
    out = []
    grouped = defaultdict(list)
    for row in successful(rows):
        key = (
            row.get("sampling_method"),
            row.get("curvature_strength"),
            row.get("seed"),
        )
        grouped[key].append(row)

    for key, group in grouped.items():
        group = sorted(group, key=lambda row: int(row["point_count"]))
        for previous, current in zip(group, group[1:]):
            prev_data = float(previous["registration.final_data_attachment"])
            cur_data = float(current["registration.final_data_attachment"])
            out.append(
                {
                    "sampling_method": key[0],
                    "curvature_strength": key[1],
                    "seed": key[2],
                    "from_point_count": previous["point_count"],
                    "to_point_count": current["point_count"],
                    "data_attachment_change": cur_data - prev_data,
                    "percent_change": (
                        100.0 * (cur_data - prev_data) / prev_data
                        if prev_data != 0
                        else math.nan
                    ),
                }
            )
    return out


def label(row: dict) -> str:
    if row.get("sampling_method") == "uniform":
        return "uniform"
    return f"curvature {row.get('curvature_strength')}"


def plot_metric_vs_point_count(rows: list[dict], metric: str, path: Path, title: str, ylabel: str) -> None:
    grouped = defaultdict(lambda: defaultdict(list))
    for row in successful(rows):
        if metric not in row or row[metric] in (None, ""):
            continue
        grouped[label(row)][int(row["point_count"])].append(float(row[metric]))

    if not grouped:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, values_by_count in sorted(grouped.items()):
        counts = sorted(values_by_count)
        means = [mean(values_by_count[count]) for count in counts]
        errors = [std(values_by_count[count]) for count in counts]
        ax.errorbar(counts, means, yerr=errors, marker="o", capsize=3, label=name)

    ax.set_xlabel("Point count")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def plot_metric_vs_strength(rows: list[dict], metric: str, path: Path, title: str, ylabel: str) -> None:
    grouped = defaultdict(lambda: defaultdict(list))
    for row in successful(rows):
        if row.get("sampling_method") != "curvature":
            continue
        if metric not in row or row[metric] in (None, ""):
            continue
        grouped[int(row["point_count"])][float(row["curvature_strength"])].append(float(row[metric]))

    if not grouped:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for count, values_by_strength in sorted(grouped.items()):
        strengths = sorted(values_by_strength)
        means = [mean(values_by_strength[strength]) for strength in strengths]
        errors = [std(values_by_strength[strength]) for strength in strengths]
        ax.errorbar(strengths, means, yerr=errors, marker="o", capsize=3, label=f"n={count}")

    ax.set_xlabel("Curvature strength")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def make_summary_figures(rows: list[dict], root: Path) -> None:
    figures = root / "summary_figures"
    plot_metric_vs_point_count(
        rows,
        "registration.final_symmetric_chamfer",
        figures / "final_chamfer_vs_point_count.png",
        "Final Chamfer versus point count",
        "Final Chamfer",
    )
    plot_metric_vs_strength(
        rows,
        "registration.final_symmetric_chamfer",
        figures / "final_chamfer_vs_curvature_strength.png",
        "Final Chamfer versus curvature strength",
        "Final Chamfer",
    )
    plot_metric_vs_point_count(
        rows,
        "registration.final_raw_regulariser",
        figures / "regulariser_vs_point_count.png",
        "Raw regulariser versus point count",
        "Raw regulariser",
    )
    plot_metric_vs_point_count(
        rows,
        "mesh_coverage.reference_to_target_distance.mean",
        figures / "mesh_coverage_vs_point_count.png",
        "Mesh coverage versus point count",
        "Reference-to-target mean distance",
    )
    plot_metric_vs_strength(
        rows,
        "sampling_distribution.curvature.high_curvature_point_proportion",
        figures / "high_curvature_proportion_vs_strength.png",
        "High-curvature sampling proportion versus strength",
        "High-curvature point proportion",
    )
    plot_metric_vs_point_count(
        rows,
        "registration.runtime_seconds",
        figures / "runtime_vs_point_count.png",
        "Runtime versus point count",
        "Runtime (seconds)",
    )


def markdown_table(rows: list[dict], columns: list[str], limit: int = 20) -> str:
    if not rows:
        return "_No data available._"
    rows = rows[:limit]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *body])


def make_report(root: Path, rows: list[dict], summary: list[dict], paired: list[dict], point_pairs: list[dict]) -> None:
    completed = len(successful(rows))
    failed = len([row for row in rows if row.get("status") not in ("completed", None)])
    report = []
    report.append("# Choice 1 Experiment Report")
    report.append("")
    report.append("## 1. Experimental question")
    report.append(
        "How do the number of source/target point-cloud samples and the target sampling distribution affect bunny registration?"
    )
    report.append("")
    report.append("## 2. Controlled variables")
    report.append(
        "Within each matched comparison the source shape, target mesh, optimiser, learning rate, SVF grid size, Euler steps, lambda schedule, regulariser, and iteration count are intended to be fixed. The runner also records whether the regenerated source tensor exactly matches the controlled source tensor."
    )
    report.append("")
    report.append("## 3. Independent variables")
    report.append(
        "The factorial sweep varies point count, random seed, sampling method, and curvature strength for curvature-aware sampling. Pairwise summaries should be read as one-factor-at-a-time comparisons."
    )
    report.append("")
    report.append("## 4. Number of completed and failed runs")
    report.append(f"Completed runs: {completed}. Failed or unreadable runs: {failed}.")
    report.append("")
    report.append("## 5. Sampling-distribution results")
    report.append(
        "Use high-curvature point proportion, nearest-neighbour distances, voxel occupancy, and regional point proportions to decide whether curvature-aware sampling actually moves samples into detailed regions. Do not infer this from overlays alone."
    )
    report.append("")
    report.append(markdown_table(summary, [
        "point_count",
        "sampling_method",
        "curvature_strength",
        "successful_runs",
        "sampling_distribution.curvature.high_curvature_point_proportion.mean",
        "mesh_coverage.reference_to_target_distance.mean.mean",
    ]))
    report.append("")
    report.append("## 6. Registration results")
    report.append(markdown_table(summary, [
        "point_count",
        "sampling_method",
        "curvature_strength",
        "successful_runs",
        "registration.final_symmetric_chamfer.mean",
        "registration.final_data_attachment.mean",
        "registration.final_raw_regulariser.mean",
    ]))
    report.append("")
    report.append("## 7. Point-count results")
    report.append(markdown_table(point_pairs, [
        "sampling_method",
        "curvature_strength",
        "seed",
        "from_point_count",
        "to_point_count",
        "percent_change",
    ]))
    report.append("")
    report.append("## 8. Curvature-strength results")
    report.append(markdown_table(paired, [
        "point_count",
        "seed",
        "curvature_strength",
        "percent_difference_data_attachment_vs_uniform",
        "percent_difference_raw_regulariser_vs_uniform",
    ]))
    report.append("")
    report.append("## 9. Regional results")
    report.append(
        "Regional labels are approximate coordinate-based regions: ears, head/face, torso, legs/feet, and tail. They are useful diagnostics, not semantic mesh annotations."
    )
    report.append("")
    report.append("## 10. Runtime trade-offs")
    report.append(
        "Runtime should be interpreted alongside point count because Chamfer distance scales with the number of pairwise distances."
    )
    report.append("")
    report.append("## 11. Limitations")
    report.append(
        "The report is automatically generated from completed runs. Strong claims require multiple seeds per configuration. Visual evidence, numerical evidence, and seed-consistent evidence should be distinguished."
    )
    report.append("")
    report.append("## 12. Conclusion")
    if completed == 0:
        report.append("No completed runs were found, so no conclusion can be drawn yet.")
    else:
        report.append(
            "This automated report provides the controlled evidence needed to answer whether curvature-aware sampling is useful. Prefer conclusions supported by paired differences across matched seeds rather than one run."
        )

    (root / "choice1_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyse controlled Choice 1 experiment outputs."
    )
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    rows = load_runs(root)
    summary = make_summary_rows(rows)
    paired = paired_sampling_differences(rows)
    point_pairs = point_count_comparisons(rows)

    write_csv(root / "choice1_runs.csv", rows)
    write_csv(root / "choice1_summary.csv", summary)
    write_csv(root / "choice1_paired_sampling_differences.csv", paired)
    write_csv(root / "choice1_point_count_comparisons.csv", point_pairs)
    make_summary_figures(rows, root)
    make_report(root, rows, summary, paired, point_pairs)

    print(f"runs read: {len(rows)}")
    print(f"completed runs: {len(successful(rows))}")
    print(f"wrote: {root / 'choice1_report.md'}")


if __name__ == "__main__":
    main()

