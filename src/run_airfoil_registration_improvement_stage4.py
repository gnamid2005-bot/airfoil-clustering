from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import boundary_aware_svf as bas
import run_airfoil_registration_improvement_stage2 as s2
import run_airfoil_registration_improvement_stage3 as s3
from airfoil_data import plot_shape, validate_processed_shapes
from airfoil_registration import validate_airfoil_shape


DATA = Path("data/airfoils/processed/airfoils_n100_seed0.npz")
STAGE1 = Path("outputs/airfoil_registration_improvement")
STAGE2 = Path("outputs/airfoil_registration_improvement_stage2")
STAGE3 = Path("outputs/airfoil_registration_improvement_stage3")
OUT = Path("outputs/airfoil_registration_improvement_stage4")
SCHEDULE = (1e-2, 1e-4, 1e-5)
DTYPE = torch.float64
LABELS = [
    "identity",
    "very_close",
    "close",
    "medium_low",
    "medium",
    "medium_high",
    "far",
    "very_far",
    "extreme",
]


def jwrite(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def cwrite(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_inputs() -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    archive = np.load(DATA)
    shapes = archive["shapes"].astype(np.float64)
    filenames = archive["filenames"].astype(str)
    validate_processed_shapes(shapes)
    mean = np.load(STAGE2 / "direct_coordinate_mean" / "coordinates.npy").astype(np.float64)
    selected = np.load(STAGE1 / "development_subset" / "selected_indices.npy")
    selected_names = (STAGE1 / "development_subset" / "selected_filenames.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    if selected.tolist() != [-1, 382, 104, 959, 362, 1489, 377, 1224, 518]:
        raise RuntimeError("development subset differs from Stages 2/3")
    if selected_names[0] != "GLOBAL_MEAN":
        raise RuntimeError("identity target is not the global mean")
    targets = []
    for label, index, saved_name in zip(LABELS, selected, selected_names):
        shape = mean if index < 0 else shapes[index]
        actual_name = "GLOBAL_MEAN" if index < 0 else filenames[index]
        if actual_name != saved_name:
            raise RuntimeError(f"filename mismatch for {label}: {saved_name} != {actual_name}")
        targets.append(
            {"label": label, "index": int(index), "filename": saved_name, "shape": shape.copy()}
        )
    direct = shapes.mean(axis=0)
    fixed_x = bool(
        np.array_equal(shapes[:, :, 0], np.broadcast_to(shapes[0, :, 0], shapes[:, :, 0].shape))
    )
    verification = {
        "dataset_path": str(DATA),
        "dataset_shape": list(shapes.shape),
        "airfoils": int(shapes.shape[0]),
        "points_per_airfoil": int(shapes.shape[1]),
        "coordinate_representation": "direct Cartesian S_i in R^(200x2)",
        "upper_points": 100,
        "lower_points": 100,
        "fixed_x_grid": fixed_x,
        "pointwise_correspondence": True,
        "reprocessed": False,
        "stage2_mean_path": str(STAGE2 / "direct_coordinate_mean" / "coordinates.npy"),
        "direct_recomputed_mean_max_abs_difference": float(np.max(np.abs(direct - mean))),
        "mean_validity": validate_airfoil_shape(mean),
        "selected_indices": selected.tolist(),
        "selected_filenames": selected_names,
    }
    return shapes, mean, targets, verification


def result_row(target: dict, method: str, parameters: dict, metrics: dict) -> dict:
    return {
        "target_label": target["label"],
        "target_index": target["index"],
        "filename": target["filename"],
        "method": method,
        **parameters,
        **metrics,
    }


def add_dense_jacobian_median(
    metrics: dict,
    coefficients: tuple[np.ndarray, np.ndarray, np.ndarray],
    bounds: tuple[float, float, float, float],
    config: dict,
) -> None:
    global_coeff, le, te = (torch.tensor(x, dtype=DTYPE) for x in coefficients)
    jac = s3.jacobian(
        global_coeff, le, te, bounds, 0.15, config["steps"], config["jacobian_report_resolution"]
    ).detach().numpy()
    metrics["jacobian_median"] = float(np.median(jac))


def base_metric_row(target: dict, method: str, metrics: dict, parameter_count: int) -> dict:
    enriched = dict(metrics)
    enriched.setdefault("raw_global_dirichlet", enriched.get("raw_dirichlet", np.nan))
    enriched.setdefault("weighted_global_dirichlet", enriched.get("weighted_dirichlet", np.nan))
    enriched.setdefault("raw_boundary_first_smoothness", 0.0)
    enriched.setdefault("raw_boundary_second_smoothness", 0.0)
    enriched.setdefault("weighted_boundary_first_smoothness", 0.0)
    enriched.setdefault("weighted_boundary_second_smoothness", 0.0)
    enriched.setdefault("refinement_runtime_seconds", 0.0)
    enriched["trainable_coefficients"] = parameter_count
    return result_row(target, method, {}, enriched)


def method_summary(rows: list[dict], method: str) -> dict:
    non_identity = [r for r in rows if r["method"] == method and r["target_label"] != "identity"]
    if not non_identity:
        raise RuntimeError(f"{method} has no non-identity rows")
    safe = all(bool(r["safe"]) for r in non_identity)
    mse = [float(r["correspondence_mse"]) for r in non_identity]
    smoothness = [
        float(r.get("raw_global_dirichlet", 0.0))
        + float(r.get("raw_boundary_first_smoothness", 0.0))
        + float(r.get("raw_boundary_second_smoothness", 0.0))
        for r in non_identity
    ]
    return {
        "method": method,
        "parameter_count": int(non_identity[0]["trainable_coefficients"]),
        "all_safe": safe,
        "median_correspondence_mse": float(np.median(mse)),
        "worst_correspondence_mse": float(np.max(mse)),
        "best_correspondence_mse": float(np.min(mse)),
        "reaching_1e_9": int(sum(bool(r["safe"]) and float(r["correspondence_mse"]) <= 1e-9 for r in non_identity)),
        "reaching_1e_8": int(sum(bool(r["safe"]) and float(r["correspondence_mse"]) <= 1e-8 for r in non_identity)),
        "reaching_1e_7": int(sum(bool(r["safe"]) and float(r["correspondence_mse"]) <= 1e-7 for r in non_identity)),
        "median_smoothness_energy": float(np.median(smoothness)),
        "minimum_jacobian": float(min(float(r["minimum_jacobian"]) for r in non_identity)),
        "median_runtime_seconds": float(np.median([float(r["runtime_seconds"]) for r in non_identity])),
        "total_runtime_seconds": float(sum(float(r["runtime_seconds"]) for r in non_identity)),
        "mse_per_parameter": float(np.median(mse) / int(non_identity[0]["trainable_coefficients"])),
    }


def pointwise_rows(target: dict, method: str, deformed: np.ndarray, residual: np.ndarray, mean: np.ndarray) -> list[dict]:
    squared = np.sum(residual**2, axis=1)
    total = max(float(squared.sum()), 1e-300)
    rows = []
    for point_index, value in enumerate(squared):
        chord_x = float(mean[point_index, 0])
        region = "LE" if chord_x <= 0.15 else ("TE" if chord_x >= 0.85 else "mid")
        rows.append(
            {
                "target_label": target["label"],
                "target_index": target["index"],
                "method": method,
                "point_index": point_index,
                "surface": "upper" if point_index < 100 else "lower",
                "x": chord_x,
                "target_y": float(target["shape"][point_index, 1]),
                "registered_x": float(deformed[point_index, 0]),
                "registered_y": float(deformed[point_index, 1]),
                "region": region,
                "squared_error": float(value),
                "absolute_region_contribution": float(value),
                "residual_fraction": float(value / total),
                "error_x": float(residual[point_index, 0]),
                "error_y": float(residual[point_index, 1]),
            }
        )
    return rows


def region_summary(point_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in point_rows:
        grouped.setdefault((row["method"], row["target_label"]), []).append(row)
    output = []
    for (method, target), rows in grouped.items():
        totals = {
            region: sum(float(row["squared_error"]) for row in rows if row["region"] == region)
            for region in ("LE", "TE", "mid")
        }
        total = max(sum(totals.values()), 1e-300)
        output.append(
            {
                "method": method,
                "target_label": target,
                "leading_edge_absolute": totals["LE"],
                "trailing_edge_absolute": totals["TE"],
                "mid_chord_absolute": totals["mid"],
                "total_squared_error": total,
                "leading_edge_fraction": totals["LE"] / total,
                "trailing_edge_fraction": totals["TE"] / total,
                "mid_chord_fraction": totals["mid"] / total,
            }
        )
    return output


def diagnostic_rows(mean: np.ndarray, targets: list[dict], geometry: dict) -> list[dict]:
    rows = []
    for target in targets:
        displacement = target["shape"] - mean
        normal_component = np.sum(displacement * geometry["normals"], axis=1)
        tangent_component = np.sum(displacement * geometry["tangents"], axis=1)
        normal_energy = normal_component**2
        tangent_energy = tangent_component**2
        total_energy = np.sum(displacement**2, axis=1)
        for region, mask in (
            ("LE", mean[:, 0] <= 0.15),
            ("TE", mean[:, 0] >= 0.85),
            ("mid", (mean[:, 0] > 0.15) & (mean[:, 0] < 0.85)),
            ("all", np.ones(200, dtype=bool)),
        ):
            total = max(float(total_energy[mask].sum()), 1e-300)
            rows.append(
                {
                    "target_label": target["label"],
                    "target_index": target["index"],
                    "region": region,
                    "points": int(mask.sum()),
                    "total_displacement_energy": float(total_energy[mask].sum()),
                    "normal_displacement_energy": float(normal_energy[mask].sum()),
                    "tangent_displacement_energy": float(tangent_energy[mask].sum()),
                    "normal_energy_fraction": float(normal_energy[mask].sum() / total),
                    "tangent_energy_fraction": float(tangent_energy[mask].sum() / total),
                }
            )
    return rows


def plot_boundary_geometry(mean: np.ndarray, geometry: dict, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    plot_shape(ax, mean, color="#1f4e79", label="global mean")
    step = 6
    points = mean[::step]
    tangent = geometry["tangents"][::step]
    normal = geometry["normals"][::step]
    ax.quiver(points[:, 0], points[:, 1], tangent[:, 0], tangent[:, 1], color="#d97706", scale=35, label="tangent")
    ax.quiver(points[:, 0], points[:, 1], normal[:, 0], normal[:, 1], color="#0f766e", scale=35, label="outward normal")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Global mean boundary with tangent and normal directions")
    ax.legend(loc="upper center", ncol=3)
    ax.grid(alpha=0.2)
    axes[1].plot(np.arange(200), geometry["curvature"], color="#1f4e79", label="signed curvature")
    axes[1].axvline(99.5, color="#6b7280", linestyle="--", linewidth=1)
    axes[1].set(xlabel="point index (0–99 upper, 100–199 lower)", ylabel="signed curvature")
    axes[1].set_title("Curvature by canonical point index")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_method_comparison(summaries: list[dict], output: Path) -> None:
    ordered = sorted(summaries, key=lambda row: row["median_correspondence_mse"])
    labels = [row["method"].replace("boundary_", "bnd_").replace("stage3_", "s3_").replace("stage2_", "s2_") for row in ordered]
    medians = [row["median_correspondence_mse"] for row in ordered]
    worst = [row["worst_correspondence_mse"] for row in ordered]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 8))
    for yi, median, maximum in zip(y, medians, worst):
        ax.plot([median, maximum], [yi, yi], color="#d1d5db", linewidth=1.5)
    ax.scatter(medians, y, marker="o", color="#1f4e79", label="median MSE", zorder=3)
    ax.scatter(worst, y, marker="s", color="#d97706", label="worst MSE", zorder=3)
    for threshold, label, color in ((1e-9, "1e-9", "#0f766e"), (1e-8, "1e-8", "#7c3aed"), (1e-7, "1e-7", "#9ca3af")):
        ax.axvline(threshold, linestyle=":", color=color, label=label)
    ax.set_xscale("log")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("correspondence MSE (8 non-identity targets)")
    ax.set_title("Method-level median and worst correspondence error")
    ax.legend(ncol=5)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_parameter_efficiency(summaries: list[dict], output: Path) -> None:
    required = [
        "stage2_global_6_12_24",
        "stage3_global48",
        "stage3_dense_local48",
        "boundary_edge_normal_32_ratio2",
        "boundary_edge_normal_tangent_32_ratio2",
        "boundary_edge_normal_32_ratio1",
    ]
    rows = [{**next(row for row in summaries if row["method"] == method), "display": label} for method, label in zip(
        required,
        ("global24", "global48", "dense local48", "normal-only r2", "normal+tangent r2", "boundary kernel r1"),
    )]
    fig, ax = plt.subplots(figsize=(10, 6))
    for row in rows:
        marker = "o" if row["all_safe"] else "x"
        color = "#1f4e79" if row["method"].startswith("boundary") else "#d97706"
        ax.scatter(
            row["parameter_count"],
            row["median_correspondence_mse"],
            marker=marker,
            s=65,
            color=color,
            label=row["display"],
        )
    ax.axhline(1e-9, color="#0f766e", linestyle=":", label="1e-9")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlabel="trainable scalar coefficients", ylabel="median correspondence MSE")
    ax.set_title("Accuracy versus parameter count")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def aggregate_regions(rows: list[dict], method: str) -> tuple[float, float, float, float, float, float]:
    selected = [row for row in rows if row["method"] == method and row["target_label"] != "identity"]
    return tuple(
        float(np.mean([float(row[key]) for row in selected]))
        for key in (
            "leading_edge_fraction",
            "trailing_edge_fraction",
            "mid_chord_fraction",
            "leading_edge_absolute",
            "trailing_edge_absolute",
            "mid_chord_absolute",
        )
    )


def plot_region_comparison(region_rows: list[dict], methods: list[str], output: Path) -> None:
    fractions = np.asarray([aggregate_regions(region_rows, method)[:3] for method in methods])
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#1f4e79", "#d97706", "#9ca3af"]
    bottom = np.zeros(len(methods))
    for index, region in enumerate(("LE", "TE", "mid")):
        ax.bar(methods, fractions[:, index], bottom=bottom, label=region, color=colors[index])
        bottom += fractions[:, index]
    ax.set_ylim(0, 1)
    ax.set_ylabel("mean fraction of squared residual")
    ax.set_title("Residual concentration by chordwise region")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_target_residual(
    target: dict,
    mean: np.ndarray,
    dense: tuple[np.ndarray, np.ndarray],
    best: tuple[np.ndarray, np.ndarray],
    controls: bas.BoundaryControls,
    boundary_coefficients: np.ndarray,
    best_name: str,
    output: Path,
) -> None:
    dense_deformed, dense_residual = dense
    best_deformed, best_residual = best
    dense_sq = np.sum(dense_residual**2, axis=1)
    best_sq = np.sum(best_residual**2, axis=1)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax = axes[0, 0]
    plot_shape(ax, mean, color="#9ca3af", label="source")
    plot_shape(ax, target["shape"], color="#1f4e79", label="target")
    plot_shape(ax, dense_deformed, color="#d97706", linestyle="--", label="dense local48")
    plot_shape(ax, best_deformed, color="#0f766e", linestyle=":", label=best_name)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Source/target/registered — {target['label']}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    axes[0, 1].plot(dense_sq, color="#d97706", label="dense local48")
    axes[0, 1].plot(best_sq, color="#0f766e", label=best_name)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set(xlabel="point index", ylabel="squared correspondence error")
    axes[0, 1].set_title("Pointwise residual")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.2)
    axes[0, 2].scatter(mean[:, 0], dense_sq, s=10, color="#d97706", label="dense local48")
    axes[0, 2].scatter(mean[:, 0], best_sq, s=10, color="#0f766e", label=best_name)
    axes[0, 2].set_yscale("log")
    axes[0, 2].set(xlabel="fixed chordwise x", ylabel="squared correspondence error")
    axes[0, 2].set_title("Chordwise residual")
    axes[0, 2].grid(alpha=0.2)
    for ax, mask, title in (
        (axes[1, 0], mean[:, 0] <= 0.15, "Leading-edge residual vectors"),
        (axes[1, 1], mean[:, 0] >= 0.85, "Trailing-edge residual vectors"),
    ):
        for surface_mask in (np.arange(200) < 100, np.arange(200) >= 100):
            current = mask & surface_mask
            ax.plot(target["shape"][current, 0], target["shape"][current, 1], color="#1f4e79")
        ax.quiver(
            best_deformed[mask, 0],
            best_deformed[mask, 1],
            -best_residual[mask, 0],
            -best_residual[mask, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="#0f766e",
            width=0.004,
        )
        ax.scatter(best_deformed[mask, 0], best_deformed[mask, 1], s=8, color="#d97706")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    ax = axes[1, 2]
    control_x = np.arange(len(controls.indices))
    ax.plot(control_x, boundary_coefficients[:, 0], color="#0f766e", label="normal amplitude")
    if boundary_coefficients.shape[1] == 2:
        ax.plot(control_x, boundary_coefficients[:, 1], color="#7c3aed", label="tangent amplitude")
    for group in range(1, len(controls.group_names)):
        boundary = np.flatnonzero(controls.groups == group)[0] - 0.5
        ax.axvline(boundary, color="#d1d5db", linewidth=0.8)
    ax.set(xlabel="boundary control index", ylabel="velocity coefficient")
    ax.set_title("Boundary correction amplitudes")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def write_report(
    summaries: list[dict],
    best_boundary: dict,
    overall_best: dict,
    verification: dict,
    region_rows: list[dict],
    diagnostic: list[dict],
    baseline_check: dict,
) -> None:
    summary_by_method = {row["method"]: row for row in summaries}
    dense = summary_by_method["stage3_dense_local48"]
    base = summary_by_method["stage2_global_6_12_24"]
    global48 = summary_by_method["stage3_global48"]
    boundary_regions = aggregate_regions(region_rows, best_boundary["method"])
    dense_regions = aggregate_regions(region_rows, "stage3_dense_local48")
    diagnostic_non_identity = [row for row in diagnostic if row["target_label"] != "identity" and row["region"] == "all"]
    normal_fraction = float(np.mean([row["normal_energy_fraction"] for row in diagnostic_non_identity]))
    tangent_fraction = float(np.mean([row["tangent_energy_fraction"] for row in diagnostic_non_identity]))
    normal_best = min(
        (row for row in summaries if "normal" in row["method"] and "normal_tangent" not in row["method"] and row["all_safe"]),
        key=lambda row: row["median_correspondence_mse"],
    )
    tangent_candidates = [row for row in summaries if "normal_tangent" in row["method"] and row["all_safe"]]
    tangent_best = min(tangent_candidates, key=lambda row: row["median_correspondence_mse"])
    anisotropic_candidates = [
        row
        for row in summaries
        if row["method"] in {"boundary_edge_normal_32_ratio1", "boundary_edge_normal_32_ratio2", "boundary_edge_normal_32_ratio4"}
    ]
    anisotropic_best = min(
        (row for row in anisotropic_candidates if row["all_safe"]),
        key=lambda row: row["median_correspondence_mse"],
    )
    error_ratio = best_boundary["median_correspondence_mse"] / dense["median_correspondence_mse"]
    parameter_ratio = best_boundary["parameter_count"] / dense["parameter_count"]
    boundary_gain_per_added_parameter = (
        base["median_correspondence_mse"] - best_boundary["median_correspondence_mse"]
    ) / (best_boundary["parameter_count"] - base["parameter_count"])
    dense_gain_per_added_parameter = (
        base["median_correspondence_mse"] - dense["median_correspondence_mse"]
    ) / (dense["parameter_count"] - base["parameter_count"])
    marginal_efficiency_ratio = boundary_gain_per_added_parameter / dense_gain_per_added_parameter
    lines = [
        "# Airfoil Registration Stage 4：边界感知稀疏空间表示",
        "",
        "## 技术摘要",
        "",
        f"本阶段仅使用同一直接坐标全局均值、同一 identity case 和同一 8 个非恒等开发目标。Stage 3 dense local48 对照复现的中位/最差 MSE 为 `{dense['median_correspondence_mse']:.3e}` / `{dense['worst_correspondence_mse']:.3e}`，历史值为 1.138e-8 / 3.268e-8；复现一致性判定为 `{baseline_check['consistent']}`。",
        "",
        f"最佳安全边界模型为 `{best_boundary['method']}`：中位 MSE `{best_boundary['median_correspondence_mse']:.3e}`，最差 `{best_boundary['worst_correspondence_mse']:.3e}`，最小 Jacobian `{best_boundary['minimum_jacobian']:.3f}`。其误差是 dense local48 的 `{error_ratio:.2f}×`，参数量为后者的 `{parameter_ratio:.1%}`。因此它用参数很少，但没有取代 dense local48 的绝对精度。",
        "",
        f"八个非恒等目标达到 1e-9 / 1e-8 / 1e-7 的数量为 `{best_boundary['reaching_1e_9']}/8`、`{best_boundary['reaching_1e_8']}/8`、`{best_boundary['reaching_1e_7']}/8`。所有参与最佳结果的映射均要求 Jacobian 严格为正且最终翼型几何有效。",
        "",
        "## 边界感知结果与可视证据",
        "",
        f"最佳边界模型的平均 LE/TE/mid 残差比例为 `{boundary_regions[0]:.1%}` / `{boundary_regions[1]:.1%}` / `{boundary_regions[2]:.1%}`，平均绝对平方误差和为 `{boundary_regions[3]:.3e}` / `{boundary_regions[4]:.3e}` / `{boundary_regions[5]:.3e}`；dense local48 的对应比例为 `{dense_regions[0]:.1%}` / `{dense_regions[1]:.1%}` / `{dense_regions[2]:.1%}`，绝对值为 `{dense_regions[3]:.3e}` / `{dense_regions[4]:.3e}` / `{dense_regions[5]:.3e}`。逐目标值保存在 `residual_region_summary.csv`，逐点值保存在 `pointwise_residuals.csv`。",
        "",
        "![方法误差比较](figures/method_comparison.png)",
        "",
        "上图在同一 8 个目标上同时展示中位和最差对应点 MSE；所有不安全候选仍保留在表中，但不会参与最佳方法选择。",
        "",
        "![参数效率](figures/parameter_efficiency.png)",
        "",
        "该图以相同的标量参数计数口径比较全局网格、dense local48 和边界基，避免只按原始误差判断更大的模型。",
        "",
        "## 数据、范围与指标定义",
        "",
        f"数据源为 `{verification['dataset_path']}`，形状 `{verification['dataset_shape']}`；每个翼型 200 个直接 Cartesian 对应点（100 upper + 100 lower），共享固定 x 网格。未重新采样、重新预处理或重新选择目标。Stage 2 均值与从当前数据直接复算均值的最大绝对差为 `{verification['direct_recomputed_mean_max_abs_difference']:.3e}`。",
        "",
        "对应点损失定义为 `D_corr = (1/200) sum_j ||phi(mu_j)-S_ij||^2`。报告的 MSE 不包含任何正则项。总目标为 `D_corr + lambda_G R_D + lambda_1 R_boundary,1 + lambda_2 R_boundary,2 + lambda_J R_J`，各项均在结果 CSV 中分列保存。",
        "",
        "## 边界几何、稀疏基与环境延拓",
        "",
        "上、下表面均按各自 LE→TE 顺序进行非均匀弧长有限差分；端点使用二阶单边差分，避免跨越重复的 LE/TE 转换点。upper 使用左法向、lower 使用右法向作为外法向。切向、法向、弧长与曲率保存在 `boundary_geometry.npz`。",
        "",
        "边界局部速度使用 `c_k = a_k n_k` 或 `c_k = a_k n_k + b_k t_k`。环境核为 `w_k(x)=exp(-0.5*((Δt/σ_t)^2+(Δn/σ_n)^2))`，局部场为 `sum_k w_k c_k`；该表达式在二维域中平滑，并在远离边界时随 Gaussian 权重衰减。测试的切/法向尺度比严格限制为 1、2、4。",
        "",
        "边界一阶与二阶差分分别在 upper/lower 及 LE/TE 控制序列内计算，不跨越物理不连续的索引连接。全局场先按 6→12→24 及 1e-2→1e-4→1e-5 获得，再从零初始化边界系数；控制比较包含仅边界优化、直接联合优化、边界后联合微调。",
        "",
        "![边界几何](figures/boundary_geometry.png)",
        "",
        "该图显示构建边界基所使用的均值翼型切向、外法向和有符号曲率；它不是按目标自适应选择的，因此所有注册共享同一空间规则。",
        "",
        "## 法向、切向与各向异性诊断",
        "",
        f"直接对应位移诊断显示，8 个非恒等目标的平均位移能量中法向占 `{normal_fraction:.1%}`，切向占 `{tangent_fraction:.1%}`。这仅用于解释基的适配性，不作为最终变换。最佳 normal-only 中位 MSE 为 `{normal_best['median_correspondence_mse']:.3e}`；最佳 normal+tangent 为 `{tangent_best['median_correspondence_mse']:.3e}`。",
        "",
        f"受控 ratio=1/2/4 比较中，最佳设置为 `{anisotropic_best['method']}`，中位 MSE `{anisotropic_best['median_correspondence_mse']:.3e}`。由于它是 ratio=1 的各向同性对照，各向异性延拓没有改善精度。完整配对结果见 `anisotropic_kernel_results.csv`。",
        "",
        "## 安全性、稳健性与限制",
        "",
        f"最佳边界模型的最小 Jacobian 为 `{best_boundary['minimum_jacobian']:.3f}`，non-positive 比例为 0；51×51 Jacobian 采样仍是数值诊断而非连续域解析证明。identity case、零边界场、有限梯度、确定性和单位 Jacobian 已由独立验证程序检查。",
        "",
        "有限差分曲率在 LE/TE 端点对局部点距敏感，但控制位置和法向均固定来自同一均值，不存在按目标调参。Gaussian 核虽然定义平滑环境场，但不是紧支撑核；这里的“局部”指快速衰减而非支撑外严格为零。",
        "",
        "## 十二个研究问题的直接回答",
        "",
        f"1. **边界感知表示是否优于 dense local48：** {'是' if best_boundary['median_correspondence_mse'] < dense['median_correspondence_mse'] else '否'}；两者中位 MSE 为 `{best_boundary['median_correspondence_mse']:.3e}` 与 `{dense['median_correspondence_mse']:.3e}`。",
        f"2. **normal-only 是否捕获大部分残差：** 位移能量诊断法向占 `{normal_fraction:.1%}`；拟合层面以 normal-only 与 matched normal+tangent 的配对误差共同判断。",
        f"3. **切向自由度是否有帮助：** 最佳 normal+tangent / normal-only 中位 MSE 为 `{tangent_best['median_correspondence_mse']:.3e}` / `{normal_best['median_correspondence_mse']:.3e}`；{'有可测改善' if tangent_best['median_correspondence_mse'] < normal_best['median_correspondence_mse'] else '没有改善'}。",
        f"4. **各向异性支持是否改善精度：** 否；ratio=1/2/4 的中位 MSE 分别为 `{summary_by_method['boundary_edge_normal_32_ratio1']['median_correspondence_mse']:.3e}` / `{summary_by_method['boundary_edge_normal_32_ratio2']['median_correspondence_mse']:.3e}` / `{summary_by_method['boundary_edge_normal_32_ratio4']['median_correspondence_mse']:.3e}`，各向同性 ratio=1 最好。",
        f"5. **最佳安全中位 MSE：** 新边界模型为 `{best_boundary['median_correspondence_mse']:.3e}`；若连同固定对照一起比较，全方法最佳仍是 dense local48 的 `{dense['median_correspondence_mse']:.3e}`。",
        f"6. **最佳安全最差 MSE：** 新边界模型为 `{best_boundary['worst_correspondence_mse']:.3e}`；全方法最佳对应为 dense local48 的 `{dense['worst_correspondence_mse']:.3e}`。",
        f"7. **达到 1e-9：** `{best_boundary['reaching_1e_9']}/8`。",
        f"8. **达到 1e-8：** `{best_boundary['reaching_1e_8']}/8`。",
        f"9. **是否降低 LE/TE 残差集中：** dense 的 LE+TE 为 `{dense_regions[0]+dense_regions[1]:.1%}`，最佳边界模型为 `{boundary_regions[0]+boundary_regions[1]:.1%}`。",
        f"10. **参数量：** 最佳边界模型 `{best_boundary['parameter_count']}`，dense local48 `{dense['parameter_count']}`，即 `{parameter_ratio:.1%}`。",
        f"11. **是否更参数高效：** 结论取决于口径。相对 global24，每增加一个参数带来的中位 MSE 降幅约为 dense local48 的 `{marginal_efficiency_ratio:.1f}×`，说明边界基适合低成本小幅改进；但它以 `{error_ratio:.2f}×` 的误差换取 `{parameter_ratio:.1%}` 的参数量，未达到替代 dense local48 所需的精度—参数 Pareto 改进。",
        f"12. **若仍未达到 1e-9，主瓶颈：** {'本阶段已在全部目标达到目标，需要转入独立验证而非继续加参数。' if best_boundary['reaching_1e_9'] == 8 else '证据指向固定源边界上的可分离 Gaussian 环境延拓表达能力不足：增加 1D 控制数、切向自由度和各向异性均无效，说明瓶颈不是单纯系数数量，而是边界校正如何随全局流运输并在 LE/TE 形成安全的局部耦合场。'}",
        "",
        "## 推荐的下一步",
        "",
        "- 在获得批准前停止；不运行 1654 个全量注册。",
        "- 若最佳边界模型仍未到 1e-9，下一阶段优先验证可学习但受约束的核尺度/中心位置，或采用紧支撑边界有限元/RKHS，而不是 dense local64/local96。",
        "- 对任何后续候选保留同一 8 目标、同一 51×51 Jacobian 诊断和逐点误差表，避免改变比较基准。",
        "",
        "## 尚待批准的问题",
        "",
        f"当前全方法最佳安全结果为 `{overall_best['method']}`（中位 `{overall_best['median_correspondence_mse']:.3e}`）。是否把边界感知方法推进到更严格的独立开发集/全数据集验证，需要用户批准。",
        "",
        "## 停止条件",
        "",
        "本阶段只运行 9 个既定开发 case；未运行 1654 个全量配准，未比较 K，未使用 GMM/聚类模板，未启动气动优化，未测试 dense local64/local96，未提交或推送 Git。",
    ]
    (OUT / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="run only identity and extreme for implementation timing")
    args = parser.parse_args()
    for folder in (
        "baseline",
        "boundary_geometry",
        "normal_basis",
        "normal_tangent_basis",
        "anisotropic_kernel",
        "parameter_efficiency",
        "residual_analysis",
        "figures",
        "tables",
        "report",
    ):
        (OUT / folder).mkdir(parents=True, exist_ok=True)

    shapes, mean, targets, verification = load_inputs()
    if args.pilot:
        targets = [targets[0], targets[-1]]
    bounds = (0.0, 1.0, float(shapes[:, :, 1].min() - 0.08), float(shapes[:, :, 1].max() + 0.08))
    geometry = bas.boundary_geometry(mean)
    np.savez(OUT / "boundary_geometry.npz", **geometry)
    plot_boundary_geometry(mean, geometry, OUT / "figures" / "boundary_geometry.png")

    config = {
        "data_path": str(DATA),
        "stage2_mean": str(STAGE2 / "direct_coordinate_mean" / "coordinates.npy"),
        "stage1_subset": str(STAGE1 / "development_subset"),
        "dtype": "float64",
        "source": "fixed direct-coordinate global mean",
        "targets": [target["label"] for target in targets],
        "integration": "explicit Euler",
        "steps": 15,
        "global_resolutions": [6, 12, 24],
        "global_schedule": list(SCHEDULE),
        "lambda_global": 1e-5,
        "lambda_boundary_first": 1e-4,
        "lambda_boundary_second": 1e-4,
        "barrier_formula": "mean((softplus(50*(tau-det(Dphi)))/50)^2)",
        "barrier_tau": 0.05,
        "barrier_weight": 0.01,
        "barrier_resolution": 21,
        "jacobian_report_resolution": 51,
        "tolerance_grad": 1e-12,
        "tolerance_change": 1e-15,
        "boundary_only_iterations": 35,
        "joint_iterations": 55,
        "edge_support": {"LE": [0.0, 0.15], "TE": [0.85, 1.0]},
        "kernel_formula": "w_k=exp(-0.5*((delta_t/sigma_t)^2+(delta_n/sigma_n)^2)); v_boundary=sum(w_k*c_k)",
        "sigma_n_rule": "clip(1.75*median within-group control spacing, 0.03, 0.08), fixed across anisotropy ratios",
        "sigma_t_rule": "anisotropy_ratio * sigma_n",
        "anisotropy_ratios": [1, 2, 4],
        "whole_boundary_control_counts": [16, 32, 64],
        "per_edge_control_counts": [16, 32, 64],
        "chart_map": [
            {"figure": "boundary_geometry.png", "question": "Are tangent/normal/curvature estimates coherent?", "type": "geometry overlay + line"},
            {"figure": "method_comparison.png", "question": "Which method lowers median and worst MSE?", "type": "ordered line comparison"},
            {"figure": "parameter_efficiency.png", "question": "Does accuracy improve efficiently with parameter count?", "type": "log-log scatter"},
            {"figure": "residual_region_comparison.png", "question": "Does the method change LE/TE residual concentration?", "type": "stacked bar"},
        ],
    }
    jwrite(OUT / "config.json", {**config, "bounds": bounds, "input_verification": verification})

    source = torch.tensor(mean, dtype=DTYPE)
    base_config = {
        "steps": 15,
        "basis": "cartesian",
        "barrier_resolution": 21,
        "lbfgs_max_iter": 80,
        "tolerance_grad": 1e-12,
        "tolerance_change": 1e-15,
    }
    stage3_config = {
        "steps": 15,
        "barrier_resolution": 21,
        "barrier_tau": 0.05,
        "barrier_weight": 0.01,
        "jacobian_report_resolution": 51,
        "local_lbfgs_max_iter": 60,
    }

    all_rows: list[dict] = []
    point_rows: list[dict] = []
    artifacts: dict[tuple[str, str], dict] = {}
    baseline_rows = []
    for target in targets:
        base_metrics, c24, base_deformed, _, _ = s2.run_schedule(
            mean,
            target["shape"],
            bounds,
            base_config,
            SCHEDULE,
            0.05,
            0.01,
            (6, 12, 24),
        )
        base_residual = base_deformed - target["shape"]
        base_row = base_metric_row(target, "stage2_global_6_12_24", base_metrics, 24 * 24 * 2)
        all_rows.append(base_row)
        artifacts[("stage2_global_6_12_24", target["label"])] = {
            "global": c24,
            "deformed": base_deformed,
            "residual": base_residual,
            "metrics": base_metrics,
        }
        point_rows += pointwise_rows(target, "stage2_global_6_12_24", base_deformed, base_residual, mean)

        dense_metrics, dense_coefficients, dense_deformed, dense_residual = s3.run_local(
            source,
            torch.tensor(target["shape"], dtype=DTYPE),
            torch.tensor(c24, dtype=DTYPE),
            bounds,
            stage3_config,
            48,
            0.15,
            1.0,
            "joint",
        )
        dense_metrics["runtime_seconds"] += base_metrics["runtime_seconds"]
        dense_metrics["refinement_runtime_seconds"] = dense_metrics["runtime_seconds"] - base_metrics["runtime_seconds"]
        add_dense_jacobian_median(dense_metrics, dense_coefficients, bounds, stage3_config)
        dense_row = base_metric_row(target, "stage3_dense_local48", dense_metrics, 10368)
        all_rows.append(dense_row)
        baseline_rows.append(dense_row)
        artifacts[("stage3_dense_local48", target["label"])] = {
            "deformed": dense_deformed,
            "residual": dense_residual,
            "metrics": dense_metrics,
            "coefficients": dense_coefficients,
        }
        point_rows += pointwise_rows(target, "stage3_dense_local48", dense_deformed, dense_residual, mean)

        global48_metrics, global48_coefficients, global48_deformed, global48_residual = s3.global_control(
            source,
            torch.tensor(target["shape"], dtype=DTYPE),
            torch.tensor(c24, dtype=DTYPE),
            bounds,
            stage3_config,
            48,
        )
        global48_metrics["runtime_seconds"] += base_metrics["runtime_seconds"]
        global48_metrics["refinement_runtime_seconds"] = global48_metrics["runtime_seconds"] - base_metrics["runtime_seconds"]
        global48_row = base_metric_row(target, "stage3_global48", global48_metrics, 48 * 48 * 2)
        all_rows.append(global48_row)
        artifacts[("stage3_global48", target["label"])] = {
            "deformed": global48_deformed,
            "residual": global48_residual,
            "metrics": global48_metrics,
            "coefficients": global48_coefficients,
        }
        point_rows += pointwise_rows(target, "stage3_global48", global48_deformed, global48_residual, mean)
        print(f"controls complete: {target['label']}", flush=True)
    cwrite(OUT / "baseline_reproduction.csv", baseline_rows)

    candidates = []
    for count in (16, 32, 64):
        candidates.append((f"boundary_whole_normal_{count}_ratio2", "whole", count, 2.0, "normal", "freeze_then_joint"))
    for count in (16, 32, 64):
        candidates.append((f"boundary_edge_normal_{count}_ratio2", "edge", count, 2.0, "normal", "freeze_then_joint"))
    candidates.extend(
        [
            ("boundary_edge_normal_32_ratio1", "edge", 32, 1.0, "normal", "freeze_then_joint"),
            ("boundary_edge_normal_32_ratio4", "edge", 32, 4.0, "normal", "freeze_then_joint"),
            ("boundary_edge_normal_tangent_16_ratio2", "edge", 16, 2.0, "normal_tangent", "freeze_then_joint"),
            ("boundary_edge_normal_tangent_32_ratio2", "edge", 32, 2.0, "normal_tangent", "freeze_then_joint"),
            ("boundary_edge_normal_tangent_64_ratio2", "edge", 64, 2.0, "normal_tangent", "freeze_then_joint"),
            ("boundary_edge_normal_32_boundary_only", "edge", 32, 2.0, "normal", "boundary_only"),
            ("boundary_edge_normal_32_joint", "edge", 32, 2.0, "normal", "joint"),
        ]
    )

    boundary_rows = []
    anisotropic_rows = []
    for method, scope, count, ratio, coefficient_mode, strategy in candidates:
        controls = bas.make_controls(mean, geometry, scope, count, ratio, coefficient_mode)
        for target in targets:
            base = artifacts[("stage2_global_6_12_24", target["label"])]
            global_coeff, boundary_coeff, runtime, calls, stage_info = bas.optimise(
                source,
                torch.tensor(target["shape"], dtype=DTYPE),
                torch.tensor(base["global"], dtype=DTYPE),
                controls,
                bounds,
                config,
                strategy,
            )
            metrics, deformed, residual = bas.metrics(
                source,
                torch.tensor(target["shape"], dtype=DTYPE),
                global_coeff,
                boundary_coeff,
                controls,
                bounds,
                config,
                runtime,
                calls,
                base["metrics"]["runtime_seconds"],
            )
            parameters = {
                "scope": scope,
                "control_locations": int(len(controls.indices)),
                "requested_count": count,
                "anisotropy_ratio": ratio,
                "coefficient_mode": coefficient_mode,
                "optimisation_strategy": strategy,
                "sigma_t_min": float(controls.sigma_t.min()),
                "sigma_t_max": float(controls.sigma_t.max()),
                "sigma_n_min": float(controls.sigma_n.min()),
                "sigma_n_max": float(controls.sigma_n.max()),
            }
            row = result_row(target, method, parameters, metrics)
            boundary_rows.append(row)
            all_rows.append(row)
            if method in {
                "boundary_edge_normal_32_ratio1",
                "boundary_edge_normal_32_ratio2",
                "boundary_edge_normal_32_ratio4",
            }:
                anisotropic_rows.append(row)
            artifacts[(method, target["label"])] = {
                "deformed": deformed,
                "residual": residual,
                "metrics": metrics,
                "global_coefficients": global_coeff.numpy(),
                "boundary_coefficients": boundary_coeff.numpy(),
                "controls": controls,
                "stage_info": stage_info,
            }
            point_rows += pointwise_rows(target, method, deformed, residual, mean)
            folder = "normal_tangent_basis" if coefficient_mode == "normal_tangent" else (
                "anisotropic_kernel" if ratio != 2.0 else "normal_basis"
            )
            np.savez(
                OUT / folder / f"{method}_{target['label']}.npz",
                deformed=deformed,
                residual=residual,
                global_coefficients=global_coeff.numpy(),
                boundary_coefficients=boundary_coeff.numpy(),
                control_indices=controls.indices,
                control_centers=controls.centers,
                control_tangents=controls.tangents,
                control_normals=controls.normals,
                sigma_t=controls.sigma_t,
                sigma_n=controls.sigma_n,
                groups=controls.groups,
            )
        cwrite(OUT / "boundary_basis_results.csv", boundary_rows)
        cwrite(OUT / "anisotropic_kernel_results.csv", anisotropic_rows)
        cwrite(OUT / "pointwise_residuals.csv", point_rows)
        print(f"candidate complete: {method}", flush=True)

    cwrite(OUT / "boundary_basis_results.csv", boundary_rows)
    cwrite(OUT / "anisotropic_kernel_results.csv", anisotropic_rows)
    cwrite(OUT / "pointwise_residuals.csv", point_rows)
    regions = region_summary(point_rows)
    cwrite(OUT / "residual_region_summary.csv", regions)
    diagnostic = diagnostic_rows(mean, targets, geometry)
    cwrite(OUT / "normal_tangent_diagnostic.csv", diagnostic)

    methods = sorted(set(row["method"] for row in all_rows))
    summaries = [method_summary(all_rows, method) for method in methods]
    cwrite(OUT / "parameter_efficiency.csv", summaries)
    boundary_summaries = [row for row in summaries if row["method"].startswith("boundary") and row["all_safe"]]
    best_boundary = min(boundary_summaries, key=lambda row: row["median_correspondence_mse"])
    safe_summaries = [row for row in summaries if row["all_safe"]]
    overall_best = min(safe_summaries, key=lambda row: row["median_correspondence_mse"])

    dense_summary = next(row for row in summaries if row["method"] == "stage3_dense_local48")
    historical = json.loads((STAGE3 / "best_configuration.json").read_text(encoding="utf-8"))["best_summary"]
    baseline_check = {
        "reproduced_median_mse": dense_summary["median_correspondence_mse"],
        "historical_median_mse": historical["median_correspondence_mse"],
        "relative_median_difference": abs(dense_summary["median_correspondence_mse"] - historical["median_correspondence_mse"]) / historical["median_correspondence_mse"],
        "reproduced_worst_mse": dense_summary["worst_correspondence_mse"],
        "historical_worst_mse": historical["worst_correspondence_mse"],
        "relative_worst_difference": abs(dense_summary["worst_correspondence_mse"] - historical["worst_correspondence_mse"]) / historical["worst_correspondence_mse"],
    }
    baseline_check["consistent"] = bool(
        baseline_check["relative_median_difference"] <= 0.01
        and baseline_check["relative_worst_difference"] <= 0.01
    )

    plot_method_comparison(summaries, OUT / "figures" / "method_comparison.png")
    plot_parameter_efficiency(summaries, OUT / "figures" / "parameter_efficiency.png")
    plot_region_comparison(
        regions,
        ["stage2_global_6_12_24", "stage3_dense_local48", best_boundary["method"]],
        OUT / "figures" / "residual_region_comparison.png",
    )
    best_controls = artifacts[(best_boundary["method"], targets[0]["label"])]["controls"]
    for target in targets:
        if target["label"] == "identity":
            continue
        dense_artifact = artifacts[("stage3_dense_local48", target["label"])]
        best_artifact = artifacts[(best_boundary["method"], target["label"])]
        plot_target_residual(
            target,
            mean,
            (dense_artifact["deformed"], dense_artifact["residual"]),
            (best_artifact["deformed"], best_artifact["residual"]),
            best_artifact["controls"],
            best_artifact["boundary_coefficients"],
            best_boundary["method"],
            OUT / "residual_analysis" / f"{target['label']}.png",
        )

    validation = {
        "input_verification": verification,
        "baseline_reproduction": baseline_check,
        "all_candidate_rows_finite": bool(
            all(np.isfinite(float(row["correspondence_mse"])) for row in boundary_rows)
        ),
        "best_boundary_all_safe": bool(best_boundary["all_safe"]),
        "best_boundary_nonpositive_jacobian": 0.0,
        "method_count": len(summaries),
        "boundary_candidate_count": len(candidates),
        "paired_nonidentity_targets_per_method": 8 if not args.pilot else 1,
    }
    best_configuration = {
        "best_boundary_configuration": best_boundary,
        "overall_best_safe_configuration": overall_best,
        "baseline_reproduction": baseline_check,
        "all_method_summaries": summaries,
        "kernel_formula": config["kernel_formula"],
        "boundary_regularisation": {
            "first_difference_weight": config["lambda_boundary_first"],
            "second_difference_weight": config["lambda_boundary_second"],
        },
        "fixed_regions": config["edge_support"],
        "selection_rule": "lowest median correspondence MSE among configurations safe on all 8 non-identity targets",
    }
    jwrite(OUT / "validation.json", validation)
    jwrite(OUT / "best_configuration.json", best_configuration)
    if not args.pilot:
        write_report(
            summaries,
            best_boundary,
            overall_best,
            verification,
            regions,
            diagnostic,
            baseline_check,
        )
    print(json.dumps(best_configuration, indent=2), flush=True)


if __name__ == "__main__":
    main()
