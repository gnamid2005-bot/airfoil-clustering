"""Independent consistency checks for the saved Stage 4 proof-of-concept."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "airfoil_registration_improvement_stage4"


def read_csv(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(a: float, b: float, *, rtol: float = 2e-10, atol: float = 1e-14) -> bool:
    return bool(np.isclose(a, b, rtol=rtol, atol=atol))


def main() -> None:
    required = [
        "boundary_geometry.npz",
        "boundary_basis_results.csv",
        "anisotropic_kernel_results.csv",
        "parameter_efficiency.csv",
        "residual_region_summary.csv",
        "pointwise_residuals.csv",
        "normal_tangent_diagnostic.csv",
        "best_configuration.json",
        "final_report.md",
        "config.json",
    ]
    checks: dict[str, bool] = {"required_outputs_present": all((OUT / name).is_file() for name in required)}

    config = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
    best = json.loads((OUT / "best_configuration.json").read_text(encoding="utf-8"))
    data = np.load(ROOT / config["data_path"])["shapes"]
    saved_mean = np.load(ROOT / config["stage2_mean"])
    checks["dataset_shape_is_1654_200_2"] = data.shape == (1654, 200, 2)
    checks["direct_coordinate_mean_exact"] = bool(np.max(np.abs(data.mean(axis=0) - saved_mean)) == 0.0)
    checks["fixed_x_grid"] = bool(np.max(np.ptp(data[:, :, 0], axis=0)) == 0.0)
    checks["target_indices_unchanged"] = config["input_verification"]["selected_indices"] == [
        -1,
        382,
        104,
        959,
        362,
        1489,
        377,
        1224,
        518,
    ]

    boundary = read_csv("boundary_basis_results.csv")
    anisotropic = read_csv("anisotropic_kernel_results.csv")
    summary = read_csv("parameter_efficiency.csv")
    regions = read_csv("residual_region_summary.csv")
    pointwise = read_csv("pointwise_residuals.csv")
    diagnostic = read_csv("normal_tangent_diagnostic.csv")
    baseline = read_csv("baseline_reproduction.csv")

    boundary_methods = sorted({row["method"] for row in boundary})
    all_methods = sorted({row["method"] for row in pointwise})
    checks["thirteen_boundary_candidates_nine_cases"] = len(boundary_methods) == 13 and len(boundary) == 13 * 9
    checks["sixteen_total_methods_nine_cases_200_points"] = len(all_methods) == 16 and len(pointwise) == 16 * 9 * 200
    checks["anisotropy_is_controlled_1_2_4"] = sorted({float(row["anisotropy_ratio"]) for row in anisotropic}) == [1.0, 2.0, 4.0]
    checks["all_boundary_candidates_safe"] = all(
        row["safe"] == "True"
        and row["valid_geometry"] == "True"
        and float(row["minimum_jacobian"]) > 0.0
        and float(row["nonpositive_jacobian_proportion"]) == 0.0
        for row in boundary
    )
    checks["all_saved_numeric_results_finite"] = all(
        np.isfinite(float(row[key]))
        for row in boundary
        for key in (
            "correspondence_mse",
            "total_objective",
            "minimum_jacobian",
            "jacobian_p01",
            "jacobian_p05",
            "jacobian_median",
        )
    )

    # Recompute each target MSE from the 200 pointwise rows, then recompute the
    # non-identity median/worst/best and threshold counts in the method table.
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in pointwise:
        grouped[(row["method"], row["target_label"])].append(float(row["squared_error"]))
    target_mse = {key: float(np.mean(values)) for key, values in grouped.items()}
    summary_ok = True
    for row in summary:
        values = [target_mse[(row["method"], label)] for label in sorted({key[1] for key in grouped if key[0] == row["method"] and key[1] != "identity"})]
        summary_ok &= len(values) == 8
        summary_ok &= close(float(row["median_correspondence_mse"]), float(np.median(values)))
        summary_ok &= close(float(row["worst_correspondence_mse"]), max(values))
        summary_ok &= close(float(row["best_correspondence_mse"]), min(values))
        summary_ok &= int(row["reaching_1e_9"]) == sum(value <= 1e-9 for value in values)
        summary_ok &= int(row["reaching_1e_8"]) == sum(value <= 1e-8 for value in values)
        summary_ok &= int(row["reaching_1e_7"]) == sum(value <= 1e-7 for value in values)
    checks["method_summaries_recompute_from_pointwise_rows"] = bool(summary_ok)

    # Region totals must partition all 200 squared errors exactly, and the
    # three reported fractions must sum to one for every nonzero case.
    pointwise_sum = {key: float(np.sum(values)) for key, values in grouped.items()}
    region_ok = len(regions) == 16 * 9
    for row in regions:
        key = (row["method"], row["target_label"])
        parts = sum(float(row[name]) for name in ("leading_edge_absolute", "trailing_edge_absolute", "mid_chord_absolute"))
        region_ok &= close(parts, float(row["total_squared_error"]))
        region_ok &= close(parts, pointwise_sum[key])
        if parts > 0.0:
            fractions = sum(float(row[name]) for name in ("leading_edge_fraction", "trailing_edge_fraction", "mid_chord_fraction"))
            region_ok &= close(fractions, 1.0)
    checks["residual_regions_partition_pointwise_error"] = bool(region_ok)

    diagnostic_ok = len(diagnostic) == 9 * 4
    for row in diagnostic:
        total = float(row["total_displacement_energy"])
        normal = float(row["normal_displacement_energy"])
        tangent = float(row["tangent_displacement_energy"])
        diagnostic_ok &= close(total, normal + tangent, rtol=1e-9)
        if total > 0.0:
            diagnostic_ok &= close(float(row["normal_energy_fraction"]) + float(row["tangent_energy_fraction"]), 1.0)
    checks["normal_tangent_energy_decomposition_closes"] = bool(diagnostic_ok)

    historical = best["baseline_reproduction"]
    baseline_values = [float(row["correspondence_mse"]) for row in baseline if row["target_label"] != "identity"]
    checks["stage3_baseline_reproduced"] = (
        bool(historical["consistent"])
        and close(float(np.median(baseline_values)), 1.1382817569856905e-8)
        and close(max(baseline_values), 3.268340443462815e-8)
    )
    checks["identity_sanity_all_methods"] = all(target_mse[(method, "identity")] <= 1e-15 for method in all_methods)

    report = (OUT / "final_report.md").read_text(encoding="utf-8")
    checks["report_answers_all_twelve_questions"] = all(f"{index}. **" in report for index in range(1, 13))
    checks["report_records_stop_conditions"] = all(
        phrase in report for phrase in ("未运行 1654", "未比较 K", "未使用 GMM", "未启动气动优化", "未提交或推送 Git")
    )

    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "counts": {
            "boundary_candidates": len(boundary_methods),
            "methods": len(all_methods),
            "development_cases": 9,
            "nonidentity_targets": 8,
            "pointwise_rows": len(pointwise),
        },
    }
    (OUT / "validation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
