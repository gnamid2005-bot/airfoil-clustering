"""Independent saved-output validation for Stage 5 Phase A--K."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "airfoil_template_number_stage5_screening"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    checks: dict[str, bool] = {}
    required = [
        "config.json",
        "global_mean.npz",
        "baseline_reproduction.json",
        "registration_results.csv",
        "velocity_field_properties.csv",
        "velocity_field_matrix.npy",
        "velocity_field_population_indices.npy",
        "split/train_heldout_indices.npz",
        "pca/pca_q20.npz",
        "pca/pca_dimension_study.csv",
        "gmm/dry_run.csv",
        "gmm/primary_grid.csv",
        "gmm/preferred_k_by_q.csv",
        "gmm/shortlist_robustness.csv",
        "gmm/shortlist.json",
        "report/screening_report.md",
        "figures/bic_vs_k_all_q.png",
        "figures/icl_vs_k_all_q.png",
        "figures/bic_heatmap.png",
        "figures/icl_heatmap.png",
        "figures/heldout_ll_heatmap.png",
    ]
    checks["required_outputs_present"] = all((OUT / path).is_file() for path in required)

    config = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
    frozen = config["frozen_registration"]
    stage3 = json.loads((ROOT / "outputs/airfoil_registration_improvement_stage3/best_configuration.json").read_text(encoding="utf-8"))
    checks["registration_algorithm_frozen"] = (
        stage3["selected_method"] == "local48_wide"
        and frozen["global_resolutions"] == [6, 12, 24]
        and frozen["local_resolution"] == 48
        and frozen["local_edge_support"] == 0.15
        and frozen["local_alpha"] == 1.0
        and frozen["steps"] == 15
        and frozen["dtype"] == "float64"
        and frozen["field_vector_dimension"] == 10368
    )

    raw = np.load(ROOT / config["data_verification"]["data_path"])
    shapes = raw["shapes"]
    saved_mean = np.load(OUT / "global_mean.npz")["coordinates"]
    checks["dataset_and_mean_exact"] = (
        shapes.shape == (1654, 200, 2)
        and np.max(np.ptp(shapes[:, :, 0], axis=0)) == 0.0
        and np.array_equal(saved_mean, shapes.mean(axis=0))
    )

    registration = rows(OUT / "registration_results.csv")
    properties = rows(OUT / "velocity_field_properties.csv")
    artifacts = sorted((OUT / "registrations").glob("*.npz"))
    safe_indices = np.asarray(sorted(int(row["target_index"]) for row in registration if row["safe"] == "True"))
    unsafe_indices = np.asarray(sorted(int(row["target_index"]) for row in registration if row["safe"] != "True"))
    checks["complete_population_registration"] = len(registration) == len(properties) == len(artifacts) == 1654
    checks["success_and_safety_counts"] = sum(row["success"] == "True" for row in registration) == 1654 and len(safe_indices) == 1651
    checks["unsafe_indices_audited"] = np.array_equal(unsafe_indices, np.asarray([3, 206, 408]))
    checks["unavailable_metrics_not_fabricated"] = all(
        row["classical_lddmm_geodesic_energy"] == "UNAVAILABLE" and row["rkhs_norm"] == "UNAVAILABLE"
        for row in properties
    )

    finite_keys = [
        "correspondence_mse",
        "dirichlet_energy_common_grid",
        "velocity_magnitude_variance",
        "jacobian_minimum",
        "jacobian_mean",
        "jacobian_variance",
        "curl_rms",
        "divergence_rms",
        "bending_energy_laplacian",
        "mean_flow_path_length",
        "kinetic_energy",
        "spectral_high_frequency_fraction",
        "field_magnitude_entropy",
    ]
    checks["all_available_properties_finite"] = all(
        np.isfinite(float(row[key])) for row in properties for key in finite_keys
    )

    matrix = np.load(OUT / "velocity_field_matrix.npy", mmap_mode="r")
    population_indices = np.load(OUT / "velocity_field_population_indices.npy")
    checks["field_matrix_shape_and_indices"] = matrix.shape == (1651, 10368) and np.array_equal(population_indices, safe_indices)
    sampled_positions = [0, len(population_indices) // 2, len(population_indices) - 1]
    vector_ok = True
    for position in sampled_positions:
        index = int(population_indices[position])
        with np.load(OUT / "registrations" / f"{index:04d}.npz") as artifact:
            expected = np.concatenate(
                [
                    artifact["global_coefficients"].ravel(order="C"),
                    artifact["le_coefficients"].ravel(order="C"),
                    artifact["te_coefficients"].ravel(order="C"),
                ]
            )
            vector_ok &= np.array_equal(matrix[position], expected)
    checks["field_vectorisation_consistent"] = bool(vector_ok)

    split = np.load(OUT / "split/train_heldout_indices.npz")
    train_indices, heldout_indices, excluded = split["train_indices"], split["heldout_indices"], split["excluded_indices"]
    checks["split_is_fixed_disjoint_and_complete"] = (
        len(train_indices) == 1321
        and len(heldout_indices) == 330
        and len(np.intersect1d(train_indices, heldout_indices)) == 0
        and np.array_equal(np.sort(np.concatenate([train_indices, heldout_indices])), safe_indices)
        and np.array_equal(excluded, unsafe_indices)
    )

    pca = np.load(OUT / "pca/pca_q20.npz")
    positions = {int(index): position for position, index in enumerate(population_indices)}
    train_positions = np.asarray([positions[int(index)] for index in train_indices])
    checks["pca_fit_on_train_only"] = np.allclose(pca["mean"], np.asarray(matrix[train_positions]).mean(axis=0), rtol=1e-12, atol=1e-14)
    pca_rows = rows(OUT / "pca/pca_dimension_study.csv")
    cumulative = [float(row["cumulative_explained_variance"]) for row in pca_rows]
    checks["pca_q_1_through_20_complete"] = len(pca_rows) == 20 and [int(row["q"]) for row in pca_rows] == list(range(1, 21)) and np.all(np.diff(cumulative) >= 0)

    gmm = rows(OUT / "gmm/primary_grid.csv")
    cells = {(int(row["q"]), int(row["K"])) for row in gmm}
    checks["gmm_400_cells_complete"] = len(gmm) == len(cells) == 400 and cells == {(q, k) for q in range(1, 21) for k in range(1, 21)}
    checks["gmm_primary_specification_fixed"] = all(row["covariance_type"] == "full" and int(row["n_init"]) == 5 and int(row["seed"]) == 20260813 for row in gmm)
    checks["icl_formula_recomputes"] = all(
        np.isclose(float(row["ICL"]), float(row["BIC"]) + 2.0 * float(row["posterior_entropy_total"]), rtol=1e-12, atol=1e-9)
        for row in gmm
    )

    preferred = rows(OUT / "gmm/preferred_k_by_q.csv")
    preferred_ok = len(preferred) == 20
    for row in preferred:
        q = int(row["q"])
        group = [value for value in gmm if int(value["q"]) == q and value["converged"] == "True"]
        preferred_ok &= int(row["K_BIC"]) == int(min(group, key=lambda value: float(value["BIC"]))["K"])
        preferred_ok &= int(row["K_ICL"]) == int(min(group, key=lambda value: float(value["ICL"]))["K"])
        preferred_ok &= int(row["K_heldout_LL"]) == int(max(group, key=lambda value: float(value["heldout_log_likelihood_mean"]))["K"])
    checks["preferred_k_recomputes"] = bool(preferred_ok)

    robustness = rows(OUT / "gmm/shortlist_robustness.csv")
    robust_cells = {(int(row["q"]), int(row["K"])) for row in robustness}
    checks["robustness_covariance_seed_grid_complete"] = len(robustness) == len(robust_cells) * 4 * 5 and all(
        row["covariance_type"] in {"spherical", "diag", "tied", "full"}
        and int(row["seed"]) in range(5)
        and int(row["n_init"]) == 3
        and np.isfinite(float(row["mean_pairwise_ARI"]))
        for row in robustness
    )
    shortlist = json.loads((OUT / "gmm/shortlist.json").read_text(encoding="utf-8"))
    checks["shortlist_is_structured"] = shortlist["primary_K_values"] == [6, 7, 8] and shortlist["sensitivity_K_values"] == [13]

    report = (OUT / "report/screening_report.md").read_text(encoding="utf-8")
    checks["report_stops_before_phase_l"] = "未构造 cluster-specific templates" in report and "未执行 Phase L–Q" in report
    checks = {key: bool(value) for key, value in checks.items()}
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "counts": {
            "registrations": len(registration),
            "safe_fields_in_pca": len(population_indices),
            "pca_dimensions": len(pca_rows),
            "gmm_cells": len(gmm),
            "robustness_fits": len(robustness),
        },
    }
    (OUT / "validation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
