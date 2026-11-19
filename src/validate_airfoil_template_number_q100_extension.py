#!/usr/bin/env python3
"""Independent validation for the Stage-5 q=100 PCA/GMM extension."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "airfoil_template_number_stage5_screening"
Q40 = ROOT / "outputs" / "airfoil_template_number_stage5_q40_extension"
OUT = ROOT / "outputs" / "airfoil_template_number_stage5_q100_extension"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def truth(value: object) -> bool:
    return value is True or str(value).lower() == "true"


def finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def main() -> None:
    checks: list[dict[str, object]] = []

    def check(name: str, condition: bool, detail: object = "") -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    required = [
        "config.json", "input_verification.json", "pca_q41_q100_summary.csv", "pca_q1_q100.csv",
        "gmm_q41_q100_cells.csv", "gmm_q1_q100_combined.csv", "preferred_k_q1_q100.csv",
        "preferred_k_sequences_q1_q100.json", "preferred_k_frequency_by_q_range.csv",
        "high_dimensional_parameter_counts.csv", "preferred_model_component_diagnostics.csv",
        "targeted_stability_ari.csv", "targeted_stability_cells.csv", "model_selection_gap_by_q.csv",
        "report/q100_extension_report.md", "run_summary.json",
        "scripts/run_airfoil_template_number_q100_extension.py",
        "figures/preferred_k_q1_q100.png", "figures/preferred_k_q1_q100.pdf",
        "figures/preferred_k_bic_q1_q100.png", "figures/preferred_k_icl_q1_q100.png",
        "figures/preferred_k_held_out_ll_q1_q100.png",
        "figures/pca_cumulative_explained_variance_q1_q100.png",
        "figures/pca_heldout_reconstruction_mse_q1_q100.png", "figures/gmm_parameter_count_vs_q.png",
        "figures/relative_bic_q1_q100_heatmap.png", "figures/relative_icl_q1_q100_heatmap.png",
        "figures/relative_heldout_ll_q1_q100_heatmap.png", "figures/best_vs_second_best_gap_q1_q100.png",
        "figures/targeted_ari_stability_vs_q.png", "figures/preferred_model_minimum_component_size_vs_q.png",
    ]
    missing = [name for name in required if not (OUT / name).exists()]
    check("required_outputs_present", not missing, missing)

    matrix = np.load(SOURCE / "velocity_field_matrix.npy", mmap_mode="r")
    population = np.load(SOURCE / "velocity_field_population_indices.npy")
    split = np.load(SOURCE / "split" / "train_heldout_indices.npz")
    train = split["train_matrix_positions"]
    heldout = split["heldout_matrix_positions"]
    check("safe_matrix_exact", matrix.shape == (1651, 10368) and matrix.dtype == np.float64, [matrix.shape, str(matrix.dtype)])
    check("safe_population_exact", population.shape == (1651,) and len(np.unique(population)) == 1651)
    check("fixed_split_counts", len(train) == 1321 and len(heldout) == 330)
    check("fixed_split_partition", np.intersect1d(train, heldout).size == 0 and np.union1d(train, heldout).size == 1651)

    config = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
    check("pca_config_exact", config["pca"] == {"fit_once": True, "iterated_power": 7, "n_components": 100,
                                                  "q_score_rule": "Z[:, :q]", "random_state": 20260813,
                                                  "svd_solver": "randomized"}, config["pca"])
    grid_cfg = config["new_gmm_grid"]
    check("gmm_config_exact", grid_cfg["q"] == [41, 100] and grid_cfg["K"] == [1, 20] and
          grid_cfg["covariance_type"] == "full" and grid_cfg["n_init"] == 5 and
          grid_cfg["random_state"] == 20260813 and grid_cfg["reg_covar"] == 1e-7 and
          grid_cfg["max_iter"] == 500 and grid_cfg["tol"] == 1e-5 and grid_cfg["init_params"] == "kmeans")
    check("no_scaling", config["scaling"] == "none")
    check("registration_read_only_policy", "no registration" in config["registration_policy"])

    with np.load(OUT / "pca" / "pca_q100.npz") as pca:
        check("pca_shapes", pca["components"].shape == (100, 10368) and
              pca["train_scores"].shape == (1321, 100) and pca["heldout_scores"].shape == (330, 100))
        check("pca_split_exact", np.array_equal(pca["train_matrix_positions"], train) and
              np.array_equal(pca["heldout_matrix_positions"], heldout))
        recomputed = (matrix[heldout] - pca["mean"]) @ pca["components"].T
        max_score_difference = float(np.max(np.abs(recomputed - pca["heldout_scores"])))
        check("pca_fitted_training_only_and_holdout_transform_consistent",
              np.allclose(recomputed, pca["heldout_scores"], rtol=1e-10, atol=1e-11), max_score_difference)

    pca_new = read_csv(OUT / "pca_q41_q100_summary.csv")
    pca_all = read_csv(OUT / "pca_q1_q100.csv")
    check("pca_new_dimensions_exact_60", len(pca_new) == 60 and [int(row["q"]) for row in pca_new] == list(range(41, 101)))
    check("pca_all_dimensions_exact_100", len(pca_all) == 100 and [int(row["q"]) for row in pca_all] == list(range(1, 101)))
    check("pca_cumulative_monotone", np.all(np.diff([float(row["cumulative_explained_variance"]) for row in pca_all]) >= -1e-14))
    check("pca_reconstruction_monotone", np.all(np.diff([float(row["heldout_reconstruction_mse"]) for row in pca_all]) <= 1e-14))
    check("pca_new_metrics_finite", all(all(finite(row[key]) for key in (
        "marginal_explained_variance_ratio", "cumulative_explained_variance", "heldout_reconstruction_mse",
        "heldout_reconstruction_rmse", "maximum_heldout_absolute_coefficient_error",
        "maximum_heldout_sample_reconstruction_rmse")) for row in pca_new))
    old_pca = read_csv(Q40 / "pca_q1_q40.csv")
    exact_old_pca = all(pca_all[index][key] == old_pca[index][source_key]
                        for index in range(40)
                        for key, source_key in (("marginal_explained_variance_ratio", "explained_variance_ratio"),
                                                ("cumulative_explained_variance", "cumulative_explained_variance"),
                                                ("train_reconstruction_mse", "train_reconstruction_mse"),
                                                ("heldout_reconstruction_mse", "heldout_reconstruction_mse")))
    check("q1_q40_pca_saved_values_reused_exactly", exact_old_pca)

    new = read_csv(OUT / "gmm_q41_q100_cells.csv")
    intended = {(q, k) for q in range(41, 101) for k in range(1, 21)}
    actual = {(int(row["q"]), int(row["K"])) for row in new}
    check("new_grid_exact_1200", len(new) == 1200 and actual == intended, len(new))
    check("new_grid_configuration", all(row["covariance_type"] == "full" and int(row["n_init"]) == 5 and
          int(row["seed"]) == 20260813 and float(row["reg_covar"]) == 1e-7 for row in new))
    valid = [row for row in new if truth(row["valid_for_selection"])]
    failed = [row for row in new if row["status"].startswith("failed")]
    nonconv = [row for row in new if row["status"] == "non_converged"]
    finite_keys = ("training_log_likelihood_mean", "heldout_log_likelihood_mean", "BIC", "ICL",
                   "posterior_entropy_total", "minimum_component_weight", "parameter_count", "runtime_seconds")
    check("valid_cells_finite", all(all(finite(row[key]) for key in finite_keys) for row in valid), len(valid))
    check("icl_formula_exact", all(np.isclose(float(row["ICL"]), float(row["BIC"]) +
          2 * float(row["posterior_entropy_total"]), rtol=1e-10, atol=1e-7) for row in valid))
    check("failed_cells_visible_and_not_selectable", all(not truth(row["valid_for_selection"]) and
          row["warning_or_failure_message"] for row in failed), len(failed))
    check("nonconverged_not_selectable", all(not truth(row["valid_for_selection"]) for row in nonconv), len(nonconv))
    parameter_ok = all(int(row["parameter_count"]) == int(row["K"]) * int(row["q"]) * (int(row["q"]) + 1) // 2 +
                       int(row["K"]) * int(row["q"]) + int(row["K"]) - 1 for row in new)
    check("gmm_parameter_formula", parameter_ok)

    combined = read_csv(OUT / "gmm_q1_q100_combined.csv")
    old_combined = read_csv(Q40 / "gmm_q1_q40_combined.csv")
    check("combined_grid_exact_2000", len(combined) == 2000 and {(int(row["q"]), int(row["K"])) for row in combined} ==
          {(q, k) for q in range(1, 101) for k in range(1, 21)})
    old_fields = list(old_combined[0].keys())
    exact_old_gmm = all(all(combined[i].get(key, "") == old_combined[i].get(key, "") for key in old_fields) for i in range(800))
    check("q1_q40_gmm_rows_reused_exactly", exact_old_gmm)

    preferred = read_csv(OUT / "preferred_k_q1_q100.csv")
    check("preferred_rows_exact_100", len(preferred) == 100 and [int(row["q"]) for row in preferred] == list(range(1, 101)))
    recomputed_ok = True
    for q in range(1, 101):
        group = [row for row in combined if int(row["q"]) == q and truth(row["valid_for_selection"])]
        row = preferred[q - 1]
        recomputed_ok &= int(row["K_BIC"]) == int(min(group, key=lambda x: float(x["BIC"]))["K"])
        recomputed_ok &= int(row["K_ICL"]) == int(min(group, key=lambda x: float(x["ICL"]))["K"])
        recomputed_ok &= int(row["K_heldout_LL"]) == int(max(group, key=lambda x: float(x["heldout_log_likelihood_mean"]))["K"])
    check("preferred_direction_and_validity", recomputed_ok)
    old_preferred = read_csv(Q40 / "preferred_k_q1_q40.csv")
    pref_keys = ("K_BIC", "K_ICL", "K_heldout_LL", "BIC_small_component_flag",
                 "ICL_small_component_flag", "heldout_LL_small_component_flag")
    check("q1_q40_preferred_reproduced", all(preferred[i][key] == old_preferred[i][key] for i in range(40) for key in pref_keys))

    params = read_csv(OUT / "high_dimensional_parameter_counts.csv")
    check("parameter_table_exact_2000", len(params) == 2000 and {(int(row["q"]), int(row["K"])) for row in params} ==
          {(q, k) for q in range(1, 101) for k in range(1, 21)})
    gaps = read_csv(OUT / "model_selection_gap_by_q.csv")
    check("gap_table_exact_300", len(gaps) == 300 and all(float(row["raw_gap"]) >= -1e-10 for row in gaps))
    stability = read_csv(OUT / "targeted_stability_ari.csv")
    check("targeted_stability_exact_20", len(stability) == 20 and {(int(row["q"]), int(row["K"])) for row in stability} ==
          {(q, k) for q in (20, 40, 60, 80, 100) for k in (2, 6, 7, 8)})
    check("targeted_stability_complete", all(int(row["successful_seeds"]) == 5 and int(row["pair_count"]) == 10 and
          all(finite(row[key]) for key in ("mean_pairwise_ARI", "minimum_pairwise_ARI", "maximum_pairwise_ARI", "std_pairwise_ARI")) for row in stability))

    verification = json.loads((OUT / "input_verification.json").read_text(encoding="utf-8"))
    hashes_ok = True
    changed = []
    for info in verification["source_fingerprint_before"]["files"].values():
        path = ROOT / info["path"]
        if digest(path) != info["sha256"] or path.stat().st_size != info["size_bytes"]:
            hashes_ok = False
            changed.append(str(path))
    current_registration_files = sorted((SOURCE / "registrations").glob("*.npz"))
    before = verification["source_fingerprint_before"]
    registration_unchanged = len(current_registration_files) == before["registration_artifact_count"] and \
        sum(path.stat().st_size for path in current_registration_files) == before["registration_artifact_total_bytes"] and \
        max(path.stat().st_mtime_ns for path in current_registration_files) == before["registration_artifact_latest_mtime_ns"]
    check("source_hashes_unchanged", hashes_ok, changed)
    check("registration_artifacts_unchanged", registration_unchanged)
    check("no_template_or_registration_outputs", not any(path for path in OUT.rglob("*") if
          "cluster_template" in path.name or "template_registration" in path.name))
    check("q_and_K_bounds", max(int(row["q"]) for row in combined) == 100 and max(int(row["K"]) for row in combined) == 20)

    figures = list((OUT / "figures").glob("*.png"))
    check("figures_nonempty", len(figures) >= 13 and all(path.stat().st_size > 10_000 for path in figures),
          [path.name for path in figures])
    report = (OUT / "report" / "q100_extension_report.md").read_text(encoding="utf-8")
    required_tokens = ("完整首选 K 序列", "q(q+1)/2", "Gaussian mixture component count", "未重新运行配准", "到此停止")
    check("report_contains_required_analysis", all(token in report for token in required_tokens))
    check("reproducibility_script_matches", digest(OUT / "scripts" / "run_airfoil_template_number_q100_extension.py") ==
          digest(ROOT / "src" / "run_airfoil_template_number_q100_extension.py"))

    result = {
        "status": "PASS" if all(item["passed"] for item in checks) else "FAIL",
        "checks": checks,
        "counts": {"safe_samples": 1651, "training_samples": 1321, "holdout_samples": 330,
                   "new_cells": len(new), "combined_cells": len(combined), "valid_new_cells": len(valid),
                   "failed_new_cells": len(failed), "non_converged_new_cells": len(nonconv)},
    }
    (OUT / "validation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    target = OUT / "scripts" / Path(__file__).name
    target.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    summary_path = OUT / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for _ in range(2):
        summary["output_storage_bytes_final"] = int(sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file()))
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
