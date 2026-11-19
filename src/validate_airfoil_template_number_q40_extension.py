#!/usr/bin/env python3
"""Independent validation for the Stage-5 q=40 extension."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "airfoil_template_number_stage5_screening"
OUT = ROOT / "outputs" / "airfoil_template_number_stage5_q40_extension"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def truth(value: str) -> bool:
    return value.lower() == "true"


def main() -> None:
    checks: list[dict[str, object]] = []

    def check(name: str, condition: bool, detail: object = "") -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    required = [
        "config.json", "input_verification.json", "pca_q1_q40.csv", "gmm_q21_q40_cells.csv",
        "gmm_q1_q40_combined.csv", "preferred_k_q1_q40.csv", "preferred_k_sequences.json",
        "small_component_diagnostics.csv", "convergence_diagnostics.csv", "report/q40_extension_report.md",
        "figures/preferred_k_q1_q40.png", "figures/preferred_k_q1_q40.pdf",
        "figures/preferred_k_q1_q40_small_component_flags.png", "figures/pca_explained_variance_q1_q40.png",
        "figures/pca_reconstruction_mse_q1_q40.png", "validation.json",
        "scripts/run_airfoil_template_number_q40_extension.py", "scripts/validate_airfoil_template_number_q40_extension.py",
    ]
    # validation.json is written after checks, so exclude it from pre-existence gate.
    missing = [name for name in required if name != "validation.json" and not (OUT / name).exists()]
    check("required_outputs_present", not missing, missing)

    matrix = np.load(SOURCE / "velocity_field_matrix.npy", mmap_mode="r")
    split = np.load(SOURCE / "split" / "train_heldout_indices.npz")
    check("safe_matrix_exact", matrix.shape == (1651, 10368) and matrix.dtype == np.float64, [matrix.shape, str(matrix.dtype)])
    check("fixed_split_counts", len(split["train_matrix_positions"]) == 1321 and len(split["heldout_matrix_positions"]) == 330)
    check("fixed_split_partition", np.intersect1d(split["train_matrix_positions"], split["heldout_matrix_positions"]).size == 0 and np.union1d(split["train_matrix_positions"], split["heldout_matrix_positions"]).size == 1651)

    config = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
    check("common_q40_pca_config", config["pca"] == {"fit_once": True, "iterated_power": 7, "n_components": 40, "q_score_rule": "Z[:, :q]", "random_state": 20260813, "svd_solver": "randomized"}, config["pca"])
    check("no_scaling", config["scaling"] == "none")
    check("full_covariance_config", config["new_gmm_grid"]["covariance_type"] == "full" and config["new_gmm_grid"]["n_init"] == 5)

    with np.load(OUT / "pca" / "pca_q40.npz") as pca:
        check("pca_shapes", pca["components"].shape == (40, 10368) and pca["train_scores"].shape == (1321, 40) and pca["heldout_scores"].shape == (330, 40))
        check("pca_split_exact", np.array_equal(pca["train_matrix_positions"], split["train_matrix_positions"]) and np.array_equal(pca["heldout_matrix_positions"], split["heldout_matrix_positions"]))
        reconstructed_scores = (matrix[split["heldout_matrix_positions"]] - pca["mean"]) @ pca["components"].T
        check("pca_holdout_transform_consistent", np.allclose(reconstructed_scores, pca["heldout_scores"], rtol=1e-10, atol=1e-11), float(np.max(np.abs(reconstructed_scores - pca["heldout_scores"]))))

    pca_rows = read_csv(OUT / "pca_q1_q40.csv")
    check("pca_rows_exact", len(pca_rows) == 40 and [int(row["q"]) for row in pca_rows] == list(range(1, 41)))
    check("pca_monotone", np.all(np.diff([float(row["cumulative_explained_variance"]) for row in pca_rows]) >= 0))
    check("heldout_reconstruction_monotone", np.all(np.diff([float(row["heldout_reconstruction_mse"]) for row in pca_rows]) <= 1e-16))

    new = read_csv(OUT / "gmm_q21_q40_cells.csv")
    intended = {(q, k) for q in range(21, 41) for k in range(1, 21)}
    actual = {(int(row["q"]), int(row["K"])) for row in new}
    check("new_grid_exact_400", len(new) == 400 and actual == intended, len(new))
    check("new_grid_configuration", all(row["covariance_type"] == "full" and int(row["n_init"]) == 5 and int(row["seed"]) == 20260813 for row in new))
    valid = [row for row in new if truth(row["valid_for_selection"])]
    finite_keys = ("BIC", "ICL", "heldout_log_likelihood_mean", "posterior_entropy_total")
    check("valid_cells_finite", all(all(np.isfinite(float(row[key])) for key in finite_keys) for row in valid), len(valid))
    check("icl_formula", all(np.isclose(float(row["ICL"]), float(row["BIC"]) + 2 * float(row["posterior_entropy_total"]), rtol=1e-10, atol=1e-7) for row in valid))
    failed_visible = [row for row in new if row["status"].startswith("failed")]
    check("failed_cells_visible", all(not truth(row["valid_for_selection"]) and row["warning_or_failure_message"] for row in failed_visible), len(failed_visible))

    combined = read_csv(OUT / "gmm_q1_q40_combined.csv")
    check("combined_grid_exact_800", len(combined) == 800 and {(int(row["q"]), int(row["K"])) for row in combined} == {(q, k) for q in range(1, 41) for k in range(1, 21)})
    check("legacy_reused_labelled", all(row["pca_basis_source"] == "legacy_separate_q20_fit_reused" for row in combined if int(row["q"]) <= 20))
    check("extension_common_basis_labelled", all(row["pca_basis_source"] == "single_common_q40_training_fit" for row in combined if int(row["q"]) >= 21))

    preferred = read_csv(OUT / "preferred_k_q1_q40.csv")
    check("preferred_rows_exact", len(preferred) == 40)
    recomputed_ok = True
    for q in range(1, 41):
        group = [row for row in combined if int(row["q"]) == q and truth(row["valid_for_selection"])]
        row = preferred[q - 1]
        recomputed_ok &= int(row["K_BIC"]) == int(min(group, key=lambda value: float(value["BIC"]))["K"])
        recomputed_ok &= int(row["K_ICL"]) == int(min(group, key=lambda value: float(value["ICL"]))["K"])
        recomputed_ok &= int(row["K_heldout_LL"]) == int(max(group, key=lambda value: float(value["heldout_log_likelihood_mean"]))["K"])
    check("preferred_direction_and_validity", recomputed_ok)

    verification = json.loads((OUT / "input_verification.json").read_text(encoding="utf-8"))
    hashes_ok = True
    for info in verification["source_fingerprint_before"]["files"].values():
        hashes_ok &= digest(ROOT / info["path"]) == info["sha256"]
    check("source_study_unchanged_no_registration_rerun", hashes_ok)
    check("no_template_outputs", not any(OUT.glob("**/*template*registration*")))
    check("q_and_K_bounds", max(int(row["q"]) for row in combined) == 40 and max(int(row["K"]) for row in combined) == 20)

    figures = list((OUT / "figures").glob("*.png"))
    check("figures_nonempty", len(figures) >= 12 and all(path.stat().st_size > 10_000 for path in figures), [path.name for path in figures])
    report_text = (OUT / "report" / "q40_extension_report.md").read_text(encoding="utf-8")
    check("report_has_required_diagnostics", all(token in report_text for token in ("十个诊断问题", "q(q+1)/2", "K=6–8", "K=13", "未重新运行注册")))
    check("reproducibility_scripts_match", digest(OUT / "scripts" / "run_airfoil_template_number_q40_extension.py") == digest(ROOT / "src" / "run_airfoil_template_number_q40_extension.py") and digest(OUT / "scripts" / "validate_airfoil_template_number_q40_extension.py") == digest(ROOT / "src" / "validate_airfoil_template_number_q40_extension.py"))

    result = {
        "status": "PASS" if all(item["passed"] for item in checks) else "FAIL",
        "checks": checks,
        "counts": {"safe_samples": 1651, "training_samples": 1321, "holdout_samples": 330, "new_cells": len(new), "combined_cells": len(combined), "valid_new_cells": len(valid), "failed_new_cells": len(failed_visible), "non_converged_new_cells": sum(row["status"] == "non_converged" for row in new)},
    }
    (OUT / "validation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
