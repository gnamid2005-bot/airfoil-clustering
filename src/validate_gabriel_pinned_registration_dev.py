from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

import gabriel_pinned_registration as implementation


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "references/pinned_registration.py"
OUT = ROOT / "outputs/gabriel_pinned_registration_dev"
DATA = ROOT / "data/airfoils/processed_gabriel/airfoils_pinned_arclength_n200.npz"


def load_reference():
    spec = importlib.util.spec_from_file_location("gabriel_reference_pinned", REFERENCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Gabriel reference module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact_reference_parse(path: Path) -> np.ndarray | None:
    if any(token in path.name.lower() for token in ("flap", "main-flap", "slat", "30p-30n")):
        return None
    with path.open() as handle:
        handle.readline()
        rows = [
            (float(parts[0]), float(parts[1]))
            for parts in (line.split() for line in handle)
            if len(parts) == 2
        ]
    coordinates = np.asarray(rows, dtype=np.float64)
    if len(coordinates) < 20 or len(coordinates) > 5000 or np.abs(coordinates).max() > 1e3:
        return None
    chord = coordinates[:, 0].max() - coordinates[:, 0].min()
    if chord < 0.5 or chord <= 0:
        return None
    coordinates = (coordinates - [coordinates[:, 0].min(), 0.0]) / chord
    if np.abs(coordinates[:, 1]).max() > 0.5:
        return None
    return coordinates


def main() -> None:
    reference = load_reference()
    archive = np.load(DATA, allow_pickle=False)
    contours = archive["contours"]
    filenames = archive["filenames"].astype(str)
    paths = archive["raw_source_paths"].astype(str)
    maximum_preprocessing_difference = 0.0
    for index, relative in enumerate(paths):
        coordinates = exact_reference_parse(ROOT / relative)
        if coordinates is None:
            raise RuntimeError(f"saved accepted file fails exact reference filter: {filenames[index]}")
        expected = reference.resample_pinned(coordinates, 200)
        maximum_preprocessing_difference = max(
            maximum_preprocessing_difference,
            float(np.max(np.abs(expected - contours[index]))),
        )

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    rng = np.random.default_rng(20260818)
    control_x = torch.from_numpy(np.linspace(-0.05, 1.05, 30))
    control_y = torch.from_numpy(np.linspace(-0.35, 0.35, 30))
    coefficients = torch.from_numpy(rng.normal(scale=0.02, size=(30, 30, 2)))
    points = torch.from_numpy(
        np.column_stack([rng.uniform(0.0, 1.0, 128), rng.uniform(-0.3, 0.3, 128)])
    )
    basis_arguments = torch.from_numpy(rng.uniform(-3.0, 3.0, 1000))
    basis_difference = float(
        torch.max(
            torch.abs(
                reference.cubic_bspline_1d(basis_arguments)
                - implementation.cubic_bspline_1d(basis_arguments)
            )
        )
    )
    velocity_difference = float(
        torch.max(
            torch.abs(
                reference.velocity(points, coefficients, control_x, control_y)
                - implementation.velocity(points, coefficients, control_x, control_y)
            )
        )
    )
    flow_difference = float(
        torch.max(
            torch.abs(
                reference.flow(points, coefficients, control_x, control_y, n_steps=5)
                - implementation.flow(points, coefficients, control_x, control_y, n_steps=5)
            )
        )
    )
    smoothness_difference = abs(
        float(reference.smoothness(coefficients))
        - float(implementation.coefficient_smoothness(coefficients))
    )
    config = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
    reference_settings = {
        "grid_n": reference.GRID_N,
        "grid_xlim": list(reference.GRID_XLIM),
        "grid_ylim": list(reference.GRID_YLIM),
        "rk_steps": reference.RK_STEPS,
        "lambda_init": reference.LAMBDA_SMOOTH,
        "regularisation_fraction": reference.REG_FRACTION,
        "n_cycles": reference.N_CYCLES,
        "max_lbfgs_iterations_per_cycle": reference.ITERS_PER_CYCLE,
        "line_search": "strong_wolfe",
        "tolerance_grad": 1e-14,
        "tolerance_change": 1e-14,
    }
    case_files = sorted((OUT / "cases").glob("*.json"))
    checks = {
        "all_saved_contours_match_reference_resampling": maximum_preprocessing_difference == 0.0,
        "cubic_basis_matches_reference": basis_difference == 0.0,
        "local_velocity_matches_reference": velocity_difference == 0.0,
        "rk4_flow_matches_reference": flow_difference == 0.0,
        "coefficient_smoothness_matches_reference": smoothness_difference == 0.0,
        "registration_defaults_match_reference_constants": config["registration_config"] == reference_settings,
        "exactly_nine_case_json_files": len(case_files) == 9,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "saved_contours_checked": len(contours),
        "maximum_preprocessing_coordinate_difference": maximum_preprocessing_difference,
        "cubic_basis_maximum_difference": basis_difference,
        "velocity_maximum_difference": velocity_difference,
        "rk4_flow_maximum_difference": flow_difference,
        "smoothness_absolute_difference": smoothness_difference,
        "reference_sha256": implementation.sha256_file(REFERENCE),
    }
    implementation.atomic_json(OUT / "source_faithfulness_validation.json", result)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
