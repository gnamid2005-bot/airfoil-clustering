from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import boundary_aware_svf as bas
import run_airfoil_registration_improvement_stage2 as s2


DATA = Path("data/airfoils/processed/airfoils_n100_seed0.npz")
MEAN = Path("outputs/airfoil_registration_improvement_stage2/direct_coordinate_mean/coordinates.npy")
SELECTED = Path("outputs/airfoil_registration_improvement/development_subset/selected_indices.npy")


def main() -> None:
    shapes = np.load(DATA)["shapes"].astype(np.float64)
    mean = np.load(MEAN).astype(np.float64)
    selected = np.load(SELECTED)
    geometry = bas.boundary_geometry(mean)
    controls = bas.make_controls(mean, geometry, "edge", 16, 2.0, "normal_tangent")
    ct = bas.controls_to_torch(controls)
    coefficients = torch.zeros((len(controls.indices), 2), dtype=bas.DTYPE, requires_grad=True)
    points = torch.tensor(mean, dtype=bas.DTYPE)
    zero_boundary = bas.boundary_velocity(points, coefficients, ct)
    global_zero = torch.zeros((24, 24, 2), dtype=bas.DTYPE)
    bounds = (0.0, 1.0, float(shapes[:, :, 1].min() - 0.08), float(shapes[:, :, 1].max() + 0.08))
    mapped1 = bas.integrate(points, global_zero, coefficients, ct, bounds, 15)
    mapped2 = bas.integrate(points, global_zero, coefficients, ct, bounds, 15)
    loss = mapped1.square().mean() + bas.boundary_smoothness(coefficients, ct["groups"])[0]
    loss.backward()
    jac = bas.jacobian(global_zero, coefficients.detach(), ct, bounds, 15, 31).detach().numpy()
    checks = {
        "dataset_shape": list(shapes.shape),
        "selected_indices": selected.tolist(),
        "same_expected_target_count": bool(selected.shape == (9,) and selected[0] == -1),
        "tangent_unit_max_error": float(np.max(np.abs(np.linalg.norm(geometry["tangents"], axis=1) - 1.0))),
        "normal_unit_max_error": float(np.max(np.abs(np.linalg.norm(geometry["normals"], axis=1) - 1.0))),
        "tangent_normal_max_dot": float(np.max(np.abs(np.sum(geometry["tangents"] * geometry["normals"], axis=1)))),
        "zero_boundary_velocity_max": float(zero_boundary.detach().abs().max()),
        "zero_velocity_identity_max_error": float((mapped1 - points).detach().abs().max()),
        "deterministic_integration_max_difference": float((mapped1 - mapped2).detach().abs().max()),
        "gradient_finite": bool(torch.isfinite(coefficients.grad).all()),
        "identity_jacobian_max_error": float(np.max(np.abs(jac - 1.0))),
        "control_locations": int(len(controls.indices)),
        "coefficient_dimensions": list(coefficients.shape),
    }
    checks["passed"] = bool(
        checks["same_expected_target_count"]
        and checks["tangent_unit_max_error"] < 1e-10
        and checks["normal_unit_max_error"] < 1e-10
        and checks["tangent_normal_max_dot"] < 1e-10
        and checks["zero_boundary_velocity_max"] == 0.0
        and checks["zero_velocity_identity_max_error"] == 0.0
        and checks["deterministic_integration_max_difference"] == 0.0
        and checks["gradient_finite"]
        and checks["identity_jacobian_max_error"] < 1e-10
    )
    print(json.dumps(checks, indent=2))
    if not checks["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
