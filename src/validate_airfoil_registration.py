import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch

from airfoil_registration import validate_airfoil_shape
from svf import evaluate_velocity_field as evaluate_velocity_field_3d
from svf_2d import (
    corresponding_point_mse,
    dirichlet_smoothness_2d,
    evaluate_velocity_field_2d,
    integrate_svf_2d,
    jacobian_determinants_2d,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Sanity checks for 2D airfoil registration.")
    parser.add_argument("--output-dir", default="outputs/airfoil_registration_validation")
    return parser.parse_args()


def main():
    _ = parse_args()
    dtype = torch.float64
    bounds = (0.0, 1.0, -0.5, 0.5)
    points = torch.tensor([[0.2, 0.1], [0.8, -0.1]], dtype=dtype)
    zero = torch.zeros(8, 8, 2, dtype=dtype)
    assert torch.allclose(integrate_svf_2d(points, zero, bounds, 5), points)
    const = torch.zeros(8, 8, 2, dtype=dtype)
    const[:, :, 0] = 0.03
    const[:, :, 1] = -0.02
    translated = integrate_svf_2d(points, const, bounds, 10)
    assert torch.allclose(translated, points + torch.tensor([0.03, -0.02], dtype=dtype), atol=1e-10)
    assert float(dirichlet_smoothness_2d(const)) < 1e-14
    assert float(corresponding_point_mse(points, points)) == 0.0
    coeffs = torch.zeros(8, 8, 2, dtype=dtype, requires_grad=True)
    velocity = evaluate_velocity_field_2d(points, coeffs, bounds)
    loss = velocity.pow(2).sum()
    loss.backward()
    assert coeffs.grad is not None
    assert torch.isfinite(coeffs.grad).all()
    jac = jacobian_determinants_2d(zero, bounds, 5, resolution=11)
    assert torch.allclose(jac, torch.ones_like(jac), atol=1e-10)
    try:
        evaluate_velocity_field_3d(torch.zeros(2, 3), torch.zeros(4, 4, 4, 3))
    except Exception as exc:
        raise AssertionError(f"3D SVF smoke check failed: {exc}") from exc
    shape = np.zeros((200, 2), dtype=np.float64)
    x = np.linspace(0.0, 1.0, 100)
    shape[:100, 0] = x
    shape[100:, 0] = x
    shape[:100, 1] = 0.05
    shape[100:, 1] = -0.05
    assert validate_airfoil_shape(shape)["valid"]
    print("airfoil registration validation checks passed")


if __name__ == "__main__":
    main()
