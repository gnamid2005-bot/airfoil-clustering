from __future__ import annotations

import torch
import torch.nn.functional as F


def validate_points_2d(points: torch.Tensor) -> None:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {tuple(points.shape)}")


def validate_coeffs_2d(coeffs: torch.Tensor) -> None:
    if coeffs.ndim != 3 or coeffs.shape[-1] != 2:
        raise ValueError(f"coeffs must have shape (Gx, Gy, 2), got {tuple(coeffs.shape)}")


def points_to_grid_coordinates(
    points: torch.Tensor,
    bounds: tuple[float, float, float, float],
) -> torch.Tensor:
    """Convert physical (x, y) points to grid_sample coordinates in [-1, 1]."""
    validate_points_2d(points)
    x_min, x_max, y_min, y_max = bounds
    x = 2.0 * (points[:, 0] - x_min) / (x_max - x_min) - 1.0
    y = 2.0 * (points[:, 1] - y_min) / (y_max - y_min) - 1.0
    return torch.stack([x, y], dim=1)


def evaluate_velocity_field_2d(
    points: torch.Tensor,
    coeffs: torch.Tensor,
    bounds: tuple[float, float, float, float],
) -> torch.Tensor:
    """
    Evaluate a 2D stationary velocity field with bilinear interpolation.

    Coefficients are stored as (Gx, Gy, 2), where axis 0 is physical x,
    axis 1 is physical y, and the last axis stores (vx, vy). For PyTorch
    grid_sample, this is converted to an image tensor (1, 2, Gy, Gx).

    grid_sample uses normalized coordinates in [-1, 1] with align_corners=True,
    so x=-1 maps to x_min and x=1 maps to x_max. Boundary handling uses
    padding_mode="border", which clamps queries outside the domain to the
    nearest velocity value.
    """
    validate_points_2d(points)
    validate_coeffs_2d(coeffs)
    image = coeffs.permute(2, 1, 0).unsqueeze(0)
    grid = points_to_grid_coordinates(points, bounds).view(1, points.shape[0], 1, 2)
    sampled = F.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.squeeze(0).squeeze(-1).transpose(0, 1)


def integrate_svf_2d(
    points: torch.Tensor,
    coeffs: torch.Tensor,
    bounds: tuple[float, float, float, float],
    num_steps: int,
) -> torch.Tensor:
    """Integrate dx/dt = v(x) over t in [0, 1] with explicit Euler steps."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    validate_points_2d(points)
    validate_coeffs_2d(coeffs)
    x = points
    dt = 1.0 / float(num_steps)
    for _ in range(num_steps):
        x = x + dt * evaluate_velocity_field_2d(x, coeffs, bounds)
    return x


def dirichlet_smoothness_2d(coeffs: torch.Tensor) -> torch.Tensor:
    """Finite-difference Dirichlet smoothness for a 2D coefficient grid."""
    validate_coeffs_2d(coeffs)
    dx = (coeffs[1:, :, :] - coeffs[:-1, :, :]).pow(2).mean()
    dy = (coeffs[:, 1:, :] - coeffs[:, :-1, :]).pow(2).mean()
    return dx + dy


def velocity_magnitude_2d(coeffs: torch.Tensor) -> torch.Tensor:
    validate_coeffs_2d(coeffs)
    return coeffs.norm(dim=2).mean()


def corresponding_point_mse(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    validate_points_2d(source)
    validate_points_2d(target)
    if source.shape != target.shape:
        raise ValueError("source and target must have identical shape for corresponding loss")
    return (source - target).pow(2).sum(dim=1).mean()


def chamfer_distance_2d(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    validate_points_2d(source)
    validate_points_2d(target)
    distances = torch.cdist(source, target, p=2).pow(2)
    return distances.min(dim=1).values.mean() + distances.min(dim=0).values.mean()


def jacobian_determinants_2d(
    coeffs: torch.Tensor,
    bounds: tuple[float, float, float, float],
    num_steps: int,
    resolution: int = 41,
) -> torch.Tensor:
    """Evaluate det(D phi) on a regular physical grid with central differences."""
    if resolution < 5:
        raise ValueError("resolution must be at least 5")
    x_min, x_max, y_min, y_max = bounds
    xs = torch.linspace(x_min, x_max, resolution, dtype=coeffs.dtype, device=coeffs.device)
    ys = torch.linspace(y_min, y_max, resolution, dtype=coeffs.dtype, device=coeffs.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    with torch.no_grad():
        transformed = integrate_svf_2d(points, coeffs, bounds, num_steps)
    field = transformed.reshape(resolution, resolution, 2)
    dx = (x_max - x_min) / float(resolution - 1)
    dy = (y_max - y_min) / float(resolution - 1)
    dphi_dx = (field[1:-1, 2:, :] - field[1:-1, :-2, :]) / (2.0 * dx)
    dphi_dy = (field[2:, 1:-1, :] - field[:-2, 1:-1, :]) / (2.0 * dy)
    return dphi_dx[:, :, 0] * dphi_dy[:, :, 1] - dphi_dx[:, :, 1] * dphi_dy[:, :, 0]
