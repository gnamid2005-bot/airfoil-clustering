from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

import run_airfoil_registration_improvement_stage2 as s2
from airfoil_registration import validate_airfoil_shape


DTYPE = torch.float64


def boundary_geometry(points: np.ndarray) -> dict[str, np.ndarray]:
    """Geometry for the canonical upper/lower LE-to-TE point ordering.

    Derivatives are taken independently on each physical surface so the
    duplicated LE/TE transition points are never crossed by a finite
    difference stencil.  Endpoint values use second-order one-sided
    differences.  Normals point outward: left normal on the upper surface,
    right normal on the lower surface.
    """
    if points.shape != (200, 2):
        raise ValueError(f"expected (200, 2), got {points.shape}")
    tangents = np.zeros_like(points, dtype=np.float64)
    normals = np.zeros_like(points, dtype=np.float64)
    curvature = np.zeros(200, dtype=np.float64)
    surface_arc = np.zeros(200, dtype=np.float64)
    surface_arc_normalized = np.zeros(200, dtype=np.float64)
    lengths = []
    for surface, sl in enumerate((slice(0, 100), slice(100, 200))):
        curve = points[sl].astype(np.float64)
        ds = np.linalg.norm(np.diff(curve, axis=0), axis=1)
        if np.any(ds <= 0):
            raise ValueError("boundary contains repeated consecutive points")
        arc = np.r_[0.0, np.cumsum(ds)]
        derivative = np.gradient(curve, arc, axis=0, edge_order=2)
        tangent = derivative / np.maximum(np.linalg.norm(derivative, axis=1, keepdims=True), 1e-15)
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        if surface == 1:
            normal = -normal
        dt_ds = np.gradient(tangent, arc, axis=0, edge_order=2)
        tangents[sl] = tangent
        normals[sl] = normal
        curvature[sl] = np.sum(dt_ds * normal, axis=1)
        surface_arc[sl] = arc
        surface_arc_normalized[sl] = arc / arc[-1]
        lengths.append(float(arc[-1]))

    le_gap = float(np.linalg.norm(points[0] - points[100]))
    te_gap = float(np.linalg.norm(points[99] - points[199]))
    perimeter = lengths[0] + le_gap + lengths[1] + te_gap
    closed_arc = np.empty(200, dtype=np.float64)
    closed_arc[:100] = lengths[0] - surface_arc[:100]
    closed_arc[100:] = lengths[0] + le_gap + surface_arc[100:]
    closed_arc /= perimeter
    return {
        "coordinates": points.astype(np.float64),
        "tangents": tangents,
        "normals": normals,
        "curvature": curvature,
        "curvature_magnitude": np.abs(curvature),
        "surface_arc_length": surface_arc,
        "surface_arc_length_normalized": surface_arc_normalized,
        "closed_arc_length_normalized": closed_arc,
        "surface_id": np.r_[np.zeros(100, dtype=np.int64), np.ones(100, dtype=np.int64)],
        "upper_surface_length": np.asarray(lengths[0]),
        "lower_surface_length": np.asarray(lengths[1]),
        "leading_edge_gap": np.asarray(le_gap),
        "trailing_edge_gap": np.asarray(te_gap),
        "closed_perimeter": np.asarray(perimeter),
    }


def _interpolate_control_group(
    indices: np.ndarray,
    arc: np.ndarray,
    coordinates: np.ndarray,
    tangents: np.ndarray,
    normals: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Place controls continuously at equal arc-length locations.

    The 64-per-edge experiment can request more controls on one surface than
    there are canonical data points inside x<=0.15 or x>=0.85.  Interpolating
    geometry along the fixed boundary is therefore required; duplicating data
    points would create a rank-deficient basis and changing the support would
    violate the experiment definition.
    """
    if count < 1 or len(indices) < 2:
        raise ValueError("each control group needs at least two candidate points")
    order = np.argsort(arc[indices])
    ordered_indices = indices[order]
    values = arc[ordered_indices]
    desired = np.linspace(values.min(), values.max(), count)
    centers = np.column_stack(
        [np.interp(desired, values, coordinates[ordered_indices, axis]) for axis in range(2)]
    )
    tangent = np.column_stack(
        [np.interp(desired, values, tangents[ordered_indices, axis]) for axis in range(2)]
    )
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-15)
    normal = np.column_stack(
        [np.interp(desired, values, normals[ordered_indices, axis]) for axis in range(2)]
    )
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-15)
    nearest = np.asarray(
        [ordered_indices[int(np.argmin(np.abs(values - value)))] for value in desired], dtype=np.int64
    )
    return nearest, centers, tangent, normal


@dataclass(frozen=True)
class BoundaryControls:
    indices: np.ndarray
    centers: np.ndarray
    tangents: np.ndarray
    normals: np.ndarray
    sigma_t: np.ndarray
    sigma_n: np.ndarray
    groups: np.ndarray
    group_names: tuple[str, ...]
    scope: str
    anisotropy_ratio: float
    coefficient_mode: str

    @property
    def scalar_coefficients(self) -> int:
        multiplier = 1 if self.coefficient_mode == "normal" else 2
        return int(len(self.indices) * multiplier)


def make_controls(
    mean: np.ndarray,
    geometry: dict[str, np.ndarray],
    scope: str,
    coefficient_count: int,
    anisotropy_ratio: float,
    coefficient_mode: str,
    edge: float = 0.15,
) -> BoundaryControls:
    """Construct whole-boundary or LE/TE-concentrated controls.

    For ``whole``, coefficient_count is the total number of boundary control
    locations.  For ``edge``, it is the number in each of LE and TE, matching
    the requested 16 LE + 16 TE convention.  Every region is split equally
    across the upper and lower surfaces.
    """
    if coefficient_mode not in {"normal", "normal_tangent"}:
        raise ValueError(coefficient_mode)
    if anisotropy_ratio not in {1.0, 2.0, 4.0}:
        raise ValueError("controlled ratios are 1, 2, and 4")
    arc = geometry["surface_arc_length"]
    groups = []
    names = []
    selected = []
    control_centers = []
    control_tangents = []
    control_normals = []
    specs: list[tuple[str, np.ndarray, int]] = []
    if scope == "whole":
        if coefficient_count % 2:
            raise ValueError("whole-boundary count must be even")
        per_surface = coefficient_count // 2
        specs = [
            ("upper_whole", np.arange(0, 100), per_surface),
            ("lower_whole", np.arange(100, 200), per_surface),
        ]
    elif scope == "edge":
        if coefficient_count % 2:
            raise ValueError("per-edge count must be even")
        per_surface_edge = coefficient_count // 2
        for label, condition in (
            ("LE", mean[:, 0] <= edge + 1e-12),
            ("TE", mean[:, 0] >= 1.0 - edge - 1e-12),
        ):
            for surface, limits in (("upper", (0, 100)), ("lower", (100, 200))):
                idx = np.arange(*limits)
                specs.append((f"{surface}_{label}", idx[condition[idx]], per_surface_edge))
    else:
        raise ValueError(scope)

    sigma_t_values = []
    for group_id, (name, candidates, count) in enumerate(specs):
        idx, centers, tangent, normal = _interpolate_control_group(
            candidates,
            arc,
            mean,
            geometry["tangents"],
            geometry["normals"],
            count,
        )
        spacing = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        # Keep normal reach independent of the number of controls. Otherwise a
        # denser 1-D basis would also become a narrower ambient field and the
        # resolution ablation would confound two factors. The 0.03--0.08 chord
        # range is shared by every target; anisotropy changes only tangent reach.
        sigma_n = float(np.clip(1.75 * np.median(spacing), 0.03, 0.08))
        sigma_t = float(anisotropy_ratio * sigma_n)
        selected.extend(idx.tolist())
        control_centers.extend(centers.tolist())
        control_tangents.extend(tangent.tolist())
        control_normals.extend(normal.tolist())
        groups.extend([group_id] * len(idx))
        sigma_t_values.extend([sigma_t] * len(idx))
        names.append(name)

    indices = np.asarray(selected, dtype=np.int64)
    sigma_t = np.asarray(sigma_t_values, dtype=np.float64)
    return BoundaryControls(
        indices=indices,
        centers=np.asarray(control_centers, dtype=np.float64),
        tangents=np.asarray(control_tangents, dtype=np.float64),
        normals=np.asarray(control_normals, dtype=np.float64),
        sigma_t=sigma_t,
        sigma_n=sigma_t / float(anisotropy_ratio),
        groups=np.asarray(groups, dtype=np.int64),
        group_names=tuple(names),
        scope=scope,
        anisotropy_ratio=float(anisotropy_ratio),
        coefficient_mode=coefficient_mode,
    )


def controls_to_torch(controls: BoundaryControls, device: torch.device | None = None) -> dict[str, torch.Tensor]:
    return {
        "centers": torch.tensor(controls.centers, dtype=DTYPE, device=device),
        "tangents": torch.tensor(controls.tangents, dtype=DTYPE, device=device),
        "normals": torch.tensor(controls.normals, dtype=DTYPE, device=device),
        "sigma_t": torch.tensor(controls.sigma_t, dtype=DTYPE, device=device),
        "sigma_n": torch.tensor(controls.sigma_n, dtype=DTYPE, device=device),
        "groups": torch.tensor(controls.groups, dtype=torch.long, device=device),
    }


def boundary_velocity(points: torch.Tensor, coefficients: torch.Tensor, control_tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    delta = points[:, None, :] - control_tensors["centers"][None, :, :]
    tangent_distance = (delta * control_tensors["tangents"][None, :, :]).sum(dim=2)
    normal_distance = (delta * control_tensors["normals"][None, :, :]).sum(dim=2)
    radius2 = (tangent_distance / control_tensors["sigma_t"][None, :]).square()
    radius2 = radius2 + (normal_distance / control_tensors["sigma_n"][None, :]).square()
    weights = torch.exp(-0.5 * radius2)
    vectors = coefficients[:, 0, None] * control_tensors["normals"]
    if coefficients.shape[1] == 2:
        vectors = vectors + coefficients[:, 1, None] * control_tensors["tangents"]
    return weights @ vectors


def composite_velocity(
    points: torch.Tensor,
    global_coefficients: torch.Tensor,
    boundary_coefficients: torch.Tensor,
    control_tensors: dict[str, torch.Tensor],
    bounds: tuple[float, float, float, float],
) -> torch.Tensor:
    return s2.velocity(points, global_coefficients, bounds, "cartesian") + boundary_velocity(
        points, boundary_coefficients, control_tensors
    )


def integrate(
    points: torch.Tensor,
    global_coefficients: torch.Tensor,
    boundary_coefficients: torch.Tensor,
    control_tensors: dict[str, torch.Tensor],
    bounds: tuple[float, float, float, float],
    steps: int,
) -> torch.Tensor:
    result = points
    dt = 1.0 / steps
    for _ in range(steps):
        result = result + dt * composite_velocity(
            result, global_coefficients, boundary_coefficients, control_tensors, bounds
        )
    return result


def boundary_smoothness(coefficients: torch.Tensor, groups: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    first_terms = []
    second_terms = []
    for group in torch.unique(groups):
        values = coefficients[groups == group]
        first_terms.append((values[1:] - values[:-1]).square().mean())
        if len(values) >= 3:
            second_terms.append((values[2:] - 2.0 * values[1:-1] + values[:-2]).square().mean())
    first = torch.stack(first_terms).mean()
    second = torch.stack(second_terms).mean() if second_terms else first * 0.0
    return first, second


def jacobian(
    global_coefficients: torch.Tensor,
    boundary_coefficients: torch.Tensor,
    control_tensors: dict[str, torch.Tensor],
    bounds: tuple[float, float, float, float],
    steps: int,
    resolution: int,
) -> torch.Tensor:
    x0, x1, y0, y1 = bounds
    xs = torch.linspace(x0, x1, resolution, dtype=DTYPE, device=global_coefficients.device)
    ys = torch.linspace(y0, y1, resolution, dtype=DTYPE, device=global_coefficients.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    mapped = integrate(
        points, global_coefficients, boundary_coefficients, control_tensors, bounds, steps
    ).reshape(resolution, resolution, 2)
    dx = (x1 - x0) / (resolution - 1)
    dy = (y1 - y0) / (resolution - 1)
    derivative_x = (mapped[1:-1, 2:] - mapped[1:-1, :-2]) / (2.0 * dx)
    derivative_y = (mapped[2:, 1:-1] - mapped[:-2, 1:-1]) / (2.0 * dy)
    return (
        derivative_x[:, :, 0] * derivative_y[:, :, 1]
        - derivative_x[:, :, 1] * derivative_y[:, :, 0]
    ).flatten()


def optimise(
    source: torch.Tensor,
    target: torch.Tensor,
    global_initial: torch.Tensor,
    controls: BoundaryControls,
    bounds: tuple[float, float, float, float],
    config: dict,
    strategy: str,
) -> tuple[torch.Tensor, torch.Tensor, float, int, dict[str, float]]:
    dimensions = 1 if controls.coefficient_mode == "normal" else 2
    global_coefficients = global_initial.detach().clone()
    boundary_coefficients = torch.zeros((len(controls.indices), dimensions), dtype=DTYPE)
    control_tensors = controls_to_torch(controls)
    total_runtime = 0.0
    total_calls = 0
    stages = []

    def run_stage(train_global: bool, iterations: int, name: str) -> None:
        nonlocal global_coefficients, boundary_coefficients, total_runtime, total_calls
        global_coefficients = global_coefficients.detach().clone().requires_grad_(train_global)
        boundary_coefficients = boundary_coefficients.detach().clone().requires_grad_(True)
        parameters = ([global_coefficients] if train_global else []) + [boundary_coefficients]
        optimizer = torch.optim.LBFGS(
            parameters,
            lr=1.0,
            max_iter=iterations,
            max_eval=int(iterations * 1.25),
            tolerance_grad=config["tolerance_grad"],
            tolerance_change=config["tolerance_change"],
            line_search_fn="strong_wolfe",
        )
        calls = 0

        def closure() -> torch.Tensor:
            nonlocal calls
            optimizer.zero_grad()
            deformed = integrate(
                source,
                global_coefficients,
                boundary_coefficients,
                control_tensors,
                bounds,
                config["steps"],
            )
            correspondence = (deformed - target).square().sum(dim=1).mean()
            global_dirichlet = s2.dirichlet(global_coefficients, bounds)
            first, second = boundary_smoothness(boundary_coefficients, control_tensors["groups"])
            jac = jacobian(
                global_coefficients,
                boundary_coefficients,
                control_tensors,
                bounds,
                config["steps"],
                config["barrier_resolution"],
            )
            barrier = s2.barrier_raw(jac, config["barrier_tau"])
            objective = (
                correspondence
                + config["lambda_global"] * global_dirichlet
                + config["lambda_boundary_first"] * first
                + config["lambda_boundary_second"] * second
                + config["barrier_weight"] * barrier
            )
            objective.backward()
            calls += 1
            return objective

        start = time.perf_counter()
        optimizer.step(closure)
        runtime = time.perf_counter() - start
        total_runtime += runtime
        total_calls += calls
        stages.append({"name": name, "runtime_seconds": runtime, "closure_evaluations": calls})
        global_coefficients = global_coefficients.detach()
        boundary_coefficients = boundary_coefficients.detach()

    if strategy in {"boundary_only", "freeze_then_joint"}:
        run_stage(False, config["boundary_only_iterations"], "boundary_only")
    if strategy in {"joint", "freeze_then_joint"}:
        run_stage(True, config["joint_iterations"], "joint")
    if strategy not in {"boundary_only", "joint", "freeze_then_joint"}:
        raise ValueError(strategy)
    return global_coefficients, boundary_coefficients, total_runtime, total_calls, {"stages": stages}


def metrics(
    source: torch.Tensor,
    target: torch.Tensor,
    global_coefficients: torch.Tensor,
    boundary_coefficients: torch.Tensor,
    controls: BoundaryControls,
    bounds: tuple[float, float, float, float],
    config: dict,
    refinement_runtime: float,
    closure_evaluations: int,
    baseline_runtime: float,
) -> tuple[dict, np.ndarray, np.ndarray]:
    control_tensors = controls_to_torch(controls)
    with torch.no_grad():
        deformed = integrate(
            source,
            global_coefficients,
            boundary_coefficients,
            control_tensors,
            bounds,
            config["steps"],
        )
        residual = deformed - target
        errors = torch.linalg.norm(residual, dim=1)
        displacement = torch.linalg.norm(deformed - source, dim=1)
        correspondence = float(errors.square().mean())
        global_dirichlet = s2.dirichlet(global_coefficients, bounds)
        first, second = boundary_smoothness(boundary_coefficients, control_tensors["groups"])
        jac = jacobian(
            global_coefficients,
            boundary_coefficients,
            control_tensors,
            bounds,
            config["steps"],
            config["jacobian_report_resolution"],
        ).cpu().numpy()
        barrier = float(s2.barrier_raw(torch.tensor(jac, dtype=DTYPE), config["barrier_tau"]))
        xs = torch.linspace(bounds[0], bounds[1], 51, dtype=DTYPE)
        ys = torch.linspace(bounds[2], bounds[3], 51, dtype=DTYPE)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        reporting_points = torch.stack((xx.flatten(), yy.flatten()), dim=1)
        velocity_magnitude = torch.linalg.norm(
            composite_velocity(
                reporting_points,
                global_coefficients,
                boundary_coefficients,
                control_tensors,
                bounds,
            ),
            dim=1,
        )
        validity = validate_airfoil_shape(deformed.cpu().numpy())
        normal_amplitude = boundary_coefficients[:, 0]
        tangent_amplitude = (
            boundary_coefficients[:, 1]
            if boundary_coefficients.shape[1] == 2
            else torch.zeros_like(normal_amplitude)
        )
    weighted_global = config["lambda_global"] * float(global_dirichlet)
    weighted_first = config["lambda_boundary_first"] * float(first)
    weighted_second = config["lambda_boundary_second"] * float(second)
    weighted_barrier = config["barrier_weight"] * barrier
    result = {
        "correspondence_mse": correspondence,
        "correspondence_rmse": math.sqrt(correspondence),
        "maximum_pointwise_error": float(errors.max()),
        "raw_global_dirichlet": float(global_dirichlet),
        "weighted_global_dirichlet": weighted_global,
        "raw_boundary_first_smoothness": float(first),
        "raw_boundary_second_smoothness": float(second),
        "weighted_boundary_first_smoothness": weighted_first,
        "weighted_boundary_second_smoothness": weighted_second,
        "raw_jacobian_barrier": barrier,
        "weighted_jacobian_barrier": weighted_barrier,
        "total_objective": correspondence + weighted_global + weighted_first + weighted_second + weighted_barrier,
        "mean_vector_field_magnitude": float(velocity_magnitude.mean()),
        "rms_vector_field_magnitude": float(torch.sqrt(velocity_magnitude.square().mean())),
        "maximum_vector_field_magnitude": float(velocity_magnitude.max()),
        "mean_displacement": float(displacement.mean()),
        "maximum_displacement": float(displacement.max()),
        "minimum_jacobian": float(jac.min()),
        "jacobian_p01": float(np.quantile(jac, 0.01)),
        "jacobian_p05": float(np.quantile(jac, 0.05)),
        "jacobian_median": float(np.median(jac)),
        "nonpositive_jacobian_proportion": float(np.mean(jac <= 0.0)),
        "refinement_runtime_seconds": float(refinement_runtime),
        "runtime_seconds": float(baseline_runtime + refinement_runtime),
        "closure_evaluations": int(closure_evaluations),
        "global_coefficients": int(global_coefficients.numel()),
        "boundary_coefficients": int(controls.scalar_coefficients),
        "trainable_coefficients": int(global_coefficients.numel() + controls.scalar_coefficients),
        "mean_absolute_normal_amplitude": float(normal_amplitude.abs().mean()),
        "maximum_absolute_normal_amplitude": float(normal_amplitude.abs().max()),
        "mean_absolute_tangent_amplitude": float(tangent_amplitude.abs().mean()),
        "maximum_absolute_tangent_amplitude": float(tangent_amplitude.abs().max()),
        "valid_geometry": bool(validity["valid"]),
        "safe": bool(validity["valid"] and jac.min() > 0.0 and np.isfinite(correspondence)),
        "converged": None,
        "convergence_status": "L-BFGS completed without exception; backend termination reason unavailable",
    }
    return result, deformed.cpu().numpy(), residual.cpu().numpy()
