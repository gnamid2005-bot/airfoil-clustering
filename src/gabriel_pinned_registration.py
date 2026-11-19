from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


DTYPE = torch.float64
EXCLUDE_TOKENS = ("flap", "main-flap", "slat", "30p-30n")


@dataclass(frozen=True)
class RegistrationConfig:
    grid_n: int = 30
    grid_xlim: tuple[float, float] = (-0.05, 1.05)
    grid_ylim: tuple[float, float] = (-0.35, 0.35)
    rk_steps: int = 5
    lambda_init: float = 1e-1
    regularisation_fraction: float = 0.2
    n_cycles: int = 8
    max_lbfgs_iterations_per_cycle: int = 100
    line_search: str = "strong_wolfe"
    tolerance_grad: float = 1e-14
    tolerance_change: float = 1e-14

    @property
    def parameter_count(self) -> int:
        return 2 * self.grid_n * self.grid_n


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


# Gabriel preprocessing, transcribed from pinned_registration.py.
def resample_open(segment: np.ndarray, n: int, include_end: bool = False) -> np.ndarray:
    arc = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(segment, axis=0), axis=1))]
    )
    requested = np.linspace(0.0, arc[-1], n if include_end else n + 1)[:n]
    return np.stack(
        [
            np.interp(requested, arc, segment[:, 0]),
            np.interp(requested, arc, segment[:, 1]),
        ],
        axis=1,
    )


def resample_pinned(coordinates: np.ndarray, n: int = 200) -> tuple[np.ndarray, int]:
    if n % 2:
        raise ValueError("n must be even")
    leading_index = int(np.argmin(coordinates[:, 0]))
    upper = coordinates[: leading_index + 1]
    lower = np.vstack([coordinates[leading_index:], coordinates[:1]])
    contour = np.vstack(
        [resample_open(upper, n // 2), resample_open(lower, n // 2)]
    )
    return contour, leading_index


def parse_gabriel_file(path: Path) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Apply Gabriel's parsing and scalar validity filters without geometric repair."""
    record: dict[str, Any] = {
        "filename": path.name,
        "raw_source_path": str(path),
        "raw_sha256": sha256_file(path),
        "excluded": False,
        "reference_filter_passed": False,
        "failure_reason": "",
    }
    if any(token in path.name.lower() for token in EXCLUDE_TOKENS):
        record.update(excluded=True, failure_reason="excluded_filename_token")
        return None, record
    try:
        with path.open() as handle:
            handle.readline()
            rows = [
                (float(parts[0]), float(parts[1]))
                for parts in (line.split() for line in handle)
                if len(parts) == 2
            ]
        coordinates = np.asarray(rows, dtype=np.float64)
    except Exception as exc:
        record["failure_reason"] = f"parse_error:{type(exc).__name__}:{exc}"
        return None, record
    record["raw_point_count"] = int(len(coordinates))
    if len(coordinates) < 20:
        record["failure_reason"] = "reference_filter:too_few_points"
        return None, record
    if len(coordinates) > 5000:
        record["failure_reason"] = "reference_filter:too_many_points"
        return None, record
    if not np.isfinite(coordinates).all():
        record["failure_reason"] = "reference_filter:nonfinite_coordinates"
        return None, record
    if float(np.abs(coordinates).max()) > 1e3:
        record["failure_reason"] = "reference_filter:coordinate_magnitude"
        return None, record
    chord = float(coordinates[:, 0].max() - coordinates[:, 0].min())
    record["raw_chord"] = chord
    if chord < 0.5 or chord <= 0.0:
        record["failure_reason"] = "reference_filter:short_or_nonpositive_chord"
        return None, record
    normalised = (coordinates - [coordinates[:, 0].min(), 0.0]) / chord
    if float(np.abs(normalised[:, 1]).max()) > 0.5:
        record["failure_reason"] = "reference_filter:normalised_y_exceeds_0.5"
        return None, record
    record["reference_filter_passed"] = True
    return normalised, record


def signed_polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def self_intersection_count(points: np.ndarray, tolerance: float = 1e-12) -> int:
    n = len(points)
    starts = points
    ends = np.roll(points, -1, axis=0)
    left, right = np.triu_indices(n, k=1)
    adjacent = (right == left + 1) | ((left == 0) & (right == n - 1))
    left = left[~adjacent]
    right = right[~adjacent]
    p1, p2 = starts[left], ends[left]
    q1, q2 = starts[right], ends[right]
    bbox = (
        (np.maximum(np.minimum(p1[:, 0], p2[:, 0]), np.minimum(q1[:, 0], q2[:, 0]))
         <= np.minimum(np.maximum(p1[:, 0], p2[:, 0]), np.maximum(q1[:, 0], q2[:, 0])) + tolerance)
        & (np.maximum(np.minimum(p1[:, 1], p2[:, 1]), np.minimum(q1[:, 1], q2[:, 1]))
           <= np.minimum(np.maximum(p1[:, 1], p2[:, 1]), np.maximum(q1[:, 1], q2[:, 1])) + tolerance)
    )
    if not bbox.any():
        return 0
    p1, p2, q1, q2 = p1[bbox], p2[bbox], q1[bbox], q2[bbox]

    def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]

    o1 = cross(p2 - p1, q1 - p1)
    o2 = cross(p2 - p1, q2 - p1)
    o3 = cross(q2 - q1, p1 - q1)
    o4 = cross(q2 - q1, p2 - q1)
    proper = (o1 * o2 < -(tolerance**2)) & (o3 * o4 < -(tolerance**2))
    collinear_touch = (
        ((np.abs(o1) <= tolerance) | (np.abs(o2) <= tolerance))
        & ((np.abs(o3) <= tolerance) | (np.abs(o4) <= tolerance))
    )
    return int(np.count_nonzero(proper | collinear_touch))


def raw_geometry_diagnostics(coordinates: np.ndarray) -> dict[str, Any]:
    x = coordinates[:, 0]
    leading_index = int(np.argmin(x))
    chord = float(x.max() - x.min())
    scale = max(chord, 1e-12)
    leading_tolerance = max(1e-10, 1e-6 * scale)
    trailing_tolerance = 0.02 * scale
    exact_min_ties = int(np.count_nonzero(np.isclose(x, x.min(), atol=1e-12, rtol=0.0)))
    near_min_ties = int(np.count_nonzero(x <= x.min() + leading_tolerance))
    unique_offsets = np.unique(np.round((x - x.min()) / scale, 14))
    second_minimum_gap = float(unique_offsets[1]) if len(unique_offsets) > 1 else math.inf
    raw_te_gap = float(np.linalg.norm(coordinates[0] - coordinates[-1]))
    duplicated_te = bool(raw_te_gap <= max(1e-10, 1e-7 * scale))
    first_is_te = bool(x[0] >= x.max() - trailing_tolerance)
    last_is_te = bool(x[-1] >= x.max() - trailing_tolerance)
    split_sized = bool(leading_index >= 2 and leading_index <= len(x) - 3)
    upper = coordinates[: leading_index + 1]
    lower = coordinates[leading_index:]
    upper_dx = np.diff(upper[:, 0]) if len(upper) > 1 else np.asarray([])
    lower_dx = np.diff(lower[:, 0]) if len(lower) > 1 else np.asarray([])
    upper_direction_fraction = float(np.mean(upper_dx <= 1e-8 * scale)) if len(upper_dx) else 0.0
    lower_direction_fraction = float(np.mean(lower_dx >= -1e-8 * scale)) if len(lower_dx) else 0.0
    upper_mean_y = float(np.mean(upper[:, 1])) if len(upper) else math.nan
    lower_mean_y = float(np.mean(lower[:, 1])) if len(lower) else math.nan
    surface_identity_ok = bool(upper_mean_y > lower_mean_y)
    expected_traversal = bool(
        split_sized
        and first_is_te
        and last_is_te
        and upper_direction_fraction >= 0.8
        and lower_direction_fraction >= 0.8
        and surface_identity_ok
    )
    upper_length = float(np.linalg.norm(np.diff(upper, axis=0), axis=1).sum()) if len(upper) > 1 else 0.0
    lower_closed = np.vstack([lower, coordinates[:1]])
    lower_length = float(np.linalg.norm(np.diff(lower_closed, axis=0), axis=1).sum())
    return {
        "raw_le_index": leading_index,
        "raw_min_x_exact_tie_count": exact_min_ties,
        "raw_min_x_near_tie_count": near_min_ties,
        "raw_second_minimum_x_gap_chords": second_minimum_gap,
        "raw_te_gap": raw_te_gap,
        "raw_te_endpoint_duplicated": duplicated_te,
        "raw_first_point_is_te": first_is_te,
        "raw_last_point_is_te": last_is_te,
        "raw_split_has_minimum_points": split_sized,
        "raw_upper_expected_x_direction_fraction": upper_direction_fraction,
        "raw_lower_expected_x_direction_fraction": lower_direction_fraction,
        "raw_upper_mean_y": upper_mean_y,
        "raw_lower_mean_y": lower_mean_y,
        "raw_upper_is_above_lower": surface_identity_ok,
        "raw_expected_selig_traversal": expected_traversal,
        "raw_upper_arc_length": upper_length,
        "raw_lower_arc_length": lower_length,
    }


def validate_pinned_contour(contour: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    result["shape_valid"] = bool(contour.shape == (200, 2))
    result["dtype"] = str(contour.dtype)
    result["finite"] = bool(np.isfinite(contour).all())
    if not result["shape_valid"] or not result["finite"]:
        result.update(valid=False, failure_reason="shape_or_finite_failure")
        return result
    segment_lengths = np.linalg.norm(np.roll(contour, -1, axis=0) - contour, axis=1)
    upper_lengths = np.linalg.norm(np.diff(contour[:101], axis=0), axis=1)
    lower_path = np.vstack([contour[100:], contour[:1]])
    lower_lengths = np.linalg.norm(np.diff(lower_path, axis=0), axis=1)
    upper_mean = float(upper_lengths.mean())
    lower_mean = float(lower_lengths.mean())
    upper_rel = np.abs(upper_lengths - upper_mean) / max(upper_mean, 1e-15)
    lower_rel = np.abs(lower_lengths - lower_mean) / max(lower_mean, 1e-15)
    area = signed_polygon_area(contour)
    intersections = self_intersection_count(contour)
    index0_te = bool(contour[0, 0] >= 0.98)
    index100_le = bool(abs(float(contour[100, 0])) <= 1e-10)
    upper_direction = float(np.mean(np.diff(contour[:100, 0]) <= 1e-8))
    lower_direction = float(np.mean(np.diff(contour[100:, 0]) >= -1e-8))
    upper_above_lower = bool(float(contour[:100, 1].mean()) > float(contour[100:, 1].mean()))
    no_zero_segments = bool(float(segment_lengths.min()) > 1e-12)
    # Gabriel samples exact, equally spaced polyline arc coordinates.  Euclidean
    # distances between adjacent output samples can shorten when an interval
    # straddles a raw vertex, so validation uses the distribution rather than a
    # single worst interval.
    approximately_uniform = bool(
        np.quantile(upper_rel, 0.95) <= 0.02
        and np.quantile(lower_rel, 0.95) <= 0.02
        and upper_lengths.std() / max(upper_mean, 1e-15) <= 0.10
        and lower_lengths.std() / max(lower_mean, 1e-15) <= 0.10
    )
    result.update(
        x_min=float(contour[:, 0].min()),
        x_max=float(contour[:, 0].max()),
        index0=list(map(float, contour[0])),
        index99=list(map(float, contour[99])),
        index100=list(map(float, contour[100])),
        index199=list(map(float, contour[199])),
        index0_is_te=index0_te,
        index100_is_le=index100_le,
        upper_expected_direction_fraction=upper_direction,
        lower_expected_direction_fraction=lower_direction,
        upper_is_above_lower=upper_above_lower,
        minimum_segment_length=float(segment_lengths.min()),
        closure_segment_length=float(segment_lengths[-1]),
        upper_segment_cv=float(upper_lengths.std() / max(upper_mean, 1e-15)),
        lower_segment_cv=float(lower_lengths.std() / max(lower_mean, 1e-15)),
        upper_segment_p95_relative_deviation=float(np.quantile(upper_rel, 0.95)),
        lower_segment_p95_relative_deviation=float(np.quantile(lower_rel, 0.95)),
        upper_segment_max_relative_deviation=float(upper_rel.max()),
        lower_segment_max_relative_deviation=float(lower_rel.max()),
        approximately_uniform_arc_length=approximately_uniform,
        no_zero_length_segments=no_zero_segments,
        signed_area=area,
        orientation="counterclockwise" if area > 0 else "clockwise",
        self_intersection_count=intersections,
    )
    checks = [
        index0_te,
        index100_le,
        upper_direction >= 0.8,
        lower_direction >= 0.8,
        upper_above_lower,
        no_zero_segments,
        approximately_uniform,
        area > 1e-8,
        intersections == 0,
        contour[:, 0].min() >= -1e-10,
        contour[:, 0].max() <= 1.0 + 1e-10,
    ]
    result["valid"] = bool(all(checks))
    failure_names = [
        name
        for name, passed in zip(
            [
                "index0_not_te",
                "index100_not_le",
                "upper_order",
                "lower_order",
                "surface_identity",
                "zero_length_segment",
                "nonuniform_resampling",
                "nonpositive_area",
                "self_intersection",
                "x_below_zero",
                "x_above_one",
            ],
            checks,
        )
        if not passed
    ]
    result["failure_reason"] = ";".join(failure_names)
    return result


# Gabriel registration, transcribed from pinned_registration.py.
def cubic_bspline_1d(t: torch.Tensor) -> torch.Tensor:
    absolute = torch.abs(t)
    inner = (2.0 / 3.0) - absolute * absolute + 0.5 * absolute**3
    outer = ((2.0 - absolute) ** 3) / 6.0
    return torch.where(
        absolute < 1.0,
        inner,
        torch.where(absolute < 2.0, outer, torch.zeros_like(t)),
    )


def local_basis(query: torch.Tensor, controls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    coordinate = (query - controls[0]) / (controls[1] - controls[0])
    first = (torch.floor(coordinate).long() - 1).clamp(0, len(controls) - 4)
    indices = first[:, None] + torch.arange(4, device=query.device)
    return first, cubic_bspline_1d(coordinate[:, None] - indices)


def velocity(
    query_xy: torch.Tensor,
    coefficients: torch.Tensor,
    control_x: torch.Tensor,
    control_y: torch.Tensor,
) -> torch.Tensor:
    ix, bx = local_basis(query_xy[:, 0], control_x)
    iy, by = local_basis(query_xy[:, 1], control_y)
    offsets = torch.arange(4, device=query_xy.device)
    patch = coefficients[
        (ix[:, None] + offsets)[:, :, None],
        (iy[:, None] + offsets)[:, None, :],
    ]
    return torch.einsum("ni,nj,nijd->nd", bx, by, patch)


def dense_velocity_reference(
    query_xy: torch.Tensor,
    coefficients: torch.Tensor,
    control_x: torch.Tensor,
    control_y: torch.Tensor,
) -> torch.Tensor:
    ux = (query_xy[:, 0, None] - control_x[0]) / (control_x[1] - control_x[0])
    uy = (query_xy[:, 1, None] - control_y[0]) / (control_y[1] - control_y[0])
    indices_x = torch.arange(len(control_x), device=query_xy.device, dtype=query_xy.dtype)
    indices_y = torch.arange(len(control_y), device=query_xy.device, dtype=query_xy.dtype)
    bx = cubic_bspline_1d(ux - indices_x[None])
    by = cubic_bspline_1d(uy - indices_y[None])
    return torch.einsum("ni,nj,ijd->nd", bx, by, coefficients)


def flow(
    initial: torch.Tensor,
    coefficients: torch.Tensor,
    control_x: torch.Tensor,
    control_y: torch.Tensor,
    n_steps: int = 5,
    return_trajectory: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    current = initial
    dt = 1.0 / n_steps
    visited = [current]
    for _ in range(n_steps):
        k1 = velocity(current, coefficients, control_x, control_y)
        q2 = current + 0.5 * dt * k1
        k2 = velocity(q2, coefficients, control_x, control_y)
        q3 = current + 0.5 * dt * k2
        k3 = velocity(q3, coefficients, control_x, control_y)
        q4 = current + dt * k3
        k4 = velocity(q4, coefficients, control_x, control_y)
        current = current + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        if return_trajectory:
            visited.extend([q2, q3, q4, current])
    if return_trajectory:
        return current, torch.stack(visited)
    return current


def coefficient_smoothness(coefficients: torch.Tensor) -> torch.Tensor:
    return ((coefficients[1:] - coefficients[:-1]) ** 2).sum() + (
        (coefficients[:, 1:] - coefficients[:, :-1]) ** 2
    ).sum()


def attachment(
    source: torch.Tensor,
    target: torch.Tensor,
    coefficients: torch.Tensor,
    control_x: torch.Tensor,
    control_y: torch.Tensor,
    rk_steps: int,
) -> torch.Tensor:
    transformed = flow(source, coefficients, control_x, control_y, rk_steps)
    return ((transformed - target) ** 2).sum(1).mean()


def optimise_registration(
    source: np.ndarray,
    target: np.ndarray,
    config: RegistrationConfig,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    torch.set_default_dtype(DTYPE)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    source_t = torch.from_numpy(np.ascontiguousarray(source, dtype=np.float64))
    target_t = torch.from_numpy(np.ascontiguousarray(target, dtype=np.float64))
    control_x = torch.linspace(*config.grid_xlim, config.grid_n, dtype=DTYPE)
    control_y = torch.linspace(*config.grid_ylim, config.grid_n, dtype=DTYPE)
    coefficients = torch.zeros(
        (config.grid_n, config.grid_n, 2), dtype=DTYPE, requires_grad=True
    )
    trace: list[dict[str, Any]] = []
    cycles: list[dict[str, Any]] = []
    global_closure = 0
    start = time.perf_counter()
    status = "completed"
    failure_reason = ""
    try:
        for cycle_index in range(config.n_cycles):
            cycle_start = time.perf_counter()
            with torch.no_grad():
                current_attachment = float(
                    attachment(
                        source_t,
                        target_t,
                        coefficients,
                        control_x,
                        control_y,
                        config.rk_steps,
                    )
                )
                current_smoothness = float(coefficient_smoothness(coefficients))
            regularisation = (
                config.lambda_init
                if cycle_index == 0
                else min(
                    config.regularisation_fraction
                    * current_attachment
                    / max(current_smoothness, 1e-12),
                    config.lambda_init,
                )
            )
            optimizer = torch.optim.LBFGS(
                [coefficients],
                max_iter=config.max_lbfgs_iterations_per_cycle,
                line_search_fn=config.line_search,
                tolerance_grad=config.tolerance_grad,
                tolerance_change=config.tolerance_change,
            )
            within_cycle = 0

            def closure() -> torch.Tensor:
                nonlocal within_cycle, global_closure
                optimizer.zero_grad()
                current_att = attachment(
                    source_t,
                    target_t,
                    coefficients,
                    control_x,
                    control_y,
                    config.rk_steps,
                )
                current_smooth = coefficient_smoothness(coefficients)
                loss = current_att + regularisation * current_smooth
                loss.backward()
                within_cycle += 1
                global_closure += 1
                gradient_norm = float(torch.linalg.vector_norm(coefficients.grad).detach())
                row = {
                    "cycle": cycle_index + 1,
                    "closure_evaluation_in_cycle": within_cycle,
                    "closure_evaluation_global": global_closure,
                    "lambda": regularisation,
                    "attachment": float(current_att.detach()),
                    "coefficient_smoothness": float(current_smooth.detach()),
                    "weighted_regularisation": float(
                        regularisation * current_smooth.detach()
                    ),
                    "total_objective": float(loss.detach()),
                    "gradient_norm": gradient_norm,
                    "elapsed_seconds": time.perf_counter() - start,
                }
                trace.append(row)
                return loss

            optimizer.step(closure)
            state = optimizer.state[coefficients]
            cycle_row = {
                "cycle": cycle_index + 1,
                "lambda": regularisation,
                "attachment_at_cycle_start": current_attachment,
                "smoothness_at_cycle_start": current_smoothness,
                "closure_evaluations": within_cycle,
                "optimizer_reported_iterations": int(state.get("n_iter", 0)),
                "optimizer_reported_function_evaluations": int(state.get("func_evals", 0)),
                "runtime_seconds": time.perf_counter() - cycle_start,
                "termination_information": "LBFGS step completed; PyTorch exposes counts but no categorical reason",
            }
            cycles.append(cycle_row)
            if progress is not None:
                progress(cycle_row)
    except Exception as exc:
        status = "failed"
        failure_reason = f"{type(exc).__name__}:{exc}"
    runtime = time.perf_counter() - start
    with torch.no_grad():
        deformed = flow(
            source_t, coefficients, control_x, control_y, config.rk_steps
        ).detach().cpu().numpy()
        final_attachment = float(
            ((torch.from_numpy(deformed) - target_t) ** 2).sum(1).mean()
        )
        final_smoothness = float(coefficient_smoothness(coefficients))
    final_lambda = float(cycles[-1]["lambda"]) if cycles else config.lambda_init
    finite = bool(
        np.isfinite(deformed).all()
        and np.isfinite(coefficients.detach().cpu().numpy()).all()
        and math.isfinite(final_attachment)
        and math.isfinite(final_smoothness)
    )
    if not finite and status == "completed":
        status = "nonfinite"
        failure_reason = "nonfinite_final_state"
    return {
        "status": status,
        "failure_reason": failure_reason,
        "finite": finite,
        "coefficients": coefficients.detach().cpu().numpy(),
        "control_x": control_x.detach().cpu().numpy(),
        "control_y": control_y.detach().cpu().numpy(),
        "deformed": deformed,
        "trace": trace,
        "cycles": cycles,
        "runtime_seconds": runtime,
        "closure_evaluations": global_closure,
        "final_lambda": final_lambda,
        "final_attachment": final_attachment,
        "final_coefficient_smoothness": final_smoothness,
        "final_weighted_regularisation": final_lambda * final_smoothness,
        "final_objective": final_attachment + final_lambda * final_smoothness,
        "config": asdict(config),
    }


def correspondence_metrics(source_or_deformed: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    residual = source_or_deformed - target
    squared = np.sum(residual**2, axis=1)
    magnitudes = np.sqrt(squared)
    mse = float(squared.mean())
    regions = {
        "TE": np.r_[0:10, 190:200],
        "LE": np.arange(90, 110),
        "mid": np.r_[10:90, 110:190],
    }
    output: dict[str, Any] = {
        "pointwise_mse": mse,
        "pointwise_rmse": math.sqrt(mse),
        "coordinatewise_mse": float(np.mean(residual**2)),
        "maximum_correspondence_error": float(magnitudes.max()),
        "median_correspondence_error": float(np.median(magnitudes)),
        "p95_correspondence_error": float(np.quantile(magnitudes, 0.95)),
        "residual_vectors": residual,
        "residual_magnitudes": magnitudes,
        "squared_errors": squared,
    }
    for name, indices in regions.items():
        values = magnitudes[indices]
        sq = squared[indices]
        output[f"{name}_point_count"] = int(len(indices))
        output[f"{name}_pointwise_mse"] = float(sq.mean())
        output[f"{name}_mean_error"] = float(values.mean())
        output[f"{name}_maximum_error"] = float(values.max())
        output[f"{name}_absolute_squared_error"] = float(sq.sum())
        output[f"{name}_squared_error_fraction"] = float(
            sq.sum() / max(squared.sum(), 1e-300)
        )
    return output


def diagnostic_grid(
    coefficients: np.ndarray,
    control_x: np.ndarray,
    control_y: np.ndarray,
    rk_steps: int,
    resolution: int = 101,
) -> dict[str, Any]:
    x = np.linspace(float(control_x[0]), float(control_x[-1]), resolution)
    y = np.linspace(float(control_y[0]), float(control_y[-1]), resolution)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    coeff_t = torch.from_numpy(np.ascontiguousarray(coefficients))
    control_x_t = torch.from_numpy(np.ascontiguousarray(control_x))
    control_y_t = torch.from_numpy(np.ascontiguousarray(control_y))
    points_t = torch.from_numpy(np.ascontiguousarray(points))
    with torch.no_grad():
        vel = velocity(points_t, coeff_t, control_x_t, control_y_t).cpu().numpy()
        transformed = flow(
            points_t, coeff_t, control_x_t, control_y_t, rk_steps
        ).cpu().numpy()
    velocity_grid = vel.reshape(resolution, resolution, 2)
    phi = transformed.reshape(resolution, resolution, 2)
    dphix_dy, dphix_dx = np.gradient(phi[:, :, 0], y, x, edge_order=2)
    dphiy_dy, dphiy_dx = np.gradient(phi[:, :, 1], y, x, edge_order=2)
    determinant = dphix_dx * dphiy_dy - dphix_dy * dphiy_dx
    dvx_dy, dvx_dx = np.gradient(velocity_grid[:, :, 0], y, x, edge_order=2)
    dvy_dy, dvy_dx = np.gradient(velocity_grid[:, :, 1], y, x, edge_order=2)
    dirichlet_density = dvx_dx**2 + dvx_dy**2 + dvy_dx**2 + dvy_dy**2
    lap_vx = np.gradient(dvx_dx, x, axis=1, edge_order=2) + np.gradient(
        dvx_dy, y, axis=0, edge_order=2
    )
    lap_vy = np.gradient(dvy_dx, x, axis=1, edge_order=2) + np.gradient(
        dvy_dy, y, axis=0, edge_order=2
    )
    bending_density = lap_vx**2 + lap_vy**2

    def integrate(field: np.ndarray) -> float:
        return float(np.trapezoid(np.trapezoid(field, x=x, axis=1), x=y, axis=0))

    magnitudes = np.linalg.norm(velocity_grid, axis=2)
    minimum_flat = int(np.argmin(determinant))
    minimum_iy, minimum_ix = np.unravel_index(minimum_flat, determinant.shape)
    return {
        "x": x,
        "y": y,
        "velocity": velocity_grid,
        "phi": phi,
        "jacobian_determinant": determinant,
        "velocity_mean_magnitude": float(magnitudes.mean()),
        "velocity_rms_magnitude": float(np.sqrt(np.mean(magnitudes**2))),
        "velocity_maximum_magnitude": float(magnitudes.max()),
        "physical_dirichlet_integral": integrate(dirichlet_density),
        "bending_laplacian_integral": integrate(bending_density),
        "jacobian_minimum": float(determinant.min()),
        "jacobian_p01": float(np.quantile(determinant, 0.01)),
        "jacobian_p05": float(np.quantile(determinant, 0.05)),
        "jacobian_median": float(np.median(determinant)),
        "jacobian_maximum": float(determinant.max()),
        "jacobian_nonpositive_fraction": float(np.mean(determinant <= 0.0)),
        "jacobian_below_0_05_fraction": float(np.mean(determinant < 0.05)),
        "jacobian_minimum_location_x": float(x[minimum_ix]),
        "jacobian_minimum_location_y": float(y[minimum_iy]),
        "jacobian_all_finite": bool(np.isfinite(determinant).all()),
    }


def deformation_metrics(
    source: np.ndarray,
    deformed: np.ndarray,
    coefficients: np.ndarray,
    grid_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    displacement = np.linalg.norm(deformed - source, axis=1)
    return {
        "mean_source_displacement": float(displacement.mean()),
        "rms_source_displacement": float(np.sqrt(np.mean(displacement**2))),
        "p95_source_displacement": float(np.quantile(displacement, 0.95)),
        "maximum_source_displacement": float(displacement.max()),
        "coefficient_l2_norm": float(np.linalg.norm(coefficients)),
        "velocity_mean_magnitude": grid_diagnostics["velocity_mean_magnitude"],
        "velocity_rms_magnitude": grid_diagnostics["velocity_rms_magnitude"],
        "velocity_maximum_magnitude": grid_diagnostics["velocity_maximum_magnitude"],
        "physical_dirichlet_integral": grid_diagnostics["physical_dirichlet_integral"],
        "bending_laplacian_integral": grid_diagnostics["bending_laplacian_integral"],
    }


def trajectory_domain_diagnostics(
    source: np.ndarray,
    coefficients: np.ndarray,
    control_x: np.ndarray,
    control_y: np.ndarray,
    rk_steps: int,
) -> dict[str, Any]:
    with torch.no_grad():
        _, trajectory = flow(
            torch.from_numpy(np.ascontiguousarray(source)),
            torch.from_numpy(np.ascontiguousarray(coefficients)),
            torch.from_numpy(np.ascontiguousarray(control_x)),
            torch.from_numpy(np.ascontiguousarray(control_y)),
            rk_steps,
            return_trajectory=True,
        )
    values = trajectory.cpu().numpy()
    outside = (
        (values[..., 0] < control_x[0])
        | (values[..., 0] > control_x[-1])
        | (values[..., 1] < control_y[0])
        | (values[..., 1] > control_y[-1])
    )
    return {
        "trajectory_any_outside_control_range": bool(outside.any()),
        "trajectory_outside_sample_fraction": float(outside.mean()),
        "trajectory_min_x": float(values[..., 0].min()),
        "trajectory_max_x": float(values[..., 0].max()),
        "trajectory_min_y": float(values[..., 1].min()),
        "trajectory_max_y": float(values[..., 1].max()),
    }
