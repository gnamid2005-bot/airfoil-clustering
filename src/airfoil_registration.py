from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from airfoil_data import plot_shape
from run_airfoil_gmm_selection import (
    log_gaussian_prob,
    pca_inverse,
    pca_transform,
    shapes_from_representation,
)
from svf_2d import (
    chamfer_distance_2d,
    corresponding_point_mse,
    dirichlet_smoothness_2d,
    integrate_svf_2d,
    jacobian_determinants_2d,
    validate_points_2d,
    velocity_magnitude_2d,
)


matplotlib.use("Agg")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def camber_thickness_from_shapes(shapes: np.ndarray) -> np.ndarray:
    half = shapes.shape[1] // 2
    upper_y = shapes[:, :half, 1]
    lower_y = shapes[:, half:, 1]
    camber = 0.5 * (upper_y + lower_y)
    thickness = upper_y - lower_y
    return np.concatenate([camber, thickness], axis=1)


def validate_airfoil_shape(shape: np.ndarray, severe_negative_thickness: float = 1e-3) -> dict:
    if shape.shape != (200, 2):
        return {"valid": False, "reason": f"bad_shape_{shape.shape}"}
    if not np.isfinite(shape).all():
        return {"valid": False, "reason": "non_finite"}
    half = shape.shape[0] // 2
    upper = shape[:half]
    lower = shape[half:]
    thickness = upper[:, 1] - lower[:, 1]
    if np.any(np.diff(upper[:, 0]) < -1e-10) or np.any(np.diff(lower[:, 0]) < -1e-10):
        return {"valid": False, "reason": "x_not_monotone"}
    if float(thickness.min()) < -severe_negative_thickness:
        return {
            "valid": False,
            "reason": "severe_negative_thickness",
            "minimum_thickness": float(thickness.min()),
        }
    return {
        "valid": True,
        "reason": "",
        "minimum_thickness": float(thickness.min()),
        "leading_edge_gap": float(np.linalg.norm(upper[0] - lower[0])),
        "trailing_edge_gap": float(np.linalg.norm(upper[-1] - lower[-1])),
    }


def load_airfoil_dataset(path: Path) -> dict:
    data = np.load(path)
    shapes = data["shapes"].astype(np.float64)
    return {
        "shapes": shapes,
        "filenames": data["filenames"].astype(str),
        "names": data["names"].astype(str),
        "x_grid": shapes[0, : shapes.shape[1] // 2, 0],
    }


def load_pca(path: Path):
    data = np.load(path)
    return type(
        "PCAFitLoaded",
        (),
        {
            "mean": data["mean"],
            "components": data["components"],
            "explained_variance": data["explained_variance"],
            "explained_variance_ratio": data["explained_variance_ratio"],
        },
    )()


def reconstruct_k1_template(dataset: dict, train_indices: np.ndarray) -> np.ndarray:
    values = camber_thickness_from_shapes(dataset["shapes"][train_indices])
    mean_values = values.mean(axis=0, keepdims=True)
    return shapes_from_representation(mean_values, dataset["x_grid"], "camber_thickness")[0]


def reconstruct_gmm_templates(
    dataset: dict,
    pca_path: Path,
    gmm_result_path: Path,
) -> tuple[np.ndarray, dict]:
    pca = load_pca(pca_path)
    result = np.load(gmm_result_path)
    values = pca_inverse(pca, result["means"])
    shapes = shapes_from_representation(values, dataset["x_grid"], "camber_thickness")
    return shapes, {
        "weights": result["weights"],
        "latent_means": result["means"],
        "covariances": result["covariances"],
    }


def gmm_posteriors_for_shapes(
    shapes: np.ndarray,
    pca_path: Path,
    gmm_result_path: Path,
    covariance_type: str = "full",
) -> np.ndarray:
    pca = load_pca(pca_path)
    result = np.load(gmm_result_path)
    values = camber_thickness_from_shapes(shapes)
    scores = pca_transform(pca, values)
    log_prob = log_gaussian_prob(
        scores,
        result["means"],
        result["covariances"],
        covariance_type,
        1e-12,
    ) + np.log(result["weights"] + 1e-300)
    log_norm = torch.logsumexp(torch.tensor(log_prob), dim=1).numpy()
    return np.exp(log_prob - log_norm[:, None])


def registration_bounds(shapes: list[np.ndarray], margin: float = 0.08) -> tuple[float, float, float, float]:
    all_points = np.concatenate(shapes, axis=0)
    y_min = float(all_points[:, 1].min() - margin)
    y_max = float(all_points[:, 1].max() + margin)
    return (0.0, 1.0, y_min, y_max)


def register_airfoils(
    source_shape: np.ndarray,
    target_shape: np.ndarray,
    config: dict,
    seed: int,
) -> dict:
    set_seed(seed)
    device = torch.device(config.get("device", "cpu"))
    dtype = torch.float64
    source = torch.tensor(source_shape, dtype=dtype, device=device)
    target = torch.tensor(target_shape, dtype=dtype, device=device)
    validate_points_2d(source)
    validate_points_2d(target)
    bounds = tuple(float(value) for value in config["bounds"])
    grid_size = int(config["grid_size"])
    coeffs = torch.zeros(grid_size, grid_size, 2, dtype=dtype, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([coeffs], lr=float(config["lr"]))
    lambda_reg = float(config["lambda_reg"])
    num_steps = int(config["num_steps"])
    num_iters = int(config["num_iters"])
    early_tol = float(config["early_stop_tol"])
    patience = int(config["early_stop_patience"])
    best_loss = float("inf")
    stale = 0
    converged = False
    history = []
    start = time.perf_counter()
    failure_reason = ""

    initial_attachment = float(corresponding_point_mse(source, target).detach().cpu())
    try:
        for iteration in range(num_iters):
            deformed = integrate_svf_2d(source, coeffs, bounds, num_steps)
            data_attachment = corresponding_point_mse(deformed, target)
            raw_regulariser = dirichlet_smoothness_2d(coeffs)
            weighted_regulariser = lambda_reg * raw_regulariser
            total_loss = data_attachment + weighted_regulariser
            if not torch.isfinite(total_loss):
                failure_reason = "non_finite_loss"
                break
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            row = {
                "iteration": iteration,
                "total_loss": float(total_loss.detach().cpu()),
                "data_attachment": float(data_attachment.detach().cpu()),
                "raw_regulariser": float(raw_regulariser.detach().cpu()),
                "weighted_regulariser": float(weighted_regulariser.detach().cpu()),
            }
            history.append(row)
            if best_loss - row["total_loss"] <= early_tol:
                stale += 1
            else:
                stale = 0
                best_loss = row["total_loss"]
            if stale >= patience:
                converged = True
                break
        else:
            converged = True
    except Exception as exc:
        failure_reason = str(exc)

    runtime = time.perf_counter() - start
    with torch.no_grad():
        deformed = integrate_svf_2d(source, coeffs, bounds, num_steps)
        final_data = corresponding_point_mse(deformed, target)
        chamfer = chamfer_distance_2d(deformed, target)
        raw_reg = dirichlet_smoothness_2d(coeffs)
        weighted_reg = lambda_reg * raw_reg
        total = final_data + weighted_reg
        displacement = deformed - source
        jac = jacobian_determinants_2d(coeffs.detach(), bounds, num_steps, int(config["jacobian_resolution"]))
        deformed_np = deformed.detach().cpu().numpy()

    jac_np = jac.detach().cpu().numpy().reshape(-1)
    validity = validate_airfoil_shape(deformed_np, float(config["severe_negative_thickness"]))
    invalid = (
        failure_reason
        or not np.isfinite(deformed_np).all()
        or not np.isfinite(float(total.detach().cpu()))
        or float(np.mean(jac_np <= float(config["jacobian_nonpositive_tol"]))) > 0.0
        or not validity["valid"]
        or not converged
    )
    if invalid and not failure_reason:
        if not converged:
            failure_reason = "not_converged"
        elif not validity["valid"]:
            failure_reason = validity["reason"]
        else:
            failure_reason = "jacobian_or_finiteness_invalid"

    return {
        "coeffs": coeffs.detach().cpu().numpy(),
        "deformed": deformed_np,
        "history": history,
        "metrics": {
            "initial_attachment": initial_attachment,
            "final_data_attachment": float(final_data.detach().cpu()),
            "final_chamfer": float(chamfer.detach().cpu()),
            "raw_regulariser": float(raw_reg.detach().cpu()),
            "weighted_regulariser": float(weighted_reg.detach().cpu()),
            "total_loss": float(total.detach().cpu()),
            "mean_displacement_norm": float(displacement.norm(dim=1).mean().detach().cpu()),
            "maximum_displacement_norm": float(displacement.norm(dim=1).max().detach().cpu()),
            "velocity_norm": float(velocity_magnitude_2d(coeffs).detach().cpu()),
            "minimum_jacobian": float(np.min(jac_np)),
            "jacobian_p01": float(np.quantile(jac_np, 0.01)),
            "jacobian_p05": float(np.quantile(jac_np, 0.05)),
            "jacobian_p50": float(np.quantile(jac_np, 0.50)),
            "nonpositive_jacobian_proportion": float(np.mean(jac_np <= float(config["jacobian_nonpositive_tol"]))),
            "jacobian_below_0_25_proportion": float(np.mean(jac_np < 0.25)),
            "jacobian_below_0_5_proportion": float(np.mean(jac_np < 0.5)),
            "iterations": len(history),
            "converged": bool(converged),
            "runtime_seconds": runtime,
            "failure_reason": failure_reason,
            "final_airfoil_valid": bool(validity["valid"]),
            **{f"validity_{key}": value for key, value in validity.items() if key != "valid"},
        },
    }


def save_registration_figure(
    source: np.ndarray,
    target: np.ndarray,
    deformed: np.ndarray,
    path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_shape(ax, source, color="tab:gray", linewidth=1.4, label="source")
    plot_shape(ax, target, color="tab:blue", linewidth=1.6, label="target")
    plot_shape(ax, deformed, color="tab:red", linewidth=1.4, linestyle="--", label="registered")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close(fig)


def save_loss_curve(history: list[dict], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label in [
        ("total_loss", "total"),
        ("data_attachment", "attachment"),
        ("weighted_regulariser", "weighted regulariser"),
    ]:
        ax.plot([row["iteration"] for row in history], [row[key] for row in history], label=label)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close(fig)
