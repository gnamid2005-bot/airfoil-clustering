import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh

from data import (
    compute_face_curvature_scores,
    extract_tar_archive,
    find_mesh_file,
    normalize_points,
)
from losses import chamfer_distance, dirichlet_smoothness


LAMBDA_SCHEDULE = [1e-1, 1e-2, 1e-3, 1e-4]
ITERATIONS_PER_STAGE = 200
DEFAULT_POINT_COUNTS = [1000, 3000, 5000, 10000]
DEFAULT_CURVATURE_STRENGTHS = [1.0, 2.0, 4.0, 8.0]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_ROOT = Path("outputs/choice1_experiments")
PLOT_LIMITS = ((-0.85, 0.85), (-0.85, 0.85), (-0.7, 0.55))
PLOT_ELEVATION = 20.0
PLOT_AZIMUTH = -60.0


def write_json(path: Path, data: dict) -> None:
    """Write JSON through a temporary file to reduce partial-result risk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(data, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    return result.stdout.strip()


def run_directory(root: Path, point_count: int, seed: int, method: str, strength: float | None) -> Path:
    if method == "uniform":
        leaf = "uniform"
    else:
        leaf = f"curvature_strength_{format_strength(strength)}"

    return root / f"point_count_{point_count}" / f"seed_{seed}" / leaf


def format_strength(strength: float | None) -> str:
    if strength is None:
        return "none"
    if float(strength).is_integer():
        return str(int(strength))
    return str(strength).replace(".", "p")


def load_bunny_mesh(tar_path: str = "data/bunny.tar") -> trimesh.Trimesh:
    extract_dir = "data/bunny_extracted"
    extract_tar_archive(tar_path, extract_dir)
    mesh_path = find_mesh_file(extract_dir)
    mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Loaded object is not a Trimesh: {type(mesh)}")
    return mesh


def deterministic_subset(points: torch.Tensor, max_points: int, seed: int) -> torch.Tensor:
    points_cpu = points.detach().cpu()
    if points_cpu.shape[0] <= max_points:
        return points_cpu

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randperm(points_cpu.shape[0], generator=generator)[:max_points]
    return points_cpu[indices]


def set_3d_axes(ax, limits=PLOT_LIMITS, elev=PLOT_ELEVATION, azim=PLOT_AZIMUTH) -> None:
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_box_aspect(
        (
            limits[0][1] - limits[0][0],
            limits[1][1] - limits[1][0],
            limits[2][1] - limits[2][0],
        )
    )
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def save_controlled_3d(
    point_sets: list[tuple[str, torch.Tensor, str]],
    save_path: Path,
    title: str,
    seed: int,
    max_points: int = 5000,
    marker_size: float = 2.0,
    limits=PLOT_LIMITS,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    for offset, (label, points, color) in enumerate(point_sets):
        sampled = deterministic_subset(points, max_points, seed + offset)
        ax.scatter(
            sampled[:, 0],
            sampled[:, 1],
            sampled[:, 2],
            s=marker_size,
            label=label,
            alpha=0.65,
            color=color,
        )

    set_3d_axes(ax, limits=limits)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def save_target_side_by_side_3d(
    left: tuple[str, torch.Tensor, str],
    right: tuple[str, torch.Tensor, str],
    save_path: Path,
    title: str,
    seed: int,
    max_points: int = 5000,
) -> None:
    """Save two target clouds in matched 3D panels."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 6))

    for index, (label, points, color) in enumerate([left, right], start=1):
        ax = fig.add_subplot(1, 2, index, projection="3d")
        sampled = deterministic_subset(points, max_points, seed + index)
        ax.scatter(
            sampled[:, 0],
            sampled[:, 1],
            sampled[:, 2],
            s=2,
            label=label,
            alpha=0.65,
            color=color,
        )
        set_3d_axes(ax)
        ax.set_title(label)
        ax.legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def save_projection(
    point_sets: list[tuple[str, torch.Tensor, str]],
    save_path: Path,
    title: str,
    axes: tuple[int, int],
    seed: int,
    max_points: int = 5000,
    marker_size: float = 2.0,
) -> None:
    labels = ["x", "y", "z"]
    axis_limits = PLOT_LIMITS
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))

    for offset, (label, points, color) in enumerate(point_sets):
        sampled = deterministic_subset(points, max_points, seed + offset)
        ax.scatter(
            sampled[:, axes[0]],
            sampled[:, axes[1]],
            s=marker_size,
            label=label,
            alpha=0.65,
            color=color,
        )

    ax.set_xlim(*axis_limits[axes[0]])
    ax.set_ylim(*axis_limits[axes[1]])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(labels[axes[0]])
    ax.set_ylabel(labels[axes[1]])
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def save_closeup_projection(
    point_sets: list[tuple[str, torch.Tensor, str]],
    save_path: Path,
    title: str,
    xlim: tuple[float, float],
    zlim: tuple[float, float],
    seed: int,
    max_points: int = 5000,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    for offset, (label, points, color) in enumerate(point_sets):
        sampled = deterministic_subset(points, max_points, seed + offset)
        ax.scatter(
            sampled[:, 0],
            sampled[:, 2],
            s=2,
            label=label,
            alpha=0.65,
            color=color,
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def save_controlled_run_figures(run_dir: Path, config: dict, checkpoint: dict) -> None:
    figures = run_dir / "controlled_figures"
    method_label = config["sampling_method"]
    if method_label == "curvature":
        method_label = f"curvature strength={config['curvature_strength']}"
    common_title = (
        f"{method_label}, n={config['point_count']}, seed={config['seed']}"
    )
    source = checkpoint["source"]
    target = checkpoint["target"]
    final_deformed = checkpoint["final_deformed"]

    save_controlled_3d(
        [("target", target, "tab:orange")],
        figures / "target_only_3d.png",
        f"Target only: {common_title}",
        config["seed"],
    )
    save_controlled_3d(
        [("source", source, "tab:blue"), ("target", target, "tab:orange")],
        figures / "initial_overlay_3d.png",
        f"Initial overlay: {common_title}",
        config["seed"],
    )
    save_controlled_3d(
        [("deformed source", final_deformed, "tab:blue"), ("target", target, "tab:orange")],
        figures / "final_overlay_3d.png",
        f"Final overlay: {common_title}",
        config["seed"],
    )
    save_controlled_3d(
        [("deformed source", final_deformed, "tab:blue")],
        figures / "final_source_only_3d.png",
        f"Final source only: {common_title}",
        config["seed"],
    )

    for name, axes in [("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2))]:
        save_projection(
            [("deformed source", final_deformed, "tab:blue"), ("target", target, "tab:orange")],
            figures / f"final_overlay_{name}.png",
            f"Final {name.upper()} projection: {common_title}",
            axes,
            config["seed"],
        )

    closeups = {
        "ears_xz": ((-0.55, 0.05), (0.05, 0.55)),
        "face_nose_xz": ((-0.65, -0.1), (-0.25, 0.35)),
        "feet_xz": ((-0.45, 0.35), (-0.65, -0.2)),
        "tail_xz": ((0.35, 0.85), (-0.25, 0.25)),
    }
    for name, (xlim, zlim) in closeups.items():
        save_closeup_projection(
            [("deformed source", final_deformed, "tab:blue"), ("target", target, "tab:orange")],
            figures / f"closeup_{name}.png",
            f"Close-up {name}: {common_title}",
            xlim,
            zlim,
            config["seed"],
        )


def chamfer_components(source: torch.Tensor, target: torch.Tensor) -> dict:
    distances = torch.cdist(source, target, p=2) ** 2
    source_to_target = distances.min(dim=1).values.mean()
    target_to_source = distances.min(dim=0).values.mean()
    return {
        "source_to_target": source_to_target.item(),
        "target_to_source": target_to_source.item(),
        "symmetric": (source_to_target + target_to_source).item(),
    }


def nearest_neighbour_stats(points: torch.Tensor) -> dict:
    points = points.detach().cpu()
    distances = torch.cdist(points, points, p=2)
    distances.fill_diagonal_(float("inf"))
    nearest = distances.min(dim=1).values
    mean = nearest.mean().item()
    std = nearest.std(unbiased=False).item()
    return {
        "mean": mean,
        "std": std,
        "min": nearest.min().item(),
        "max": nearest.max().item(),
        "median": nearest.median().item(),
        "coefficient_of_variation": std / mean if mean > 0 else math.nan,
    }


def voxel_occupancy(points: torch.Tensor, resolution: int) -> dict:
    points = points.detach().cpu()
    lows = torch.tensor([axis[0] for axis in PLOT_LIMITS])
    highs = torch.tensor([axis[1] for axis in PLOT_LIMITS])
    idx = ((points - lows) / (highs - lows) * resolution).floor().long()
    idx = torch.maximum(idx, torch.zeros_like(idx))
    idx = torch.minimum(idx, torch.full_like(idx, resolution - 1))
    linear = (
        idx[:, 0] * resolution * resolution
        + idx[:, 1] * resolution
        + idx[:, 2]
    )
    counts = torch.bincount(linear, minlength=resolution**3).float()
    occupied = counts[counts > 0]
    return {
        "resolution": resolution,
        "occupied_voxels": int(occupied.numel()),
        "occupancy_proportion": occupied.numel() / float(resolution**3),
        "max_points_per_occupied_voxel": int(occupied.max().item()) if occupied.numel() else 0,
        "mean_points_per_occupied_voxel": occupied.mean().item() if occupied.numel() else 0.0,
    }


def tensor_quantiles(values: torch.Tensor, quantiles: Iterable[float]) -> dict:
    return {
        str(q): torch.quantile(values, q).item()
        for q in quantiles
    }


def curvature_distribution(
    target_metadata: dict | None,
    high_curvature_quantile: float = 0.75,
) -> dict:
    if not target_metadata:
        return {}

    face_indices = target_metadata.get("sampled_face_indices")
    face_scores = target_metadata.get("face_curvature_scores")
    if face_indices is None or face_scores is None:
        return {}

    sampled_scores = face_scores[face_indices].float()
    threshold = torch.quantile(face_scores.float(), high_curvature_quantile)
    high = sampled_scores >= threshold
    return {
        "high_curvature_threshold_quantile": high_curvature_quantile,
        "high_curvature_threshold": threshold.item(),
        "high_curvature_point_count": int(high.sum().item()),
        "high_curvature_point_proportion": high.float().mean().item(),
        "sampled_face_curvature_mean": sampled_scores.mean().item(),
        "sampled_face_curvature_median": sampled_scores.median().item(),
        "sampled_face_curvature_std": sampled_scores.std(unbiased=False).item(),
        "sampled_face_curvature_quantiles": tensor_quantiles(
            sampled_scores,
            [0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
        ),
    }


def fixed_reference_points(num_points: int = 20000, seed: int = 12345) -> torch.Tensor:
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        mesh = load_bunny_mesh()
        points_np, _ = trimesh.sample.sample_surface(mesh, num_points)
    finally:
        np.random.set_state(rng_state)

    points = torch.tensor(points_np, dtype=torch.float32)
    return normalize_points(points, scale=0.8)


def distance_summary(a: torch.Tensor, b: torch.Tensor) -> dict:
    a = a.detach().cpu()
    b = b.detach().cpu()
    chunks = []
    for start in range(0, a.shape[0], 1000):
        distances = torch.cdist(a[start:start + 1000], b, p=2)
        chunks.append(distances.min(dim=1).values)
    nearest = torch.cat(chunks)
    return {
        "mean": nearest.mean().item(),
        "median": nearest.median().item(),
        "p95": torch.quantile(nearest, 0.95).item(),
        "max": nearest.max().item(),
    }


def region_masks(points: torch.Tensor) -> dict[str, torch.Tensor]:
    x = points[:, 0]
    z = points[:, 2]
    return {
        "ears_approx": (x < 0.05) & (z > 0.15),
        "head_face_approx": (x < -0.25) & (z > -0.25) & (z < 0.35),
        "torso_approx": (x >= -0.2) & (x <= 0.55) & (z >= -0.25) & (z <= 0.25),
        "legs_feet_approx": (z < -0.25),
        "tail_approx": (x > 0.45) & (z > -0.2) & (z < 0.2),
    }


def regional_metrics(
    target: torch.Tensor,
    final_deformed: torch.Tensor,
    reference: torch.Tensor,
) -> dict:
    metrics = {}
    target_masks = region_masks(target)
    reference_masks = region_masks(reference)

    for name, mask in target_masks.items():
        count = int(mask.sum().item())
        region = {
            "target_point_count": count,
            "target_point_proportion": count / float(target.shape[0]),
        }

        if count > 0:
            target_region = target[mask]
            region["local_target_to_final_mean_distance"] = distance_summary(
                target_region,
                final_deformed,
            )["mean"]

        ref_mask = reference_masks[name]
        if int(ref_mask.sum().item()) > 0 and count > 0:
            region["local_reference_to_target_mean_distance"] = distance_summary(
                reference[ref_mask],
                target[mask],
            )["mean"]

        metrics[name] = region

    return metrics


def compute_run_metrics(config: dict, checkpoint: dict) -> dict:
    source = checkpoint["source"]
    target = checkpoint["target"]
    final_deformed = checkpoint["final_deformed"]
    coeffs = checkpoint["coeffs"]
    reference = fixed_reference_points()
    chamfer = chamfer_components(final_deformed, target)
    data_history = checkpoint["data_attachment_history"]
    best_data = min(data_history)

    metrics = {
        "status": "completed",
        "point_count": config["point_count"],
        "source_point_count": int(source.shape[0]),
        "target_point_count": int(target.shape[0]),
        "seed": config["seed"],
        "sampling_method": config["sampling_method"],
        "curvature_strength": config.get("curvature_strength"),
        "registration": {
            "final_symmetric_chamfer": chamfer["symmetric"],
            "final_source_to_target_chamfer_component": chamfer["source_to_target"],
            "final_target_to_source_chamfer_component": chamfer["target_to_source"],
            "final_total_loss": checkpoint["total_loss_history"][-1],
            "final_data_attachment": checkpoint["data_attachment_history"][-1],
            "final_raw_regulariser": checkpoint["regulariser_history"][-1],
            "final_weighted_regulariser": checkpoint["weighted_regulariser_history"][-1],
            "best_data_attachment": best_data,
            "best_data_attachment_iteration": data_history.index(best_data),
            "runtime_seconds": checkpoint.get("runtime_seconds"),
            "recomputed_final_regulariser": dirichlet_smoothness(coeffs).item(),
        },
        "sampling_distribution": {
            "nearest_neighbour_distance": nearest_neighbour_stats(target),
            "voxel_occupancy": {
                "resolution_16": voxel_occupancy(target, 16),
                "resolution_32": voxel_occupancy(target, 32),
            },
            "curvature": curvature_distribution(
                checkpoint.get("target_metadata")
            ),
        },
        "mesh_coverage": {
            "reference_to_target_distance": distance_summary(reference, target),
            "target_to_reference_distance": distance_summary(target, reference),
        },
        "regional": regional_metrics(target, final_deformed, reference),
        "histories": {
            "total_loss": checkpoint["total_loss_history"],
            "data_attachment": checkpoint["data_attachment_history"],
            "raw_regulariser": checkpoint["regulariser_history"],
            "weighted_regulariser": checkpoint["weighted_regulariser_history"],
            "lambda": checkpoint["lambda_history"],
        },
    }

    return metrics


def flatten_dict(data: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_dict(value, name))
        elif isinstance(value, list):
            continue
        else:
            flat[name] = value
    return flat


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
