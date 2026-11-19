import argparse
import contextlib
import io
import math
import resource
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import torch

from choice1_utils import (
    LAMBDA_SCHEDULE,
    PLOT_LIMITS,
    chamfer_components,
    flatten_dict,
    get_git_commit,
    read_json,
    save_closeup_projection,
    save_controlled_3d,
    save_projection,
    set_3d_axes,
    write_json,
)
from losses import dirichlet_smoothness
from svf import evaluate_velocity_field, integrate_svf
from train import build_source_points, build_target_points, set_random_seed, train


matplotlib.use("Agg")

DEFAULT_ROOT = Path("outputs/choice2_experiments")
SMOKE_GRID_SIZES = [8, 16]
PILOT_GRID_SIZES = [8, 10, 12, 16, 24]
PILOT_SEEDS = [0, 1]
POINT_COUNT = 5000
CURVATURE_STRENGTH = 4.0


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def mode_defaults(mode: str) -> dict:
    if mode == "smoke":
        return {
            "grid_sizes": SMOKE_GRID_SIZES,
            "seeds": [0],
            "num_iters": 2,
        }
    return {
        "grid_sizes": PILOT_GRID_SIZES,
        "seeds": PILOT_SEEDS,
        "num_iters": 800,
    }


def run_dir(root: Path, grid_size: int, seed: int) -> Path:
    return root / f"grid_size_{grid_size}" / f"seed_{seed}"


def build_run_configs(args) -> list[dict]:
    defaults = mode_defaults(args.mode)
    grid_sizes = parse_int_list(args.grid_sizes) if args.grid_sizes else defaults["grid_sizes"]
    seeds = parse_int_list(args.seeds) if args.seeds else defaults["seeds"]
    num_iters = args.num_iters if args.num_iters is not None else defaults["num_iters"]

    if any(size <= 3 for size in grid_sizes):
        raise ValueError("Grid sizes must be greater than 3 for cubic B-spline support.")
    if num_iters <= 0:
        raise ValueError("--num-iters must be positive.")

    configs = []
    root = Path(args.output_root)
    for grid_size in grid_sizes:
        for seed in seeds:
            directory = run_dir(root, grid_size, seed)
            control_points = grid_size**3
            scalar_coefficients = 3 * control_points
            configs.append(
                {
                    "study": "choice2_grid_resolution",
                    "grid_size": grid_size,
                    "control_points_per_dimension": grid_size,
                    "number_of_control_points": control_points,
                    "declared_scalar_velocity_coefficients": scalar_coefficients,
                    "seed": seed,
                    "source": "cube",
                    "target": "bunny",
                    "point_count": POINT_COUNT,
                    "source_point_count": POINT_COUNT,
                    "target_point_count": POINT_COUNT,
                    "target_sampling": "curvature",
                    "curvature_strength": CURVATURE_STRENGTH,
                    "num_steps": args.num_steps,
                    "num_iters": num_iters,
                    "lr": args.lr,
                    "optimizer": "Adam",
                    "lambda_schedule": LAMBDA_SCHEDULE,
                    "iterations_per_stage": 200,
                    "print_every": args.print_every,
                    "save_every": args.save_every,
                    "device": args.device,
                    "run_dir": str(directory),
                    "figure_dir": str(directory / "figures"),
                    "checkpoint_dir": str(directory / "checkpoints"),
                    "git_commit": get_git_commit(),
                }
            )

    if args.max_runs is not None:
        configs = configs[: args.max_runs]

    return configs


def is_completed(directory: Path) -> bool:
    metrics_path = directory / "metrics.json"
    config_path = directory / "config.json"
    if not metrics_path.exists() or not config_path.exists():
        return False
    try:
        metrics = read_json(metrics_path)
        config = read_json(config_path)
    except Exception:
        return False
    return metrics.get("status") == "completed" and config.get("status") == "completed"


def train_args_from_config(config: dict) -> SimpleNamespace:
    return SimpleNamespace(
        source=config["source"],
        target=config["target"],
        num_points=config["point_count"],
        grid_size=config["grid_size"],
        num_steps=config["num_steps"],
        num_iters=config["num_iters"],
        lr=config["lr"],
        print_every=config["print_every"],
        save_every=config["save_every"],
        device=config["device"],
        figure_dir=config["figure_dir"],
        checkpoint_dir=config["checkpoint_dir"],
        target_sampling=config["target_sampling"],
        curvature_strength=config["curvature_strength"],
        seed=config["seed"],
    )


def controlled_clouds(config: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
    set_random_seed(config["seed"])
    source = build_source_points(
        source_type=config["source"],
        num_points=config["point_count"],
        device=config["device"],
    ).detach().cpu()
    target, metadata = build_target_points(
        target_type=config["target"],
        num_points=config["point_count"],
        device=config["device"],
        target_sampling=config["target_sampling"],
        curvature_strength=config["curvature_strength"],
        return_metadata=True,
    )
    return source, target.detach().cpu(), metadata


def peak_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux commonly reports KiB.
    if usage > 10_000_000:
        return usage / (1024 * 1024)
    return usage / 1024


def coefficient_parameter_count(coeffs: torch.Tensor) -> int:
    return int(coeffs.numel())


def gradient_magnitude(coeffs: torch.Tensor) -> dict:
    dx = coeffs[1:, :, :, :] - coeffs[:-1, :, :, :]
    dy = coeffs[:, 1:, :, :] - coeffs[:, :-1, :, :]
    dz = coeffs[:, :, 1:, :] - coeffs[:, :, :-1, :]
    magnitudes = torch.cat(
        [
            dx.norm(dim=-1).flatten(),
            dy.norm(dim=-1).flatten(),
            dz.norm(dim=-1).flatten(),
        ]
    )
    return {
        "mean": magnitudes.mean().item(),
        "max": magnitudes.max().item(),
    }


def velocity_magnitude_stats(coeffs: torch.Tensor, sample_resolution: int = 12) -> dict:
    points = regular_grid_points(sample_resolution)
    with torch.no_grad():
        velocities = evaluate_velocity_field(points, coeffs)
    magnitudes = velocities.norm(dim=1)
    return {
        "sample_resolution": sample_resolution,
        "mean_velocity_magnitude": magnitudes.mean().item(),
        "max_velocity_magnitude": magnitudes.max().item(),
    }


def regular_grid_points(resolution: int) -> torch.Tensor:
    values = torch.linspace(-0.8, 0.8, resolution)
    xx, yy, zz = torch.meshgrid(values, values, values, indexing="ij")
    return torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)


def jacobian_diagnostics(coeffs: torch.Tensor, num_steps: int, resolution: int = 7) -> dict:
    """Numerically approximate Jacobian determinants of the final transform."""
    values = torch.linspace(-0.75, 0.75, resolution)
    h = (values[1] - values[0]).item()
    xx, yy, zz = torch.meshgrid(values, values, values, indexing="ij")
    points = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

    with torch.no_grad():
        transformed = integrate_svf(points, coeffs, num_steps=num_steps)

    field = transformed.reshape(resolution, resolution, resolution, 3)
    determinants = []
    for i in range(1, resolution - 1):
        for j in range(1, resolution - 1):
            for k in range(1, resolution - 1):
                dphi_dx = (field[i + 1, j, k] - field[i - 1, j, k]) / (2 * h)
                dphi_dy = (field[i, j + 1, k] - field[i, j - 1, k]) / (2 * h)
                dphi_dz = (field[i, j, k + 1] - field[i, j, k - 1]) / (2 * h)
                jacobian = torch.stack([dphi_dx, dphi_dy, dphi_dz], dim=1)
                determinants.append(torch.det(jacobian))

    det = torch.stack(determinants)
    return {
        "sample_resolution": resolution,
        "minimum": det.min().item(),
        "mean": det.mean().item(),
        "maximum": det.max().item(),
        "proportion_less_equal_zero": (det <= 0).float().mean().item(),
        "proportion_below_0_1": (det < 0.1).float().mean().item(),
    }


def compute_choice2_metrics(config: dict, checkpoint: dict, memory_before_mb: float) -> dict:
    coeffs = checkpoint["coeffs"]
    final_deformed = checkpoint["final_deformed"]
    target = checkpoint["target"]
    chamfer = chamfer_components(final_deformed, target)
    data_history = checkpoint["data_attachment_history"]
    best_data = min(data_history)
    velocity = velocity_magnitude_stats(coeffs)
    gradients = gradient_magnitude(coeffs)
    jacobian = jacobian_diagnostics(coeffs, config["num_steps"])
    actual_parameters = coefficient_parameter_count(coeffs)

    return {
        "status": "completed",
        "grid_size": config["grid_size"],
        "number_of_control_points": config["number_of_control_points"],
        "declared_scalar_velocity_coefficients": config["declared_scalar_velocity_coefficients"],
        "actual_learnable_parameter_count": actual_parameters,
        "parameter_count_matches_declaration": actual_parameters == config["declared_scalar_velocity_coefficients"],
        "seed": config["seed"],
        "point_count": config["point_count"],
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
            "time_per_iteration_seconds": checkpoint.get("runtime_seconds") / config["num_iters"],
            "recomputed_final_regulariser": dirichlet_smoothness(coeffs).item(),
        },
        "deformation_quality": {
            "mean_velocity_magnitude": velocity["mean_velocity_magnitude"],
            "maximum_velocity_magnitude": velocity["max_velocity_magnitude"],
            "mean_velocity_gradient_magnitude": gradients["mean"],
            "maximum_velocity_gradient_magnitude": gradients["max"],
            "jacobian_determinant": jacobian,
        },
        "memory": {
            "peak_memory_mb": peak_memory_mb(),
            "memory_before_run_mb": memory_before_mb,
        },
        "histories": {
            "total_loss": checkpoint["total_loss_history"],
            "data_attachment": checkpoint["data_attachment_history"],
            "raw_regulariser": checkpoint["regulariser_history"],
            "weighted_regulariser": checkpoint["weighted_regulariser_history"],
            "lambda": checkpoint["lambda_history"],
        },
    }


def save_velocity_visualisation(run_directory: Path, config: dict, coeffs: torch.Tensor) -> None:
    figures = run_directory / "controlled_figures"
    points = regular_grid_points(12)
    with torch.no_grad():
        velocities = evaluate_velocity_field(points, coeffs)
    magnitudes = velocities.norm(dim=1)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=magnitudes,
        s=8,
        cmap="viridis",
    )
    set_3d_axes(ax)
    ax.set_title(
        f"Velocity magnitude, grid={config['grid_size']}, seed={config['seed']}"
    )
    fig.colorbar(scatter, ax=ax, shrink=0.7)
    plt.tight_layout()
    plt.savefig(figures / "velocity_magnitude_3d.png", dpi=200)
    plt.close(fig)


def save_control_lattice_visualisation(run_directory: Path, config: dict, coeffs: torch.Tensor) -> None:
    figures = run_directory / "controlled_figures"
    grid_size = coeffs.shape[0]
    values = torch.linspace(-0.8, 0.8, grid_size)
    xx, yy, zz = torch.meshgrid(values, values, values, indexing="ij")
    points = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
    magnitudes = coeffs.norm(dim=-1).flatten()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=magnitudes,
        s=8,
        cmap="magma",
    )
    set_3d_axes(ax)
    ax.set_title(
        f"Control lattice magnitude, grid={grid_size}, seed={config['seed']}"
    )
    fig.colorbar(scatter, ax=ax, shrink=0.7)
    plt.tight_layout()
    plt.savefig(figures / "control_lattice_magnitude_3d.png", dpi=200)
    plt.close(fig)


def save_choice2_figures(run_directory: Path, config: dict, checkpoint: dict) -> None:
    figures = run_directory / "controlled_figures"
    source = checkpoint["source"]
    target = checkpoint["target"]
    final_deformed = checkpoint["final_deformed"]
    title_suffix = f"grid={config['grid_size']}, n=5000, seed={config['seed']}"

    save_controlled_3d(
        [("deformed source", final_deformed, "tab:blue"), ("target", target, "tab:orange")],
        figures / "final_overlay_3d.png",
        f"Final overlay: {title_suffix}",
        config["seed"],
    )
    save_controlled_3d(
        [("deformed source", final_deformed, "tab:blue")],
        figures / "final_source_only_3d.png",
        f"Final source only: {title_suffix}",
        config["seed"],
    )

    for name, axes in [("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2))]:
        save_projection(
            [("deformed source", final_deformed, "tab:blue"), ("target", target, "tab:orange")],
            figures / f"final_overlay_{name}.png",
            f"Final {name.upper()} projection: {title_suffix}",
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
            f"Close-up {name}: {title_suffix}",
            xlim,
            zlim,
            config["seed"],
        )

    save_velocity_visualisation(run_directory, config, checkpoint["coeffs"])
    save_control_lattice_visualisation(run_directory, config, checkpoint["coeffs"])


def load_checkpoint(config: dict):
    path = Path(config["checkpoint_dir"]) / "cube_to_bunny.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def verify_matched_clouds(configs: list[dict]) -> None:
    comparisons = []
    by_seed: dict[int, list[dict]] = {}
    for config in configs:
        if is_completed(Path(config["run_dir"])):
            by_seed.setdefault(config["seed"], []).append(config)

    for seed, group in sorted(by_seed.items()):
        checkpoints = [(config, load_checkpoint(config)) for config in group]
        checkpoints = [(config, checkpoint) for config, checkpoint in checkpoints if checkpoint is not None]
        if not checkpoints:
            continue
        reference_config, reference_checkpoint = checkpoints[0]
        for config, checkpoint in checkpoints[1:]:
            comparisons.append(
                {
                    "seed": seed,
                    "reference_grid_size": reference_config["grid_size"],
                    "comparison_grid_size": config["grid_size"],
                    "source_tensors_exactly_identical": bool(
                        torch.equal(reference_checkpoint["source"], checkpoint["source"])
                    ),
                    "target_tensors_exactly_identical": bool(
                        torch.equal(reference_checkpoint["target"], checkpoint["target"])
                    ),
                    "only_grid_size_differs": only_grid_size_differs(reference_config, config),
                }
            )

    if configs:
        root = Path(configs[0]["run_dir"]).parents[1]
        write_json(root / "matched_grid_comparisons.json", {"comparisons": comparisons})


def only_grid_size_differs(left: dict, right: dict) -> bool:
    allowed = {
        "grid_size",
        "control_points_per_dimension",
        "number_of_control_points",
        "declared_scalar_velocity_coefficients",
        "run_dir",
        "figure_dir",
        "checkpoint_dir",
    }
    keys = set(left) | set(right)
    return all(left.get(key) == right.get(key) for key in keys if key not in allowed)


def save_seed_side_by_side(configs: list[dict]) -> None:
    by_seed: dict[int, list[dict]] = {}
    for config in configs:
        if is_completed(Path(config["run_dir"])):
            by_seed.setdefault(config["seed"], []).append(config)

    for seed, group in sorted(by_seed.items()):
        group = sorted(group, key=lambda config: config["grid_size"])
        checkpoints = [(config, load_checkpoint(config)) for config in group]
        checkpoints = [(config, checkpoint) for config, checkpoint in checkpoints if checkpoint is not None]
        if not checkpoints:
            continue

        root = Path(checkpoints[0][0]["run_dir"]).parents[1]
        figures = root / "side_by_side_figures"
        figures.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(20, 4))
        for index, (config, checkpoint) in enumerate(checkpoints, start=1):
            ax = fig.add_subplot(1, len(checkpoints), index, projection="3d")
            final_deformed = checkpoint["final_deformed"]
            target = checkpoint["target"]
            ax.scatter(
                final_deformed[:, 0],
                final_deformed[:, 1],
                final_deformed[:, 2],
                s=1.2,
                color="tab:blue",
                alpha=0.55,
                label="source",
            )
            ax.scatter(
                target[:, 0],
                target[:, 1],
                target[:, 2],
                s=1.2,
                color="tab:orange",
                alpha=0.55,
                label="target",
            )
            set_3d_axes(ax)
            ax.set_title(f"grid {config['grid_size']}")
        fig.suptitle(f"Final overlays by grid size, seed={seed}")
        plt.tight_layout()
        plt.savefig(figures / f"seed_{seed}_final_overlay_all_grid_sizes.png", dpi=200)
        plt.close(fig)


def run_one(config: dict, args) -> dict:
    directory = Path(config["run_dir"])
    if is_completed(directory) and not args.force:
        print(f"skip completed: {directory}", flush=True)
        return {"status": "skipped", "run_dir": str(directory)}

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "figures").mkdir(exist_ok=True)
    (directory / "checkpoints").mkdir(exist_ok=True)
    write_json(directory / "config.json", {**config, "status": "running"})

    source, target, target_metadata = controlled_clouds(config)
    train_args = train_args_from_config(config)
    memory_before = peak_memory_mb()
    start = time.perf_counter()

    try:
        with (directory / "run.log").open("w", encoding="utf-8") as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                checkpoint = train(
                    train_args,
                    source_override=source,
                    target_override=target,
                    target_metadata=target_metadata,
                    save_standard_figures=True,
                )

        runtime = time.perf_counter() - start
        metrics = compute_choice2_metrics(config, checkpoint, memory_before)
        metrics["runtime_seconds_wall_clock"] = runtime
        metrics["source_tensor_verified_identical_to_control"] = bool(
            torch.equal(source, checkpoint["source"])
        )
        metrics["target_tensor_verified_identical_to_control"] = bool(
            torch.equal(target, checkpoint["target"])
        )
        save_choice2_figures(directory, config, checkpoint)
        write_json(directory / "metrics.json", metrics)
        write_json(directory / "config.json", {**config, "status": "completed"})
        print(f"completed: {directory} ({runtime:.1f}s)", flush=True)
        return {"status": "completed", "run_dir": str(directory), "runtime": runtime}
    except Exception as exc:
        with (directory / "run.log").open("a", encoding="utf-8") as log_file:
            log_file.write("\nFAILED\n")
            log_file.write(traceback.format_exc())
        failure = {
            **config,
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(directory / "metrics.json", failure)
        write_json(directory / "config.json", {**config, "status": "failed"})
        print(f"failed: {directory}: {exc}", flush=True)
        return {"status": "failed", "run_dir": str(directory), "error": str(exc)}


def estimate_output_size_mb(num_runs: int) -> float:
    return 7.0 * num_runs


def print_plan(configs: list[dict], args) -> None:
    print(f"mode: {args.mode}", flush=True)
    print(f"output root: {args.output_root}", flush=True)
    print(f"planned runs: {len(configs)}", flush=True)
    print(f"approx output size: {estimate_output_size_mb(len(configs)):.0f} MB", flush=True)
    for config in configs:
        print(
            f"- grid={config['grid_size']} seed={config['seed']} "
            f"iters={config['num_iters']} params={config['declared_scalar_velocity_coefficients']} "
            f"-> {config['run_dir']}",
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run controlled Choice 2 B-spline grid-size experiments."
    )
    parser.add_argument("--mode", choices=["smoke", "pilot"], default="smoke")
    parser.add_argument("--output-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--grid-sizes", default=None, help="Comma-separated grid sizes.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds.")
    parser.add_argument("--num-iters", type=int, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_runs is not None and args.max_runs <= 0:
        raise ValueError("--max-runs must be positive.")

    configs = build_run_configs(args)
    print_plan(configs, args)
    if args.dry_run:
        return

    results = []
    for index, config in enumerate(configs, start=1):
        print(f"[{index}/{len(configs)}] running {config['run_dir']}", flush=True)
        results.append(run_one(config, args))

    completed_configs = [
        config for config in configs if is_completed(Path(config["run_dir"]))
    ]
    verify_matched_clouds(completed_configs)
    save_seed_side_by_side(completed_configs)

    completed = sum(1 for result in results if result["status"] == "completed")
    skipped = sum(1 for result in results if result["status"] == "skipped")
    failed = sum(1 for result in results if result["status"] == "failed")
    print(f"done: completed={completed}, skipped={skipped}, failed={failed}", flush=True)


if __name__ == "__main__":
    main()

