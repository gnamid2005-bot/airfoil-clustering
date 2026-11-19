import argparse
import contextlib
import copy
import io
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import torch

from choice1_utils import (
    DEFAULT_CURVATURE_STRENGTHS,
    DEFAULT_POINT_COUNTS,
    DEFAULT_ROOT,
    DEFAULT_SEEDS,
    LAMBDA_SCHEDULE,
    compute_run_metrics,
    format_strength,
    get_git_commit,
    read_json,
    run_directory,
    save_controlled_3d,
    save_controlled_run_figures,
    save_target_side_by_side_3d,
    write_json,
)
from train import build_source_points, set_random_seed, train


matplotlib.use("Agg")


def parse_number_list(value: str, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def mode_defaults(mode: str) -> dict:
    if mode == "smoke":
        return {
            "point_counts": [1000],
            "curvature_strengths": [4.0],
            "seeds": [0],
            "num_iters": 2,
        }

    if mode == "pilot":
        return {
            "point_counts": [1000, 5000],
            "curvature_strengths": [2.0, 4.0],
            "seeds": [0, 1],
            "num_iters": 800,
        }

    return {
        "point_counts": DEFAULT_POINT_COUNTS,
        "curvature_strengths": DEFAULT_CURVATURE_STRENGTHS,
        "seeds": DEFAULT_SEEDS,
        "num_iters": 800,
    }


def build_run_configs(args) -> list[dict]:
    defaults = mode_defaults(args.mode)
    point_counts = (
        parse_number_list(args.point_counts, int)
        if args.point_counts
        else defaults["point_counts"]
    )
    strengths = (
        parse_number_list(args.curvature_strengths, float)
        if args.curvature_strengths
        else defaults["curvature_strengths"]
    )
    seeds = (
        parse_number_list(args.seeds, int)
        if args.seeds
        else defaults["seeds"]
    )
    num_iters = args.num_iters if args.num_iters is not None else defaults["num_iters"]

    if any(count <= 0 for count in point_counts):
        raise ValueError("All point counts must be positive.")
    if any(strength <= 0 for strength in strengths):
        raise ValueError("Curvature strengths must be positive.")
    if num_iters <= 0:
        raise ValueError("--num-iters must be positive.")

    configs = []
    for point_count in point_counts:
        for seed in seeds:
            configs.append(
                one_config(
                    args,
                    point_count=point_count,
                    seed=seed,
                    sampling_method="uniform",
                    curvature_strength=None,
                    num_iters=num_iters,
                )
            )
            for strength in strengths:
                configs.append(
                    one_config(
                        args,
                        point_count=point_count,
                        seed=seed,
                        sampling_method="curvature",
                        curvature_strength=strength,
                        num_iters=num_iters,
                    )
                )

    if args.max_runs is not None:
        configs = configs[: args.max_runs]

    return configs


def one_config(
    args,
    point_count: int,
    seed: int,
    sampling_method: str,
    curvature_strength: float | None,
    num_iters: int,
) -> dict:
    root = Path(args.output_root)
    run_dir = run_directory(
        root,
        point_count,
        seed,
        sampling_method,
        curvature_strength,
    )
    return {
        "study": "choice1_factorial",
        "point_count": point_count,
        "seed": seed,
        "sampling_method": sampling_method,
        "curvature_strength": curvature_strength,
        "source": "cube",
        "target": "bunny",
        "source_point_count": point_count,
        "target_point_count": point_count,
        "grid_size": args.grid_size,
        "num_steps": args.num_steps,
        "num_iters": num_iters,
        "lr": args.lr,
        "optimizer": "Adam",
        "lambda_schedule": LAMBDA_SCHEDULE,
        "iterations_per_stage": 200,
        "print_every": args.print_every,
        "save_every": args.save_every,
        "device": args.device,
        "run_dir": str(run_dir),
        "figure_dir": str(run_dir / "figures"),
        "checkpoint_dir": str(run_dir / "checkpoints"),
        "git_commit": get_git_commit(),
    }


def is_completed(run_dir: Path) -> bool:
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.json"
    if not metrics_path.exists() or not config_path.exists():
        return False
    try:
        metrics = read_json(metrics_path)
    except Exception:
        return False
    return metrics.get("status") == "completed"


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
        target_sampling=config["sampling_method"],
        curvature_strength=(
            config["curvature_strength"]
            if config["curvature_strength"] is not None
            else 4.0
        ),
        seed=config["seed"],
    )


def build_controlled_source(config: dict) -> torch.Tensor:
    set_random_seed(config["seed"])
    return build_source_points(
        source_type=config["source"],
        num_points=config["point_count"],
        device=config["device"],
    ).detach().cpu()


def run_one(config: dict, args) -> dict:
    run_dir = Path(config["run_dir"])
    if is_completed(run_dir) and not args.force:
        print(f"skip completed: {run_dir}")
        return {"status": "skipped", "run_dir": str(run_dir)}

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    write_json(run_dir / "config.json", {**config, "status": "running"})

    source = build_controlled_source(config)
    train_args = train_args_from_config(config)
    log_buffer = io.StringIO()
    start = time.perf_counter()

    try:
        with (run_dir / "run.log").open("w", encoding="utf-8") as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                checkpoint = train(
                    train_args,
                    source_override=source,
                    save_standard_figures=True,
                )

        runtime = time.perf_counter() - start
        metrics = compute_run_metrics(config, checkpoint)
        metrics["runtime_seconds_wall_clock"] = runtime
        metrics["source_tensor_verified_identical_to_control"] = bool(
            torch.equal(source, checkpoint["source"])
        )
        write_json(run_dir / "metrics.json", metrics)
        write_json(run_dir / "config.json", {**config, "status": "completed"})
        save_controlled_run_figures(run_dir, config, checkpoint)
        print(f"completed: {run_dir} ({runtime:.1f}s)")
        return {"status": "completed", "run_dir": str(run_dir), "runtime": runtime}
    except Exception as exc:
        log_buffer.write(traceback.format_exc())
        with (run_dir / "run.log").open("a", encoding="utf-8") as log_file:
            log_file.write("\nFAILED\n")
            log_file.write(log_buffer.getvalue())
        failure = {
            **config,
            "status": "failed",
            "error": str(exc),
            "traceback": log_buffer.getvalue(),
        }
        write_json(run_dir / "metrics.json", failure)
        write_json(run_dir / "config.json", {**config, "status": "failed"})
        print(f"failed: {run_dir}: {exc}")
        return {"status": "failed", "run_dir": str(run_dir), "error": str(exc)}


def create_matched_side_by_side(configs: list[dict]) -> None:
    comparisons = []
    by_key = {}
    for config in configs:
        key = (config["point_count"], config["seed"])
        by_key.setdefault(key, []).append(config)

    for (_, _), group in by_key.items():
        uniform = [cfg for cfg in group if cfg["sampling_method"] == "uniform"]
        if not uniform:
            continue
        uniform_ckpt = load_checkpoint(uniform[0])
        if uniform_ckpt is None:
            continue

        for cfg in group:
            if cfg["sampling_method"] != "curvature":
                continue
            curv_ckpt = load_checkpoint(cfg)
            if curv_ckpt is None:
                continue

            same_source = bool(
                torch.equal(
                    uniform_ckpt["source"],
                    curv_ckpt["source"],
                )
            )
            comparisons.append(
                {
                    "point_count": cfg["point_count"],
                    "seed": cfg["seed"],
                    "uniform_run_dir": uniform[0]["run_dir"],
                    "curvature_run_dir": cfg["run_dir"],
                    "curvature_strength": cfg["curvature_strength"],
                    "source_tensors_exactly_identical": same_source,
                    "matched_training_settings": matched_training_settings(
                        uniform[0],
                        cfg,
                    ),
                }
            )

            out_dir = Path(cfg["run_dir"]) / "controlled_figures"
            out_dir.mkdir(parents=True, exist_ok=True)
            title = (
                f"Target-only comparison, n={cfg['point_count']}, "
                f"seed={cfg['seed']}, strength={cfg['curvature_strength']}"
            )
            save_controlled_3d(
                [
                    ("uniform target", uniform_ckpt["target"], "tab:orange"),
                    ("curvature target", curv_ckpt["target"], "tab:blue"),
                ],
                out_dir / "target_overlay_uniform_vs_curvature_3d.png",
                title,
                cfg["seed"],
            )
            save_target_side_by_side_3d(
                ("uniform target", uniform_ckpt["target"], "tab:orange"),
                ("curvature target", curv_ckpt["target"], "tab:blue"),
                out_dir / "target_side_by_side_uniform_vs_curvature_3d.png",
                title,
                cfg["seed"],
            )

    if configs:
        root = Path(configs[0]["run_dir"]).parents[2]
        write_json(root / "matched_comparisons.json", {"comparisons": comparisons})


def matched_training_settings(left: dict, right: dict) -> bool:
    keys = [
        "point_count",
        "seed",
        "source",
        "target",
        "grid_size",
        "num_steps",
        "num_iters",
        "lr",
        "optimizer",
        "lambda_schedule",
        "iterations_per_stage",
    ]
    return all(left.get(key) == right.get(key) for key in keys)


def load_checkpoint(config: dict):
    path = Path(config["checkpoint_dir"]) / "cube_to_bunny.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def estimate_output_size_mb(num_runs: int) -> float:
    # A conservative rough estimate: checkpoint plus figures per run.
    return num_runs * 18.0


def print_plan(configs: list[dict], args) -> None:
    print(f"mode: {args.mode}")
    print(f"output root: {args.output_root}")
    print(f"planned runs: {len(configs)}")
    print(f"approx output size: {estimate_output_size_mb(len(configs)):.0f} MB")
    if args.mode == "full":
        print("Full mode is large; this command only runs if you explicitly invoked --mode full.")
    for config in configs:
        strength = config["curvature_strength"]
        label = config["sampling_method"]
        if strength is not None:
            label += f" strength={format_strength(strength)}"
        print(
            f"- n={config['point_count']} seed={config['seed']} "
            f"{label} iters={config['num_iters']} -> {config['run_dir']}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run controlled Choice 1 point-cloud sampling experiments."
    )
    parser.add_argument("--mode", choices=["smoke", "pilot", "full"], default="smoke")
    parser.add_argument("--output-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--point-counts", default=None, help="Comma-separated point counts.")
    parser.add_argument("--curvature-strengths", default=None, help="Comma-separated strengths.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds.")
    parser.add_argument("--num-iters", type=int, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run completed runs.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--grid-size", type=int, default=10)
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
        print(f"[{index}/{len(configs)}] running {config['run_dir']}")
        results.append(run_one(config, args))

    completed_configs = [
        config
        for config in configs
        if is_completed(Path(config["run_dir"]))
    ]
    create_matched_side_by_side(completed_configs)

    completed = sum(1 for result in results if result["status"] == "completed")
    skipped = sum(1 for result in results if result["status"] == "skipped")
    failed = sum(1 for result in results if result["status"] == "failed")
    print(f"done: completed={completed}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
