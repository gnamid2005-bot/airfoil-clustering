from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from airfoil_data import plot_shape
from airfoil_registration import (
    camber_thickness_from_shapes,
    gmm_posteriors_for_shapes,
    load_airfoil_dataset,
    reconstruct_gmm_templates,
    reconstruct_k1_template,
    register_airfoils,
    registration_bounds,
    save_csv,
    save_loss_curve,
    save_registration_figure,
    validate_airfoil_shape,
    write_json,
)


matplotlib.use("Agg")


DEFAULT_OUTPUT_DIR = Path("outputs/airfoil_registration_validation")
DEFAULT_GMM_DIR = Path("outputs/airfoil_gmm_selection")
DEFAULT_DATA_PATH = Path("data/airfoils/processed/airfoils_n100_seed0.npz")


def select_smoke_targets(dataset: dict, gmm_dir: Path, count: int = 8) -> list[int]:
    heldout = np.load(gmm_dir / "heldout_indices.npy")
    extreme_heldout = np.load(gmm_dir / "extreme_group_heldout_indices.npy")
    values = camber_thickness_from_shapes(dataset["shapes"])
    train = np.load(gmm_dir / "train_indices.npy")
    global_mean = values[train].mean(axis=0, keepdims=True)
    distances = np.sqrt(np.sum((values - global_mean) ** 2, axis=1))
    heldout_sorted = heldout[np.argsort(distances[heldout])]
    close = heldout_sorted[:2].tolist()
    mid_order = heldout[np.argsort(np.abs(distances[heldout] - np.median(distances[heldout])))]
    medium = mid_order[:2].tolist()
    far = heldout_sorted[-2:].tolist()
    extreme = extreme_heldout[:2].tolist()
    selected = []
    for index in close + medium + far + extreme:
        if int(index) not in selected:
            selected.append(int(index))
    for index in heldout_sorted:
        if len(selected) >= count:
            break
        if int(index) not in selected:
            selected.append(int(index))
    return selected[:count]


def save_template_plot(templates: list[tuple[str, np.ndarray]], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, shape in templates:
        plot_shape(ax, shape, linewidth=1.8, label=label)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke validation for 2D airfoil SVF registration.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--gmm-dir", default=str(DEFAULT_GMM_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--smoke-target-count", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=15)
    parser.add_argument("--num-iters", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--lambda-reg", type=float, default=1e-3)
    parser.add_argument("--early-stop-tol", type=float, default=1e-10)
    parser.add_argument("--early-stop-patience", type=int, default=40)
    parser.add_argument("--jacobian-resolution", type=int, default=41)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    gmm_dir = Path(args.gmm_dir)
    for child in ["templates", "targets", "smoke_runs", "figures"]:
        (output_dir / child).mkdir(parents=True, exist_ok=True)
    dataset = load_airfoil_dataset(Path(args.data_path))
    train_indices = np.load(gmm_dir / "train_indices.npy")
    k1 = reconstruct_k1_template(dataset, train_indices)
    k2_templates, k2_metadata = reconstruct_gmm_templates(
        dataset,
        gmm_dir / "pca" / "pca_q20.npz",
        gmm_dir / "runs" / "q20" / "k2" / "full" / "seed0" / "result.npz",
    )
    templates = {
        "k1": [k1],
        "k2": [k2_templates[0], k2_templates[1]],
    }
    bounds = registration_bounds([k1, k2_templates[0], k2_templates[1], dataset["shapes"].reshape(-1, 2)], margin=0.08)
    reg_config = {
        "grid_size": args.grid_size,
        "num_steps": args.num_steps,
        "num_iters": args.num_iters,
        "lr": args.lr,
        "lambda_reg": args.lambda_reg,
        "early_stop_tol": args.early_stop_tol,
        "early_stop_patience": args.early_stop_patience,
        "jacobian_resolution": args.jacobian_resolution,
        "jacobian_nonpositive_tol": -1e-6,
        "severe_negative_thickness": 1e-3,
        "bounds": bounds,
        "device": "cpu",
        "optimizer": "Adam",
        "data_attachment": "corresponding_point_mse",
    }
    write_json(output_dir / "config.json", reg_config)
    np.save(output_dir / "templates" / "k1_global_mean.npy", k1)
    np.save(output_dir / "templates" / "k2_template_0.npy", k2_templates[0])
    np.save(output_dir / "templates" / "k2_template_1.npy", k2_templates[1])
    save_template_plot([("K=1 global mean", k1)], output_dir / "templates" / "k1_global_mean.png", "K=1 template")
    save_template_plot(
        [("K=2 template 0", k2_templates[0]), ("K=2 template 1", k2_templates[1])],
        output_dir / "templates" / "k2_templates.png",
        "K=2 GMM templates",
    )
    template_meta = {
        "k1_validity": validate_airfoil_shape(k1),
        "k2_validity": [validate_airfoil_shape(shape) for shape in k2_templates],
        "k2_weights": k2_metadata["weights"].tolist(),
        "source": "q20 K=2 full seed0 GMM means and training global mean",
    }
    write_json(output_dir / "templates" / "template_metadata.json", template_meta)

    target_indices = select_smoke_targets(dataset, gmm_dir, args.smoke_target_count)
    extreme_heldout = set(np.load(gmm_dir / "extreme_group_heldout_indices.npy").tolist())
    values = camber_thickness_from_shapes(dataset["shapes"])
    global_mean = values[train_indices].mean(axis=0, keepdims=True)
    distances = np.sqrt(np.sum((values - global_mean) ** 2, axis=1))
    target_rows = []
    for index in target_indices:
        target_rows.append(
            {
                "target_index": index,
                "filename": dataset["filenames"][index],
                "airfoil_name": dataset["names"][index],
                "distance_to_global_mean": float(distances[index]),
                "previous_extreme_heldout": index in extreme_heldout,
            }
        )
        np.save(output_dir / "targets" / f"target_{index}.npy", dataset["shapes"][index])
    save_csv(output_dir / "targets" / "smoke_targets.csv", target_rows)

    posteriors = gmm_posteriors_for_shapes(
        dataset["shapes"][target_indices],
        gmm_dir / "pca" / "pca_q20.npz",
        gmm_dir / "runs" / "q20" / "k2" / "full" / "seed0" / "result.npz",
    )
    result_rows = []
    for local_target, target_index in enumerate(target_indices):
        target = dataset["shapes"][target_index]
        target_name = dataset["filenames"][target_index]
        for strategy, source_list in templates.items():
            for template_index, source in enumerate(source_list):
                posterior = 1.0 if strategy == "k1" else float(posteriors[local_target, template_index])
                run_id = f"target{target_index}_{strategy}_template{template_index}"
                run_dir = output_dir / "smoke_runs" / run_id
                summary_path = run_dir / "summary.json"
                if summary_path.exists() and not args.force:
                    metrics = json.loads(summary_path.read_text(encoding="utf-8"))
                    deformed = np.load(run_dir / "deformed.npy")
                    history = list(csv.DictReader((run_dir / "history.csv").open(newline="", encoding="utf-8")))
                else:
                    run_dir.mkdir(parents=True, exist_ok=True)
                    registered = register_airfoils(source, target, reg_config, args.seed)
                    deformed = registered["deformed"]
                    history = registered["history"]
                    metrics = registered["metrics"]
                    np.save(run_dir / "deformed.npy", deformed)
                    np.save(run_dir / "coeffs.npy", registered["coeffs"])
                    save_csv(run_dir / "history.csv", history)
                    write_json(summary_path, metrics)
                    save_registration_figure(
                        source,
                        target,
                        deformed,
                        output_dir / "figures" / f"{run_id}.png",
                        f"{target_name}: {strategy} template {template_index}",
                    )
                    save_loss_curve(history, output_dir / "figures" / f"{run_id}_loss.png", f"{run_id} loss")
                result_rows.append(
                    {
                        "target_index": target_index,
                        "filename": target_name,
                        "strategy": strategy,
                        "k": 1 if strategy == "k1" else 2,
                        "template_index": template_index,
                        "gmm_posterior_probability": posterior,
                        "gmm_highest_posterior_template": int(np.argmax(posteriors[local_target])),
                        **metrics,
                    }
                )
    for row in result_rows:
        row["selected_template"] = False
        row["minimum_energy_template_agrees_with_gmm"] = False
    for target_index in target_indices:
        k1_rows = [row for row in result_rows if row["target_index"] == target_index and row["strategy"] == "k1"]
        k2_rows = [row for row in result_rows if row["target_index"] == target_index and row["strategy"] == "k2"]
        best_k2 = min(
            [row for row in k2_rows if row["failure_reason"] == ""],
            key=lambda row: row["total_loss"],
            default=min(k2_rows, key=lambda row: row["total_loss"]),
        )
        for row in k1_rows:
            row["selected_template"] = True
        for row in k2_rows:
            row["selected_template"] = row is best_k2 or row in k1_rows
            row["minimum_energy_template_agrees_with_gmm"] = (
                row["strategy"] == "k2"
                and best_k2["template_index"] == row["gmm_highest_posterior_template"]
            )
    save_csv(output_dir / "registration_results.csv", result_rows)

    paired = []
    for target_index in target_indices:
        k1_row = next(row for row in result_rows if row["target_index"] == target_index and row["strategy"] == "k1")
        k2_candidates = [row for row in result_rows if row["target_index"] == target_index and row["strategy"] == "k2"]
        best_k2 = min(
            [row for row in k2_candidates if row["failure_reason"] == ""],
            key=lambda row: row["total_loss"],
            default=min(k2_candidates, key=lambda row: row["total_loss"]),
        )
        paired.append(
            {
                "target_index": target_index,
                "filename": k1_row["filename"],
                "previous_extreme_heldout": target_index in extreme_heldout,
                "k1_total_loss": k1_row["total_loss"],
                "k2_total_loss": best_k2["total_loss"],
                "delta_total_loss": best_k2["total_loss"] - k1_row["total_loss"],
                "delta_attachment": best_k2["final_data_attachment"] - k1_row["final_data_attachment"],
                "delta_regularisation": best_k2["weighted_regulariser"] - k1_row["weighted_regulariser"],
                "k2_selected_template": best_k2["template_index"],
                "k2_gmm_highest_posterior_template": best_k2["gmm_highest_posterior_template"],
                "k2_selection_agrees_with_gmm": best_k2["template_index"] == best_k2["gmm_highest_posterior_template"],
            }
        )
    save_csv(output_dir / "paired_comparison.csv", paired)
    save_summary_figures(output_dir, paired, result_rows)
    summary = {
        "target_count": len(target_indices),
        "attempted_registrations": len(result_rows),
        "failure_count": int(sum(bool(row["failure_reason"]) for row in result_rows)),
        "k1_failure_count": int(sum(bool(row["failure_reason"]) for row in result_rows if row["strategy"] == "k1")),
        "k2_failure_count": int(sum(bool(row["failure_reason"]) for row in result_rows if row["strategy"] == "k2")),
        "mean_delta_total_loss": float(np.mean([row["delta_total_loss"] for row in paired])),
        "mean_delta_attachment": float(np.mean([row["delta_attachment"] for row in paired])),
        "mean_delta_regularisation": float(np.mean([row["delta_regularisation"] for row in paired])),
        "extreme_mean_delta_total_loss": float(np.mean([row["delta_total_loss"] for row in paired if row["previous_extreme_heldout"]])),
        "mean_runtime_seconds": float(np.mean([row["runtime_seconds"] for row in result_rows])),
        "hyperparameters": reg_config,
    }
    write_json(output_dir / "smoke_summary.json", summary)
    write_report(output_dir, summary, paired, result_rows, target_rows)
    print(json.dumps(summary, indent=2))


def save_summary_figures(output_dir: Path, paired: list[dict], rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(paired))
    ax.plot(x, [row["k1_total_loss"] for row in paired], marker="o", label="K=1")
    ax.plot(x, [row["k2_total_loss"] for row in paired], marker="o", label="K=2 best")
    ax.set_xlabel("smoke target")
    ax.set_ylabel("total loss")
    ax.set_title("Paired K=1 vs K=2 registration cost")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "paired_cost.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0.0, color="black", linewidth=1)
    ax.bar(x, [row["delta_total_loss"] for row in paired])
    ax.set_xlabel("smoke target")
    ax.set_ylabel("K=2 best - K=1 total loss")
    ax.set_title("Negative delta favours K=2")
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "delta_total_loss.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    selected = [row for row in rows if row.get("selected_template")]
    ax.scatter([row["initial_attachment"] for row in selected], [row["final_data_attachment"] for row in selected])
    ax.set_xlabel("initial corresponding-point MSE")
    ax.set_ylabel("final corresponding-point MSE")
    ax.set_title("Initial vs final attachment")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "initial_vs_final_attachment.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist([row["minimum_jacobian"] for row in rows], bins=20)
    ax.set_xlabel("minimum det(D phi)")
    ax.set_ylabel("registration count")
    ax.set_title("Jacobian determinant diagnostics")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "jacobian_minimums.png", dpi=200)
    plt.close(fig)


def write_report(output_dir: Path, summary: dict, paired: list[dict], rows: list[dict], targets: list[dict]) -> None:
    lines = [
        "# Airfoil 2D SVF Registration Smoke Report",
        "",
        "This smoke test compares K=1 global-mean registration with K=2 GMM-template registration on the same held-out targets. Registration uses a separate 2D SVF implementation, not the existing 3D bunny code.",
        "",
        "## Hyperparameters",
        "",
        *(f"- {key}: {value}" for key, value in summary["hyperparameters"].items()),
        "",
        "## Summary",
        "",
        f"- targets: {summary['target_count']}",
        f"- attempted registrations: {summary['attempted_registrations']}",
        f"- failures: {summary['failure_count']}",
        f"- mean delta total loss (K=2 - K=1): {summary['mean_delta_total_loss']:.6g}",
        f"- mean delta attachment: {summary['mean_delta_attachment']:.6g}",
        f"- mean delta weighted regularisation: {summary['mean_delta_regularisation']:.6g}",
        f"- extreme-target mean delta total loss: {summary['extreme_mean_delta_total_loss']:.6g}",
        f"- mean runtime per registration: {summary['mean_runtime_seconds']:.2f}s",
        "",
        "Negative delta means the best K=2 template improved over K=1 for that target.",
        "",
        "## Paired Results",
        "",
        "| target | extreme | delta total | K2 template | agrees with posterior |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in paired:
        lines.append(
            f"| {row['target_index']} | {row['previous_extreme_heldout']} | {row['delta_total_loss']:.6g} | {row['k2_selected_template']} | {row['k2_selection_agrees_with_gmm']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This smoke test is only a validation of the downstream experiment plumbing and a first signal. It should not be treated as the final 50-target result.",
            "",
        ]
    )
    (output_dir / "smoke_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
