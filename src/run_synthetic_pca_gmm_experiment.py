import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from run_airfoil_gmm_selection import (
    entropy_summary,
    fit_gmm,
    fit_pca,
    gmm_parameter_count,
    pca_inverse,
    pca_transform,
    predict_responsibilities,
)
from synthetic_closed_loops import generate_dataset, polygon_area, polygon_perimeter, save_dataset


matplotlib.use("Agg")

DEFAULT_OUTPUT_DIR = Path("outputs/synthetic_shape_clustering")


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_shapes(shapes: np.ndarray) -> np.ndarray:
    return shapes.reshape(shapes.shape[0], -1)


def stratified_split(labels: np.ndarray, heldout_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train = []
    heldout = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        indices = rng.permutation(indices)
        n_heldout = int(round(len(indices) * heldout_fraction))
        heldout.extend(indices[:n_heldout])
        train.extend(indices[n_heldout:])
    return np.asarray(sorted(train), dtype=np.int64), np.asarray(sorted(heldout), dtype=np.int64)


def confusion_matrix(true_labels: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
    n_true = int(true_labels.max()) + 1
    n_pred = int(pred_labels.max()) + 1
    matrix = np.zeros((n_true, n_pred), dtype=np.int64)
    np.add.at(matrix, (true_labels, pred_labels), 1)
    return matrix


def permutation_metrics(true_labels: np.ndarray, pred_labels: np.ndarray, family_names: np.ndarray) -> dict:
    matrix = confusion_matrix(true_labels, pred_labels)
    cost = -matrix
    row_ind, col_ind = linear_sum_assignment(cost)
    matched = int(matrix[row_ind, col_ind].sum())
    mapping = {int(col): int(row) for row, col in zip(row_ind, col_ind)}
    mapped = np.asarray([mapping.get(int(label), -1) for label in pred_labels], dtype=np.int64)
    aligned_matrix = confusion_matrix(true_labels, np.maximum(mapped, 0))

    recalls = {}
    precisions = {}
    for label, name in enumerate(family_names):
        tp = float(np.sum((true_labels == label) & (mapped == label)))
        fn = float(np.sum((true_labels == label) & (mapped != label)))
        fp = float(np.sum((true_labels != label) & (mapped == label)))
        recalls[f"recall_{name}"] = tp / (tp + fn) if tp + fn > 0 else 0.0
        precisions[f"precision_{name}"] = tp / (tp + fp) if tp + fp > 0 else 0.0

    return {
        "accuracy": matched / len(true_labels),
        "confusion_matrix": matrix.tolist(),
        "aligned_confusion_matrix": aligned_matrix.tolist(),
        "label_mapping_pred_to_true": mapping,
        **recalls,
        **precisions,
    }


def evaluate_predictions(true_labels: np.ndarray, pred_labels: np.ndarray, responsibilities: np.ndarray, family_names: np.ndarray) -> dict:
    entropy = entropy_summary(responsibilities)
    permutation = permutation_metrics(true_labels, pred_labels, family_names)
    return {
        "ari": float(adjusted_rand_score(true_labels, pred_labels)),
        "nmi": float(normalized_mutual_info_score(true_labels, pred_labels)),
        "posterior_entropy": entropy["classification_entropy"],
        "average_entropy": entropy["average_entropy"],
        "component_sizes": json.dumps(np.bincount(pred_labels, minlength=responsibilities.shape[1]).astype(int).tolist()),
        **{key: value for key, value in permutation.items() if key not in {"confusion_matrix", "aligned_confusion_matrix", "label_mapping_pred_to_true"}},
        "confusion_matrix": json.dumps(permutation["confusion_matrix"]),
        "aligned_confusion_matrix": json.dumps(permutation["aligned_confusion_matrix"]),
    }


def validate_shapes(shapes: np.ndarray, target_perimeter: float) -> dict:
    perimeters = np.asarray([polygon_perimeter(shape) for shape in shapes])
    centroids = shapes.mean(axis=1)
    areas = np.asarray([polygon_area(shape) for shape in shapes])
    consecutive = np.linalg.norm(np.roll(shapes, -1, axis=1) - shapes, axis=2)
    start_x = shapes[:, 0, 0]
    max_x = shapes[:, :, 0].max(axis=1)
    return {
        "n_shapes": int(shapes.shape[0]),
        "points_per_shape": int(shapes.shape[1]),
        "dimension": int(shapes.shape[1] * 2),
        "perimeter_min": float(perimeters.min()),
        "perimeter_max": float(perimeters.max()),
        "perimeter_max_abs_error": float(np.max(np.abs(perimeters - target_perimeter))),
        "centroid_max_norm": float(np.max(np.linalg.norm(centroids, axis=1))),
        "all_finite": bool(np.isfinite(shapes).all()),
        "all_ccw": bool(np.all(areas > 0.0)),
        "deterministic_rightmost_start": bool(np.allclose(start_x, max_x, atol=1e-10, rtol=0.0)),
        "min_consecutive_distance": float(consecutive.min()),
    }


def pca_family_diagnostics(scores: np.ndarray, labels: np.ndarray, family_names: np.ndarray, q: int) -> list[dict]:
    rows = []
    centroids = {}
    spreads = {}
    for label, name in enumerate(family_names):
        values = scores[labels == label, :q]
        centroid = values.mean(axis=0)
        centroids[str(name)] = centroid
        spreads[str(name)] = float(np.mean(np.linalg.norm(values - centroid, axis=1)))
        rows.append(
            {
                "q": q,
                "diagnostic": "within_family_spread",
                "family_a": str(name),
                "family_b": "",
                "value": spreads[str(name)],
                "centroid": json.dumps(centroid.tolist()),
            }
        )
    for i, name_a in enumerate(family_names):
        for j, name_b in enumerate(family_names):
            if j <= i:
                continue
            rows.append(
                {
                    "q": q,
                    "diagnostic": "between_centroid_distance",
                    "family_a": str(name_a),
                    "family_b": str(name_b),
                    "value": float(np.linalg.norm(centroids[str(name_a)] - centroids[str(name_b)])),
                    "centroid": "",
                }
            )
    return rows


def pairwise_stability(labels_by_seed: list[tuple[int, np.ndarray]]) -> dict:
    values = []
    for left in range(len(labels_by_seed)):
        for right in range(left + 1, len(labels_by_seed)):
            values.append(adjusted_rand_score(labels_by_seed[left][1], labels_by_seed[right][1]))
    if not values:
        return {"mean_pairwise_stability_ari": 1.0, "min_pairwise_stability_ari": 1.0, "max_pairwise_stability_ari": 1.0}
    return {
        "mean_pairwise_stability_ari": float(np.mean(values)),
        "min_pairwise_stability_ari": float(np.min(values)),
        "max_pairwise_stability_ari": float(np.max(values)),
    }


def make_stability_rows(rows: list[dict]) -> list[dict]:
    stability_rows = []
    keys = sorted({(row["pipeline"], row.get("q", ""), row["covariance_type"], row["k"]) for row in rows})
    for pipeline, q, covariance_type, k in keys:
        subset = [
            (int(row["seed"]), row["_heldout_pred"])
            for row in rows
            if row["pipeline"] == pipeline and row.get("q", "") == q and row["covariance_type"] == covariance_type and row["k"] == k
        ]
        stability_rows.append(
            {
                "pipeline": pipeline,
                "q": q,
                "covariance_type": covariance_type,
                "k": k,
                "n_seeds": len(subset),
                **pairwise_stability(subset),
            }
        )
    return stability_rows


def save_shape_examples(shapes: np.ndarray, labels: np.ndarray, family_names: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, len(family_names), figsize=(4 * len(family_names), 3.5))
    axes = np.atleast_1d(axes)
    for label, ax in enumerate(axes):
        candidates = np.flatnonzero(labels == label)[:5]
        for index in candidates:
            shape = shapes[index]
            ax.plot(np.r_[shape[:, 0], shape[0, 0]], np.r_[shape[:, 1], shape[0, 1]], alpha=0.65)
        ax.set_title(str(family_names[label]))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220)
    plt.close(fig)


def save_pca_scatter(scores: np.ndarray, labels: np.ndarray, family_names: np.ndarray, dims: tuple[int, int], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for label, name in enumerate(family_names):
        mask = labels == label
        ax.scatter(scores[mask, dims[0]], scores[mask, dims[1]], s=26, alpha=0.8, label=str(name))
    ax.set_xlabel(f"PC{dims[0] + 1}")
    ax.set_ylabel(f"PC{dims[1] + 1}")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220)
    plt.close(fig)


def save_confusion(matrix_json: str, family_names: np.ndarray, path: Path, title: str) -> None:
    matrix = np.asarray(json.loads(matrix_json), dtype=np.int64)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    image = ax.imshow(matrix, cmap="Blues")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    ax.set_yticks(np.arange(len(family_names)), family_names)
    ax.set_xlabel("Predicted component")
    ax.set_ylabel("True family")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220)
    plt.close(fig)


def save_metric_plot(rows: list[dict], pipeline: str, y_key: str, path: Path, title: str, ylabel: str) -> None:
    subset = [row for row in rows if row["pipeline"] == pipeline]
    if not subset:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cov in sorted({row["covariance_type"] for row in subset}):
        cov_rows = [row for row in subset if row["covariance_type"] == cov]
        groups = sorted({row.get("q", "raw") for row in cov_rows}, key=str)
        for group in groups:
            line = sorted([row for row in cov_rows if row.get("q", "raw") == group], key=lambda row: row["k"])
            label = f"{cov}, q={group}" if pipeline == "pca_gmm" else cov
            ax.plot([row["k"] for row in line], [row[y_key] for row in line], marker="o", label=label)
    ax.set_xlabel("K")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220)
    plt.close(fig)


def save_ari_vs_q(rows: list[dict], raw_reference: float, path: Path) -> None:
    selected = []
    for q in sorted({row["q"] for row in rows if row["pipeline"] == "pca_gmm"}):
        q_rows = [row for row in rows if row["pipeline"] == "pca_gmm" and row["q"] == q]
        selected.append((q, max(row["ari"] for row in q_rows)))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot([q for q, _ in selected], [ari for _, ari in selected], marker="o", label="Best PCA-GMM")
    ax.axhline(raw_reference, color="black", linestyle="--", label="Best raw-GMM")
    ax.set_xlabel("PCA dimension q")
    ax.set_ylabel("Held-out ARI")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("PCA-GMM versus raw-GMM")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220)
    plt.close(fig)


def save_mean_shapes(shapes: np.ndarray, rows: list[dict], output_dir: Path) -> None:
    for row in rows:
        mean_shapes = row.pop("_mean_shapes", None)
        if mean_shapes is None:
            continue
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        for comp, shape in enumerate(mean_shapes):
            ax.plot(np.r_[shape[:, 0], shape[0, 0]], np.r_[shape[:, 1], shape[0, 1]], label=f"component {comp}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        ax.set_title(f"{row['pipeline']} mean shapes")
        plt.tight_layout()
        name = f"{row['pipeline']}_q{row.get('q', 'raw')}_k{row['k']}_{row['covariance_type']}_seed{row['seed']}.png"
        plt.savefig(output_dir / "figures" / name, dpi=220)
        plt.close(fig)


def run_gmm_eval(
    X_train: np.ndarray,
    X_heldout: np.ndarray,
    y_train: np.ndarray,
    y_heldout: np.ndarray,
    family_names: np.ndarray,
    k: int,
    covariance_type: str,
    seed: int,
    n_init: int,
    reg_covar: float,
    max_iter: int,
    pipeline: str,
    q: int | None = None,
) -> tuple[dict, object, np.ndarray]:
    fit_start = time.perf_counter()
    model = fit_gmm(X_train, k, covariance_type, seed, n_init, reg_covar, max_iter, 1e-5)
    fit_runtime = time.perf_counter() - fit_start
    train_resp, train_pred, train_mean_ll = predict_responsibilities(model, X_train)
    heldout_resp, heldout_pred, heldout_mean_ll = predict_responsibilities(model, X_heldout)
    train_total_ll = train_mean_ll * X_train.shape[0]
    params = gmm_parameter_count(X_train.shape[1], k, covariance_type)
    bic = -2.0 * train_total_ll + params * math.log(X_train.shape[0])
    aic = -2.0 * train_total_ll + 2.0 * params
    icl = bic + 2.0 * entropy_summary(train_resp)["classification_entropy"]
    heldout_metrics = evaluate_predictions(y_heldout, heldout_pred, heldout_resp, family_names)
    train_metrics = evaluate_predictions(y_train, train_pred, train_resp, family_names)
    row = {
        "pipeline": pipeline,
        "q": q if q is not None else "",
        "k": k,
        "covariance_type": covariance_type,
        "seed": seed,
        "n_init": n_init,
        "bic": float(bic),
        "icl": float(icl),
        "aic": float(aic),
        "fit_runtime_seconds": float(fit_runtime),
        "train_mean_log_likelihood": float(train_mean_ll),
        "heldout_mean_log_likelihood": float(heldout_mean_ll),
        "component_weights": json.dumps(model.weights.tolist()),
        "converged": bool(model.converged),
        "iterations": int(model.iterations),
        "train_ari": train_metrics["ari"],
        "train_nmi": train_metrics["nmi"],
        **heldout_metrics,
        "_heldout_pred": heldout_pred,
    }
    return row, model, heldout_pred


def summarize_selection(rows: list[dict], true_k: int) -> list[dict]:
    summary = []
    keys = sorted({(row["pipeline"], row.get("q", ""), row["covariance_type"]) for row in rows})
    for pipeline, q, covariance_type in keys:
        subset = [row for row in rows if row["pipeline"] == pipeline and row.get("q", "") == q and row["covariance_type"] == covariance_type]
        by_k = []
        for k in sorted({row["k"] for row in subset}):
            k_rows = [row for row in subset if row["k"] == k]
            by_k.append(
                {
                    "k": k,
                    "mean_bic": float(np.mean([row["bic"] for row in k_rows])),
                    "mean_icl": float(np.mean([row["icl"] for row in k_rows])),
                    "mean_heldout_ll": float(np.mean([row["heldout_mean_log_likelihood"] for row in k_rows])),
                    "mean_ari": float(np.mean([row["ari"] for row in k_rows])),
                    "std_ari": float(np.std([row["ari"] for row in k_rows])),
                }
            )
        best_bic = min(by_k, key=lambda row: row["mean_bic"])
        best_icl = min(by_k, key=lambda row: row["mean_icl"])
        best_ll = max(by_k, key=lambda row: row["mean_heldout_ll"])
        best_ari = max(by_k, key=lambda row: row["mean_ari"])
        summary.append(
            {
                "pipeline": pipeline,
                "q": q,
                "covariance_type": covariance_type,
                "true_k": true_k,
                "k_bic": best_bic["k"],
                "k_icl": best_icl["k"],
                "k_best_heldout_ll": best_ll["k"],
                "k_best_ari": best_ari["k"],
                "bic_recovers_true_k": best_bic["k"] == true_k,
                "icl_recovers_true_k": best_icl["k"] == true_k,
                "best_mean_ari": best_ari["mean_ari"],
                "best_ari_std_across_seeds": best_ari["std_ari"],
            }
        )
    return summary


def interpret_results(rows: list[dict], summary_rows: list[dict], true_k: int) -> str:
    pca_best = max([row for row in rows if row["pipeline"] == "pca_gmm"], key=lambda row: row["ari"])
    raw_best = max([row for row in rows if row["pipeline"] == "raw_gmm"], key=lambda row: row["ari"])
    low_q = max([row for row in rows if row["pipeline"] == "pca_gmm" and row["q"] in {2, 3}], key=lambda row: row["ari"])
    high_q = max([row for row in rows if row["pipeline"] == "pca_gmm" and row["q"] in {10, 20}], key=lambda row: row["ari"])
    gap = raw_best["ari"] - pca_best["ari"]

    if pca_best["ari"] >= 0.90 and raw_best["ari"] >= 0.90 and gap <= 0.05:
        verdict = "结论：PCA-GMM 和 raw-GMM 都能很好恢复真实 family；这个 synthetic 设置下没有证据表明单一 global PCA 会破坏 family 结构。"
    elif pca_best["ari"] >= 0.90 and gap < -0.15:
        verdict = "结论：PCA-GMM 明显优于 raw-GMM；这个 synthetic 设置下不支持 Gabriel 关于单一 global PCA 会破坏 family 结构的担忧。"
    elif high_q["ari"] - low_q["ari"] > 0.20 and high_q["ari"] >= 0.80:
        verdict = "结论：低维 PCA 会损失 family 信息，但较高 q 可以恢复；主要风险更像是过激降维，而不是 global mean 本身。"
    elif gap > 0.15 and pca_best["ari"] < 0.80:
        verdict = "结论：raw-GMM 明显优于 PCA-GMM；这支持 Gabriel 对单一 global PCA 表示可能扭曲多 family 结构的担忧。"
    else:
        verdict = "结论：PCA-GMM 与 raw-GMM 的 family recovery 差异不构成强证据；应结合 q、covariance 和稳定性解读。"

    pca_k3 = max([row for row in rows if row["pipeline"] == "pca_gmm" and row["k"] == true_k], key=lambda row: row["ari"])
    raw_k3 = max([row for row in rows if row["pipeline"] == "raw_gmm" and row["k"] == true_k], key=lambda row: row["ari"])
    return (
        f"{verdict}\n\n"
        f"Best PCA-GMM: q={pca_best['q']}, K={pca_best['k']}, cov={pca_best['covariance_type']}, ARI={pca_best['ari']:.4f}.\n\n"
        f"Best K={true_k} PCA-GMM: q={pca_k3['q']}, cov={pca_k3['covariance_type']}, ARI={pca_k3['ari']:.4f}.\n\n"
        f"Best raw-GMM: K={raw_best['k']}, cov={raw_best['covariance_type']}, ARI={raw_best['ari']:.4f}.\n\n"
        f"Best K={true_k} raw-GMM: cov={raw_k3['covariance_type']}, ARI={raw_k3['ari']:.4f}.\n\n"
        "BIC/ICL/held-out likelihood 选择的是 Gaussian component count；它不必等同于 semantic family count。"
    )


def run_experiment(args: argparse.Namespace) -> dict:
    start = time.perf_counter()
    output_dir = Path(args.output_dir)
    experiment_dir = output_dir / args.experiment_name
    figures_dir = experiment_dir / "figures"
    tables_dir = experiment_dir / "tables"
    models_dir = experiment_dir / "models"
    report_dir = experiment_dir / "report"
    models_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    counts = {"ellipse": args.n_ellipses, "rectangle": args.n_rectangles}
    if args.include_octagons:
        counts["octagon"] = args.n_octagons
    dataset = generate_dataset(counts, seed=args.seed, m=args.m, target_perimeter=args.perimeter, dense_points=args.dense_points)
    data_path = output_dir / "data" / f"{args.experiment_name}.npz"
    save_dataset(data_path, dataset)

    shapes = dataset["shapes"]
    labels = dataset["true_labels"]
    family_names = dataset["family_names"].astype(str)
    validation = validate_shapes(shapes, args.perimeter)
    X = flatten_shapes(shapes)
    train_idx, heldout_idx = stratified_split(labels, args.heldout_fraction, args.split_seed)
    X_train = X[train_idx]
    X_heldout = X[heldout_idx]
    y_train = labels[train_idx]
    y_heldout = labels[heldout_idx]
    split_summary = {
        "train_total": int(len(train_idx)),
        "heldout_total": int(len(heldout_idx)),
        "train_counts": {str(family_names[label]): int(np.sum(y_train == label)) for label in np.unique(labels)},
        "heldout_counts": {str(family_names[label]): int(np.sum(y_heldout == label)) for label in np.unique(labels)},
    }

    save_shape_examples(shapes, labels, family_names, figures_dir / "example_shape_overlays.png")

    rows = []
    pca_rows = []
    pca_structure_rows = []
    selected_predictions = None
    selected_confusion = None

    max_q = max(args.q_values)
    pca_full = fit_pca(X_train, max_q)
    all_scores_full = pca_transform(pca_full, X)
    save_pca_scatter(all_scores_full, labels, family_names, (0, 1), figures_dir / "pca_true_pc1_pc2.png", "PCA scores by true family")
    if max_q >= 3:
        save_pca_scatter(all_scores_full, labels, family_names, (0, 2), figures_dir / "pca_true_pc1_pc3.png", "PCA scores by true family")
        save_pca_scatter(all_scores_full, labels, family_names, (1, 2), figures_dir / "pca_true_pc2_pc3.png", "PCA scores by true family")
    for q in args.q_values:
        pca = fit_pca(X_train, q)
        train_scores = pca_transform(pca, X_train)
        heldout_scores = pca_transform(pca, X_heldout)
        pca_rows.append(
            {
                "q": q,
                "explained_variance_ratio": json.dumps(pca.explained_variance_ratio.tolist()),
                "cumulative_explained_variance": float(np.sum(pca.explained_variance_ratio)),
            }
        )
        pca_structure_rows.extend(pca_family_diagnostics(pca_transform(pca, X), labels, family_names, q))
        np.savez_compressed(models_dir / f"pca_q{q}.npz", mean=pca.mean, components=pca.components, explained_variance_ratio=pca.explained_variance_ratio)
        for covariance_type in args.pca_covariances:
            for k in args.k_values:
                for seed in args.gmm_seeds:
                    row, model, pred = run_gmm_eval(
                        train_scores,
                        heldout_scores,
                        y_train,
                        y_heldout,
                        family_names,
                        k,
                        covariance_type,
                        seed,
                        args.n_init,
                        args.reg_covar,
                        args.max_iter,
                        "pca_gmm",
                        q=q,
                    )
                    if k == args.true_k:
                        means = pca_inverse(pca, model.means).reshape(k, args.m, 2)
                        row["_mean_shapes"] = means
                    rows.append(row)
                    if q == args.q_values[0] and k == args.true_k and covariance_type == args.pca_covariances[0] and seed == args.gmm_seeds[0]:
                        selected_predictions = pred
                        selected_confusion = row["confusion_matrix"]

    for covariance_type in args.raw_covariances:
        for k in args.k_values:
            for seed in args.gmm_seeds:
                row, model, _ = run_gmm_eval(
                    X_train,
                    X_heldout,
                    y_train,
                    y_heldout,
                    family_names,
                    k,
                    covariance_type,
                    seed,
                    args.n_init,
                    args.reg_covar,
                    args.max_iter,
                    "raw_gmm",
                )
                if k == args.true_k:
                    row["_mean_shapes"] = model.means.reshape(k, args.m, 2)
                rows.append(row)

    if selected_predictions is not None:
        fake_scores = all_scores_full[heldout_idx]
        save_pca_scatter(fake_scores, selected_predictions, np.asarray([f"comp_{i}" for i in sorted(np.unique(selected_predictions))]), (0, 1), figures_dir / "pca_gmm_labels_pc1_pc2.png", "Held-out PCA scores by GMM label")
    if selected_confusion is not None:
        save_confusion(selected_confusion, family_names, figures_dir / "selected_pca_gmm_confusion_matrix.png", "Selected PCA-GMM confusion")

    save_metric_plot(rows, "pca_gmm", "ari", figures_dir / "ari_vs_k_pca_gmm.png", "ARI versus K for PCA-GMM", "Held-out ARI")
    save_metric_plot(rows, "pca_gmm", "bic", figures_dir / "bic_vs_k_pca_gmm.png", "BIC versus K for PCA-GMM", "BIC")
    save_metric_plot(rows, "pca_gmm", "icl", figures_dir / "icl_vs_k_pca_gmm.png", "ICL versus K for PCA-GMM", "ICL")
    save_metric_plot(rows, "pca_gmm", "heldout_mean_log_likelihood", figures_dir / "heldout_ll_vs_k_pca_gmm.png", "Held-out likelihood versus K for PCA-GMM", "Mean held-out log-likelihood")
    save_metric_plot(rows, "raw_gmm", "ari", figures_dir / "ari_vs_k_raw_gmm.png", "ARI versus K for raw-GMM", "Held-out ARI")
    raw_best = max(row["ari"] for row in rows if row["pipeline"] == "raw_gmm")
    save_ari_vs_q(rows, raw_best, figures_dir / "ari_vs_pca_dimension_with_raw_reference.png")
    save_mean_shapes(shapes, rows, experiment_dir)

    stability_rows = make_stability_rows(rows)
    result_rows = [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]
    summary_rows = summarize_selection(result_rows, args.true_k)
    bias_rows = []
    for q in args.q_values:
        q_candidates = [row for row in result_rows if row["pipeline"] == "pca_gmm" and row["q"] == q]
        best = max(q_candidates, key=lambda row: row["ari"])
        raw = max([row for row in result_rows if row["pipeline"] == "raw_gmm"], key=lambda row: row["ari"])
        bias_rows.append(
            {
                "representation": "pca_gmm",
                "q": q,
                "best_k": best["k"],
                "covariance_type": best["covariance_type"],
                "heldout_ari": best["ari"],
                "heldout_nmi": best["nmi"],
                "accuracy": best["accuracy"],
                "best_raw_gmm_ari": raw_best,
                "ari_gap_pca_minus_raw": best["ari"] - raw_best,
            }
        )
    raw_best_row = max([row for row in result_rows if row["pipeline"] == "raw_gmm"], key=lambda row: row["ari"])
    bias_rows.append(
        {
            "representation": "raw_gmm",
            "q": "",
            "best_k": raw_best_row["k"],
            "covariance_type": raw_best_row["covariance_type"],
            "heldout_ari": raw_best_row["ari"],
            "heldout_nmi": raw_best_row["nmi"],
            "accuracy": raw_best_row["accuracy"],
            "best_raw_gmm_ari": raw_best,
            "ari_gap_pca_minus_raw": 0.0,
        }
    )

    best_models = []
    for pipeline in ["pca_gmm", "raw_gmm"]:
        candidates = [row for row in result_rows if row["pipeline"] == pipeline]
        if candidates:
            best_models.append({"role": f"best_{pipeline}", **max(candidates, key=lambda row: row["ari"])})
        k3 = [row for row in candidates if row["k"] == args.true_k]
        if k3:
            best_models.append({"role": f"best_k{args.true_k}_{pipeline}", **max(k3, key=lambda row: row["ari"])})
    pca_candidates = [row for row in result_rows if row["pipeline"] == "pca_gmm"]
    if pca_candidates:
        best_models.append({"role": "bic_selected_pca_gmm", **min(pca_candidates, key=lambda row: row["bic"])})
        best_models.append({"role": "icl_selected_pca_gmm", **min(pca_candidates, key=lambda row: row["icl"])})
        best_models.append({"role": "heldout_ll_selected_pca_gmm", **max(pca_candidates, key=lambda row: row["heldout_mean_log_likelihood"])})

    for best in best_models:
        save_confusion(
            best["confusion_matrix"],
            family_names,
            figures_dir / f"{best['role']}_confusion_matrix.png",
            f"{best['role']} confusion",
        )

    save_csv(tables_dir / "all_results.csv", result_rows)
    save_csv(tables_dir / f"{args.experiment_name}_results.csv", result_rows)
    save_csv(tables_dir / "ground_truth_metrics.csv", result_rows)
    save_csv(tables_dir / "model_selection_summary.csv", summary_rows)
    save_csv(tables_dir / "pca_bias_comparison.csv", bias_rows)
    save_csv(tables_dir / "stability_results.csv", stability_rows)
    save_csv(tables_dir / "best_models_summary.csv", best_models)
    save_csv(tables_dir / f"{args.experiment_name}_pca_diagnostics.csv", pca_rows)
    save_csv(tables_dir / "pca_structure_diagnostics.csv", pca_structure_rows)
    write_json(experiment_dir / "config.json", vars(args))

    runtime = time.perf_counter() - start
    report = {
        "data_path": str(data_path),
        "runtime_seconds": runtime,
        "validation": validation,
        "split_summary": split_summary,
        "pca_diagnostics": pca_rows,
        "best_pca_gmm_ari": max(row["ari"] for row in result_rows if row["pipeline"] == "pca_gmm"),
        "best_raw_gmm_ari": raw_best,
        "best_models": best_models,
        "model_selection_summary": summary_rows,
        "example_generation_parameters": [json.loads(item) for item in dataset["generation_parameters"][:4]],
    }
    write_json(report_dir / "summary.json", report)
    title = "Synthetic three-family full experiment" if args.include_octagons else "Synthetic shape clustering smoke test"
    conclusion = interpret_results(result_rows, summary_rows, args.true_k)
    report_text = (
        f"# {title}\n\n"
        f"Experiment: `{args.experiment_name}`\n\n"
        f"Runtime seconds: {runtime:.3f}\n\n"
        f"Perimeter max abs error: {validation['perimeter_max_abs_error']:.6g}\n\n"
        f"Best PCA-GMM held-out ARI: {report['best_pca_gmm_ari']:.6f}\n\n"
        f"Best raw-GMM held-out ARI: {report['best_raw_gmm_ari']:.6f}\n\n"
        f"{conclusion}\n"
    )
    (report_dir / "final_report.md").write_text(
        report_text,
        encoding="utf-8",
    )
    (experiment_dir / "final_report.md").write_text(
        report_text,
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled synthetic closed-loop PCA/GMM experiments.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--experiment-name", default="experiment1_smoke")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--m", type=int, default=100)
    parser.add_argument("--perimeter", type=float, default=4.0)
    parser.add_argument("--dense-points", type=int, default=2000)
    parser.add_argument("--n-ellipses", type=int, default=20)
    parser.add_argument("--n-rectangles", type=int, default=20)
    parser.add_argument("--n-octagons", type=int, default=0)
    parser.add_argument("--include-octagons", action="store_true")
    parser.add_argument("--true-k", type=int, default=2)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--q-values", type=parse_int_list, default="2,5")
    parser.add_argument("--k-values", type=parse_int_list, default="1,2,3")
    parser.add_argument("--pca-covariances", type=parse_str_list, default="spherical,diagonal")
    parser.add_argument("--raw-covariances", type=parse_str_list, default="spherical,diagonal")
    parser.add_argument("--gmm-seeds", type=parse_int_list, default="0,1")
    parser.add_argument("--n-init", type=int, default=2)
    parser.add_argument("--reg-covar", type=float, default=1e-6)
    parser.add_argument("--max-iter", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    report = run_experiment(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
