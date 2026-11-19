import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from run_airfoil_gmm_selection import fit_pca, pca_inverse, pca_transform
from synthetic_closed_loops import generate_dataset, load_dataset, polygon_area, polygon_perimeter


PROTECTED_OUTPUTS = [
    Path("outputs/airfoil_gmm_selection"),
    Path("outputs/airfoil_registration_validation"),
    Path("outputs/airfoil_kmeans"),
    Path("outputs/airfoil_k_selection"),
]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate synthetic closed-loop PCA/GMM smoke outputs.")
    parser.add_argument("--output-dir", default="outputs/synthetic_shape_clustering")
    parser.add_argument("--experiment-name", default="experiment1_smoke")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--m", type=int, default=100)
    parser.add_argument("--perimeter", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    data_path = output_dir / "data" / f"{args.experiment_name}.npz"
    assert data_path.exists(), f"Missing synthetic dataset: {data_path}"
    data = load_dataset(data_path)
    shapes = data["shapes"]
    labels = data["true_labels"]

    assert shapes.ndim == 3 and shapes.shape[1:] == (args.m, 2)
    assert np.isfinite(shapes).all()
    perimeters = np.asarray([polygon_perimeter(shape) for shape in shapes])
    assert np.max(np.abs(perimeters - args.perimeter)) < 2e-3
    assert np.max(np.linalg.norm(shapes.mean(axis=1), axis=1)) < 1e-10
    assert np.all([polygon_area(shape) > 0.0 for shape in shapes])
    assert np.min(np.linalg.norm(np.roll(shapes, -1, axis=1) - shapes, axis=2)) > 1e-8

    regenerated = generate_dataset({"ellipse": 20, "rectangle": 20}, seed=args.seed, m=args.m, target_perimeter=args.perimeter)
    assert np.array_equal(shapes, regenerated["shapes"])
    assert np.array_equal(labels, regenerated["true_labels"])

    unique, counts = np.unique(labels, return_counts=True)
    assert unique.tolist() == [0, 1]
    assert counts.tolist() == [20, 20]

    X = shapes.reshape(shapes.shape[0], -1)
    train = np.r_[0:16, 20:36]
    heldout = np.asarray([i for i in range(40) if i not in set(train.tolist())])
    pca = fit_pca(X[train], 2)
    heldout_scores = pca_transform(pca, X[heldout])
    reconstructed = pca_inverse(pca, heldout_scores)
    assert reconstructed.shape == X[heldout].shape
    assert pca.mean.shape == (args.m * 2,)
    assert pca.components.shape == (2, args.m * 2)

    labels_a = np.asarray([0, 0, 1, 1, 2, 2])
    labels_b = np.asarray([2, 2, 0, 0, 1, 1])
    assert abs(adjusted_rand_score(labels_a, labels_b) - 1.0) < 1e-12

    results_path = output_dir / "tables" / f"{args.experiment_name}_results.csv"
    assert results_path.exists(), f"Missing results table: {results_path}"
    rows = read_csv(results_path)
    assert rows
    assert {"pca_gmm", "raw_gmm"}.issubset({row["pipeline"] for row in rows})
    assert all("true_labels" not in row for row in rows)
    assert all(Path(row).parts[:2] != ("outputs", "airfoil_gmm_selection") for row in [])

    for protected in PROTECTED_OUTPUTS:
        if protected.exists():
            assert protected != output_dir
            assert not any(part == "synthetic_shape_clustering" for part in protected.parts)

    required = [
        output_dir / "config.json",
        output_dir / "final_report.md",
        output_dir / "tables" / "model_selection_summary.csv",
        output_dir / "tables" / "ground_truth_metrics.csv",
        output_dir / "tables" / "pca_bias_comparison.csv",
        output_dir / "tables" / "stability_results.csv",
        output_dir / args.experiment_name / "figures" / "example_shape_overlays.png",
        output_dir / args.experiment_name / "figures" / "pca_true_pc1_pc2.png",
        output_dir / args.experiment_name / "figures" / "selected_pca_gmm_confusion_matrix.png",
    ]
    for path in required:
        assert path.exists(), f"Missing expected smoke output: {path}"

    print("synthetic shape smoke validation checks passed")
    print(f"max perimeter abs error: {np.max(np.abs(perimeters - args.perimeter)):.6g}")
    print(f"centroid max norm: {np.max(np.linalg.norm(shapes.mean(axis=1), axis=1)):.6g}")


if __name__ == "__main__":
    main()

