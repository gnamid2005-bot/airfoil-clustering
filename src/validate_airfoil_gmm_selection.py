import argparse
from pathlib import Path

import numpy as np

from run_airfoil_gmm_selection import (
    gmm_parameter_count,
    load_dataset,
    previous_extreme_indices,
    read_json,
    stratified_split,
)
from run_airfoil_k_selection import adjusted_rand_index


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lightweight validation checks for airfoil PCA-GMM smoke outputs."
    )
    parser.add_argument("--output-dir", default="outputs/airfoil_gmm_selection")
    parser.add_argument("--data-path", default="data/airfoils/processed/airfoils_n100_seed0.npz")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    dataset = load_dataset(Path(args.data_path), "camber_thickness", 1e-3)

    split_a = stratified_split(
        dataset["X"],
        dataset["filenames"],
        seed=0,
        heldout_fraction=0.2,
        registration_subset_size=50,
        k5_assignments_path=Path("outputs/airfoil_kmeans/k5_seed0/assignments.csv"),
    )
    split_b = stratified_split(
        dataset["X"],
        dataset["filenames"],
        seed=0,
        heldout_fraction=0.2,
        registration_subset_size=50,
        k5_assignments_path=Path("outputs/airfoil_kmeans/k5_seed0/assignments.csv"),
    )
    assert np.array_equal(split_a["train_indices"], split_b["train_indices"])
    assert np.array_equal(split_a["heldout_indices"], split_b["heldout_indices"])

    labels = np.asarray([0, 0, 1, 1, 2, 2])
    permuted = np.asarray([2, 2, 0, 0, 1, 1])
    assert abs(adjusted_rand_index(labels, permuted) - 1.0) < 1e-12

    assert gmm_parameter_count(5, 3, "spherical") == (3 - 1) + 3 * 5 + 3
    assert gmm_parameter_count(5, 3, "diagonal") == (3 - 1) + 3 * 5 + 3 * 5
    assert gmm_parameter_count(5, 3, "tied") == (3 - 1) + 3 * 5 + 5 * 6 // 2
    assert gmm_parameter_count(5, 3, "full") == (3 - 1) + 3 * 5 + 3 * 5 * 6 // 2

    for q in [5]:
        pca = np.load(output_dir / "pca" / f"pca_q{q}.npz")
        assert pca["components"].shape[0] == q
        assert pca["train_scores"].shape[1] == q
        assert pca["heldout_scores"].shape[1] == q

    for covariance_type in ["spherical", "diagonal"]:
        for k in [1, 2, 3]:
            for seed in [0, 1]:
                run_dir = output_dir / "runs" / "q5" / f"k{k}" / covariance_type / f"seed{seed}"
                summary = read_json(run_dir / "summary.json")
                assert summary["status"] == "completed"
                assert np.isfinite(summary["bic"])
                assert np.isfinite(summary["icl"])
                result = np.load(run_dir / "result.npz")
                weights = result["weights"]
                train_resp = result["train_responsibilities"]
                heldout_resp = result["heldout_responsibilities"]
                assert weights.shape == (k,)
                assert abs(float(np.sum(weights)) - 1.0) < 1e-8
                assert np.allclose(train_resp.sum(axis=1), 1.0, atol=1e-8)
                assert np.allclose(heldout_resp.sum(axis=1), 1.0, atol=1e-8)
                if k == 1:
                    assert abs(summary["classification_entropy"]) < 1e-8
                    assert abs(summary["average_entropy"]) < 1e-8

    assert previous_extreme_indices(
        dataset["filenames"],
        Path("outputs/airfoil_kmeans/k5_seed0/assignments.csv"),
    ).shape[0] == 31
    summary = read_json(output_dir / "split_summary.json")
    assert summary["previous_extreme_count"] == 31
    assert summary["previous_extreme_in_train"] == 25
    assert summary["previous_extreme_in_heldout"] == 6
    extreme_train = np.load(output_dir / "extreme_group_train_indices.npy")
    extreme_heldout = np.load(output_dir / "extreme_group_heldout_indices.npy")
    assert extreme_train.shape[0] == 25
    assert extreme_heldout.shape[0] == 6
    assert set(extreme_train.tolist()).isdisjoint(set(extreme_heldout.tolist()))
    print("airfoil GMM validation checks passed")


if __name__ == "__main__":
    main()
