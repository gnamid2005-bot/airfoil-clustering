import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture

from run_airfoil_gmm_selection import read_json, write_json
from run_airfoil_k_selection import adjusted_rand_index


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-check one custom airfoil GMM smoke run against scikit-learn."
    )
    parser.add_argument("--output-dir", default="outputs/airfoil_gmm_selection")
    parser.add_argument("--q", type=int, default=5)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--covariance-type", default="diagonal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-init", type=int, default=2)
    parser.add_argument("--reg-covar", type=float, default=1e-6)
    parser.add_argument("--max-iter", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    sklearn_covariance = {
        "diagonal": "diag",
        "spherical": "spherical",
        "tied": "tied",
        "full": "full",
    }[args.covariance_type]
    pca = np.load(output_dir / "pca" / f"pca_q{args.q}.npz")
    X_train = pca["train_scores"]

    run_dir = (
        output_dir
        / "runs"
        / f"q{args.q}"
        / f"k{args.k}"
        / args.covariance_type
        / f"seed{args.seed}"
    )
    custom_summary = read_json(run_dir / "summary.json")
    custom_result = np.load(run_dir / "result.npz")
    custom_labels = custom_result["train_labels"]

    sklearn_model = GaussianMixture(
        n_components=args.k,
        covariance_type=sklearn_covariance,
        n_init=args.n_init,
        reg_covar=args.reg_covar,
        max_iter=args.max_iter,
        init_params="kmeans",
        random_state=args.seed,
    )
    sklearn_model.fit(X_train)
    sklearn_labels = sklearn_model.predict(X_train)
    ari = adjusted_rand_index(custom_labels, sklearn_labels)

    result = {
        "configuration": {
            "q": args.q,
            "k": args.k,
            "custom_covariance_type": args.covariance_type,
            "sklearn_covariance_type": sklearn_covariance,
            "seed": args.seed,
            "sklearn_n_init": args.n_init,
            "custom_n_init": custom_summary["n_init"],
            "reg_covar": args.reg_covar,
            "max_iter": args.max_iter,
        },
        "custom": {
            "training_log_likelihood_per_sample": custom_summary["training_log_likelihood_per_sample"],
            "bic": custom_summary["bic"],
            "weights": custom_summary["component_weights"],
            "converged": custom_summary["converged"],
            "iterations": custom_summary["em_iterations"],
        },
        "sklearn": {
            "training_log_likelihood_per_sample": float(sklearn_model.score(X_train)),
            "bic": float(sklearn_model.bic(X_train)),
            "weights": sklearn_model.weights_.tolist(),
            "converged": bool(sklearn_model.converged_),
            "iterations": int(sklearn_model.n_iter_),
        },
        "comparison": {
            "hard_assignment_ari": ari,
            "absolute_log_likelihood_difference": abs(
                float(sklearn_model.score(X_train))
                - custom_summary["training_log_likelihood_per_sample"]
            ),
            "absolute_bic_difference": abs(float(sklearn_model.bic(X_train)) - custom_summary["bic"]),
            "note": (
                "Differences reflect optimisation and initialisation rather than "
                "label permutation: the custom implementation uses transparent "
                "k-means++ hard initialisation and keeps the best custom EM run; "
                "sklearn uses its own kmeans initialisation and EM details. "
                "Large ARI differences indicate different local optima, so final "
                "screening should rely on multiple seeds and stability diagnostics."
            ),
        },
    }
    write_json(output_dir / "sklearn_crosscheck.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
