"""Read-only audit of Gabriel production assignments against R=200 modes."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
PROD = ROOT / "outputs/gabriel_k2_k8_energy_magnitude"
MODAL = ROOT / "outputs/airfoil_kmeans_modal_assignments_r200"
DIRECT = ROOT / "data/airfoils/processed/airfoils_n100_seed0.npz"
PINNED = ROOT / "data/airfoils/processed_gabriel/airfoils_pinned_arclength_n200.npz"
OUT = ROOT / "outputs/gabriel_assignment_modal_audit"
KS = range(2, 10)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def optimal_label_difference(a: np.ndarray, b: np.ndarray, k: int) -> tuple[int, float, dict[int, int]]:
    contingency = np.zeros((k, k), dtype=np.int64)
    np.add.at(contingency, (a, b), 1)
    rows, cols = linear_sum_assignment(-contingency)
    mapping = {int(c): int(r) for r, c in zip(rows, cols)}
    matched_b = np.asarray([mapping[int(v)] for v in b])
    n_diff = int(np.count_nonzero(a != matched_b))
    return n_diff, n_diff / len(a), mapping


def flag(ari: float) -> str:
    if ari == 1.0: return "identical"
    if ari >= .95: return "nearly_identical"
    if ari >= .80: return "moderately_different"
    return "substantially_different"


def main() -> None:
    if OUT.exists(): raise FileExistsError(f"Refusing to overwrite {OUT}")
    (OUT / "modal_templates").mkdir(parents=True)
    started = time.perf_counter()
    config = json.loads((PROD / "config.json").read_text())
    assignments = read_csv(PROD / "clustering_assignments.csv")
    template_summary = read_csv(PROD / "template_summary.csv")
    modal_summary = {int(r["K"]): r for r in read_csv(MODAL / "modal_assignment_summary_k2_k20.csv")}
    with np.load(DIRECT, allow_pickle=False) as z:
        direct_names = z["filenames"].astype(str)
    with np.load(PINNED, allow_pickle=False) as z:
        contours = z["contours"].astype(np.float64); pinned_names = z["filenames"].astype(str)
        existing_indices = z["matching_existing_dataset_indices"].astype(int)
    matched_pinned = np.flatnonzero(existing_indices >= 0)
    matched_direct = existing_indices[matched_pinned]
    if not np.array_equal(direct_names[matched_direct], pinned_names[matched_pinned]):
        raise RuntimeError("Saved cross-dataset filename mapping is inconsistent")

    output_rows, cluster_rows = [], []
    all_template_exact, all_members_exact, all_checkpoint_refs = True, True, True
    for k in KS:
        ar = sorted([r for r in assignments if int(r["K"]) == k], key=lambda r: int(r["target_index"]))
        if len(ar) != 1652 or [int(r["target_index"]) for r in ar] != list(range(1652)):
            raise RuntimeError(f"K={k} production assignment incomplete")
        if [r["filename"] for r in ar] != pinned_names.tolist():
            raise RuntimeError(f"K={k} production assignment filename order mismatch")
        prod_labels = np.asarray([int(r["cluster_id"]) for r in ar], dtype=np.int32)
        modal_file = MODAL / "assignments" / f"modal_assignment_K{k:02d}.npy"
        modal_all = np.load(modal_file).astype(np.int32)
        modal_labels = modal_all[matched_direct]
        prod_common = prod_labels[matched_pinned]
        ari = float(adjusted_rand_score(prod_common, modal_labels))
        nmi = float(normalized_mutual_info_score(prod_common, modal_labels))
        n_diff, diff_fraction, label_map = optimal_label_difference(prod_common, modal_labels, k)
        exact = bool(ari == 1.0 and n_diff == 0)

        production_templates, production_ids = [], []
        for cluster in range(k):
            p = PROD / "templates" / f"K{k:02d}" / f"K{k:02d}_cluster{cluster:02d}.npz"
            with np.load(p, allow_pickle=False) as z:
                coordinates = z["coordinates"].astype(np.float64); members = z["member_indices"].astype(int)
            expected_members = np.flatnonzero(prod_labels == cluster)
            members_exact = bool(np.array_equal(members, expected_members))
            coords_exact = bool(np.array_equal(coordinates, contours[members].mean(axis=0)))
            all_members_exact &= members_exact; all_template_exact &= coords_exact
            production_templates.append(coordinates); production_ids.append(p.name)
        production_templates = np.stack(production_templates)

        modal_templates = np.stack([contours[matched_pinned][modal_labels == cluster].mean(axis=0) for cluster in range(k)])
        modal_counts = np.bincount(modal_labels, minlength=k)
        np.savez_compressed(OUT / "modal_templates" / f"modal_templates_K{k:02d}.npz",
                            coordinates=modal_templates, cluster_ids=np.arange(k),
                            member_counts_common_cohort=modal_counts,
                            common_pinned_indices=matched_pinned,
                            common_modal_labels=modal_labels)
        mse_matrix = np.mean((production_templates[:, None] - modal_templates[None, :]) ** 2, axis=(2, 3))
        prow, mcol = linear_sum_assignment(mse_matrix)
        mses = mse_matrix[prow, mcol]
        l2s = np.sqrt(np.sum((production_templates[prow] - modal_templates[mcol]) ** 2, axis=(1, 2)))
        for pc, mc, mse, l2 in zip(prow, mcol, mses, l2s):
            cluster_rows.append({"K": k, "production_cluster": int(pc), "modal_cluster": int(mc),
                                 "production_template_file": production_ids[int(pc)],
                                 "production_cluster_size_all_1652": int(np.sum(prod_labels == pc)),
                                 "modal_cluster_size_common_1644": int(modal_counts[mc]),
                                 "pointwise_mse": float(mse), "template_l2": float(l2)})

        checkpoints_reference_templates = True
        for i, label in enumerate(prod_labels):
            cp = PROD / "registrations" / f"K{k:02d}" / f"{i:04d}.npz"
            with np.load(cp, allow_pickle=False) as z:
                metrics = json.loads(str(z["metrics_json"]))
            expected_id = f"K{k:02d}_cluster{int(label):02d}"
            if metrics.get("source_template_id") != expected_id or int(metrics.get("cluster_id", -1)) != int(label):
                checkpoints_reference_templates = False; break
        all_checkpoint_refs &= checkpoints_reference_templates
        mr = modal_summary[k]
        output_rows.append({"K": k,
            "registration_assignment_source": "outputs/gabriel_k2_k8_energy_magnitude/clustering_assignments.csv",
            "registration_seed": config["K9_extension"]["kmeans"]["random_state"] if k == 9 else config["kmeans"]["random_state"],
            "registration_n_init": config["K9_extension"]["kmeans"]["n_init"] if k == 9 else config["kmeans"]["n_init"],
            "n_airfoils_production": 1652, "n_airfoils_compared": len(matched_pinned),
            "production_cluster_sizes": json.dumps(np.bincount(prod_labels, minlength=k).tolist()),
            "production_templates_verified_from_assignment": all(np.array_equal(production_templates[c], contours[prod_labels == c].mean(axis=0)) for c in range(k)),
            "production_checkpoints_reference_assignment_templates": checkpoints_reference_templates,
            "modal_assignment_source": str(modal_file.relative_to(ROOT)),
            "modal_seed": int(mr["modal_representative_seed"]), "modal_frequency": int(mr["modal_frequency"]),
            "modal_fraction": float(mr["modal_fraction"]),
            "ARI_registration_vs_modal": ari, "NMI_registration_vs_modal": nmi,
            "exact_same_partition": exact, "n_airfoils_different": n_diff,
            "fraction_airfoils_different": diff_fraction,
            "mean_template_mse": float(mses.mean()), "max_template_mse": float(mses.max()),
            "mean_template_l2": float(l2s.mean()), "max_template_l2": float(l2s.max()),
            "descriptive_flag": flag(ari), "optimal_modal_to_production_label_map": json.dumps(label_map, sort_keys=True)})

    write_csv(OUT / "registration_vs_modal_assignment_summary.csv", output_rows)
    write_csv(OUT / "template_matching_details.csv", cluster_rows)
    validation = {"kmeans_rerun": False, "registration_rerun": False, "templates_rebuilt_in_production": False,
        "production_assignment_file": str((PROD / "clustering_assignments.csv").relative_to(ROOT)),
        "production_assignment_rows": len(assignments), "expected_rows": 1652 * 8,
        "production_dataset_shape": list(contours.shape), "direct_modal_dataset_n": len(direct_names),
        "common_airfoils": len(matched_pinned),
        "pinned_without_modal_label": pinned_names[existing_indices < 0].tolist(),
        "direct_without_pinned_counterpart": sorted(set(direct_names) - set(pinned_names)),
        "production_template_member_indices_match_assignment": all_members_exact,
        "production_template_coordinates_are_exact_means": all_template_exact,
        "all_13216_registration_checkpoints_reference_expected_templates": all_checkpoint_refs,
        "template_comparison_caveat": "modal templates use 1644 common pinned contours; production templates use all 1652 pinned contours",
        "descriptive_thresholds": {"identical": "ARI == 1", "nearly_identical": "ARI >= 0.95", "moderately_different": "0.80 <= ARI < 0.95", "substantially_different": "ARI < 0.80"},
        "runtime_seconds": time.perf_counter() - started,
        "source_sha256": {"production_assignments": sha256(PROD / "clustering_assignments.csv"),
                          "production_config": sha256(PROD / "config.json"),
                          "modal_summary": sha256(MODAL / "modal_assignment_summary_k2_k20.csv"),
                          "pinned_dataset": sha256(PINNED), "direct_dataset": sha256(DIRECT)}}
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__": main()
