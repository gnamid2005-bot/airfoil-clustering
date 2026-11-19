"""Read-only post-processing of saved R=200 direct-shape K-means labels."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score

SOURCE = Path("outputs/airfoil_kmeans_stability_r200")
OUTPUT = Path("outputs/airfoil_kmeans_modal_assignments_r200")
KS = range(2, 21)


def canonicalize(labels: np.ndarray) -> np.ndarray:
    """Relabel clusters 0,1,... in order of first occurrence."""
    labels = np.asarray(labels)
    out = np.empty(labels.shape, dtype=np.int32)
    mapping: dict[int, int] = {}
    next_label = 0
    for i, raw in enumerate(labels):
        value = int(raw)
        if value not in mapping:
            mapping[value] = next_label
            next_label += 1
        out[i] = mapping[value]
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title); ax.set_xlabel("Number of clusters K"); ax.set_ylabel(ylabel)
    ax.set_xticks(list(KS)); ax.grid(True, alpha=.22, color="#9CA3AF")
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    (OUTPUT / "figures").mkdir(parents=True)
    (OUTPUT / "assignments").mkdir()
    started = time.perf_counter()

    tests = {
        "permutation_example": bool(np.array_equal(canonicalize([5,5,2,2,8,8]), [0,0,1,1,2,2])),
        "arbitrary_permutation_equal": bool(np.array_equal(canonicalize([0,0,1,2,2,1]), canonicalize([7,7,4,9,9,4]))),
        "different_partition_not_equal": bool(not np.array_equal(canonicalize([0,0,1,1]), canonicalize([0,1,0,1]))),
        "idempotent": bool(np.array_equal(canonicalize(canonicalize([9,2,9,4,2])), canonicalize([9,2,9,4,2]))),
    }
    if not all(tests.values()):
        raise AssertionError(tests)

    ari_sums = {k: np.zeros(200, dtype=np.float64) for k in KS}
    ari_counts = {k: np.zeros(200, dtype=np.int32) for k in KS}
    pair_rows = 0
    with (SOURCE / "tables" / "pairwise_ari_r200.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            k, a, b, ari = int(row["k"]), int(row["seed_a"]), int(row["seed_b"]), float(row["ari"])
            ari_sums[k][a] += ari; ari_sums[k][b] += ari
            ari_counts[k][a] += 1; ari_counts[k][b] += 1; pair_rows += 1

    summary_rows, all_seed_rows = [], []
    total_vectors = 0
    for k in KS:
        path = SOURCE / "all_runs" / f"labels_k{k:02d}.npz"
        with np.load(path, allow_pickle=False) as z:
            labels = np.asarray(z["labels"]); seeds = np.asarray(z["seeds"])
        total_vectors += labels.shape[0]
        canonical = np.stack([canonicalize(v) for v in labels])
        groups: dict[bytes, list[int]] = defaultdict(list)
        representatives: dict[bytes, np.ndarray] = {}
        for i, seed in enumerate(seeds):
            key = canonical[i].tobytes()
            groups[key].append(int(seed)); representatives[key] = canonical[i]
        max_frequency = max(map(len, groups.values()))
        tied = sorted([(min(seed_list), key, seed_list) for key, seed_list in groups.items() if len(seed_list) == max_frequency])
        representative_seed, modal_key, modal_seeds = tied[0]
        modal_assignment = representatives[modal_key]
        means = ari_sums[k] / ari_counts[k]
        medoid_seed = int(np.argmax(means))
        medoid_assignment = canonical[medoid_seed]
        mode_medoid_ari = float(adjusted_rand_score(modal_assignment, medoid_assignment))
        equals = bool(np.array_equal(modal_assignment, medoid_assignment))
        np.save(OUTPUT / "assignments" / f"modal_assignment_K{k:02d}.npy", modal_assignment)
        metadata = {"k": k, "canonical_assignment_file": f"modal_assignment_K{k:02d}.npy",
                    "representative_seed": representative_seed, "all_seeds": modal_seeds,
                    "frequency": max_frequency, "fraction": max_frequency / 200,
                    "n_tied_modal_partitions": len(tied),
                    "all_tied_modal_seed_lists": [v[2] for v in tied]}
        (OUTPUT / "assignments" / f"modal_assignment_K{k:02d}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if len(tied) > 1:
            for tie_index, (_, key, seed_list) in enumerate(tied, 1):
                np.save(OUTPUT / "assignments" / f"modal_assignment_K{k:02d}_tie{tie_index:02d}.npy", representatives[key])
        summary_rows.append({"K": k, "n_runs": 200, "n_unique_partitions": len(groups),
            "modal_frequency": max_frequency, "modal_fraction": max_frequency / 200,
            "n_tied_modal_partitions": len(tied), "modal_representative_seed": representative_seed,
            "modal_seed_list": json.dumps(modal_seeds),
            "all_tied_modal_seed_lists": json.dumps([v[2] for v in tied]),
            "ari_medoid_seed": medoid_seed, "ari_medoid_mean_ari": float(means[medoid_seed]),
            "ari_mode_vs_medoid": mode_medoid_ari, "mode_equals_medoid": equals})
        for seed in seeds:
            all_seed_rows.append({"K": k, "seed": int(seed), "mean_ari_to_other_199": float(means[int(seed)]),
                                  "is_ari_medoid": int(seed) == medoid_seed,
                                  "is_selected_modal_partition": int(seed) in modal_seeds})

    write_csv(OUTPUT / "modal_assignment_summary_k2_k20.csv", summary_rows)
    write_csv(OUTPUT / "per_seed_mean_ari_k2_k20.csv", all_seed_rows)
    k = np.asarray([r["K"] for r in summary_rows])
    plots = [
        ("modal_partition_frequency_vs_k.png", [r["modal_fraction"] for r in summary_rows], "Modal exact-partition fraction across 200 runs", "Modal fraction"),
        ("unique_partitions_vs_k.png", [r["n_unique_partitions"] for r in summary_rows], "Unique Exact Partitions Among 200 Runs", "Number of unique partitions"),
        ("modal_vs_medoid_similarity_vs_k.png", [r["ari_mode_vs_medoid"] for r in summary_rows], "Similarity Between Modal and ARI-Medoid Partitions", "ARI: modal vs ARI medoid"),
    ]
    for filename, y, title, ylabel in plots:
        fig, ax = plt.subplots(figsize=(11, 6.4)); ax.plot(k, y, "o-", color="#1D4ED8", lw=2)
        style(ax, title, ylabel)
        if "fraction" in filename or "similarity" in filename: ax.set_ylim(-.02, 1.02)
        fig.tight_layout(); fig.savefig(OUTPUT / "figures" / filename, dpi=300); plt.close(fig)

    validation = {"source_directory": str(SOURCE), "k_values": list(KS), "n_k_values": 19,
        "seeds_per_k": 200, "total_label_vectors": total_vectors, "airfoils_per_vector": 1654,
        "all_3800_assignments_available": total_vectors == 3800,
        "kmeans_rerun": False, "canonicalization_tests": tests,
        "pairwise_ari_rows_consumed": pair_rows, "expected_pairwise_ari_rows": 378100,
        "each_seed_has_199_ari_comparisons": all(np.all(ari_counts[k] == 199) for k in KS),
        "all_medoid_mean_ari_finite": bool(np.isfinite([r["ari_medoid_mean_ari"] for r in summary_rows]).all()),
        "all_saved_assignments_length_1654": all(np.load(OUTPUT/"assignments"/f"modal_assignment_K{k:02d}.npy").shape == (1654,) for k in KS),
        "runtime_seconds": time.perf_counter() - started,
        "source_label_file_sha256": {f"K{k:02d}": sha256(SOURCE/"all_runs"/f"labels_k{k:02d}.npz") for k in KS}}
    (OUTPUT / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
