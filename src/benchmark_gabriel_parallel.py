"""Controlled process-count benchmark for Gabriel's frozen registration."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import resource
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import run_gabriel_k2_k8_energy_magnitude as production

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/gabriel_parallel_benchmark"
METRICS = (
    "final_pointwise_mse",
    "physical_dirichlet_integral",
    "velocity_mean_magnitude",
    "jacobian_minimum",
)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def run_one_with_resource(payload):
    row = production.run_one(payload)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is bytes on macOS and KiB on Linux.
    scale = 1 if os.uname().sysname == "Darwin" else 1024
    return row, int(usage.ru_maxrss * scale)


def benchmark_one(workers: int, contours, filenames, templates, assignments, indices):
    amap = {(int(a["K"]), int(a["target_index"])): int(a["cluster_id"]) for a in assignments}
    run_dir = OUT / "artifacts" / f"workers_{workers:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.glob("*.npz")):
        raise RuntimeError(f"benchmark artifact directory is not empty: {run_dir}")
    payloads = []
    for idx in indices:
        cluster = amap[(2, int(idx))]
        source, _, template_id = templates[(2, cluster)]
        artifact = run_dir / f"target_{idx:04d}.npz"
        payloads.append((2, cluster, int(idx), str(filenames[idx]), template_id,
                         source, contours[idx], str(artifact)))
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    rows = []
    worker_peak_rss = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one_with_resource, payload) for payload in payloads]
        for future in as_completed(futures):
            row, peak_rss = future.result()
            rows.append(row)
            worker_peak_rss.append(peak_rss)
    wall = time.perf_counter() - start
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu_seconds = ((usage_after.ru_utime + usage_after.ru_stime)
                         - (usage_before.ru_utime + usage_before.ru_stime))
    peak_worker_rss = max(worker_peak_rss, default=0)
    rows.sort(key=lambda row: int(row["target_index"]))
    atomic_csv(OUT / f"per_registration_workers_{workers}.csv", rows)
    failures = sum(row.get("status") != "completed" for row in rows)
    invalid = sum(not bool(row.get("valid_registration", False)) for row in rows)
    return rows, {
        "workers": workers,
        "n_registrations": len(rows),
        "wall_time_seconds": wall,
        "registrations_per_second": len(rows) / wall,
        "wall_seconds_per_registration": wall / len(rows),
        "average_aggregate_cpu_percent": 100 * child_cpu_seconds / wall,
        "approx_peak_total_memory_mib": peak_worker_rss * workers / 2**20,
        "peak_per_worker_memory_mib": peak_worker_rss / 2**20,
        "failures": failures,
        "invalid": invalid,
    }


def numerical_consistency(all_rows: dict[int, list[dict]]) -> dict:
    baseline = {int(r["target_index"]): r for r in all_rows[4]}
    details = []
    all_consistent = True
    for workers, rows in all_rows.items():
        if workers == 4:
            continue
        for row in rows:
            ref = baseline[int(row["target_index"])]
            for metric in METRICS:
                a, b = float(ref[metric]), float(row[metric])
                ok = bool(np.isclose(a, b, rtol=1e-10, atol=1e-12, equal_nan=True))
                all_consistent &= ok
                details.append({"workers": workers, "target_index": row["target_index"],
                                "metric": metric, "reference": a, "value": b,
                                "absolute_difference": abs(a-b), "consistent": ok})
    atomic_csv(OUT / "numerical_consistency.csv", details)
    return {"all_consistent": all_consistent,
            "rtol": 1e-10, "atol": 1e-12,
            "maximum_absolute_difference_by_metric": {
                metric: max((d["absolute_difference"] for d in details if d["metric"] == metric), default=0.0)
                for metric in METRICS}}


def make_figures(summary: list[dict]) -> None:
    workers = [r["workers"] for r in summary]
    for field, ylabel, title, filename, color in (
        ("registrations_per_second", "Registrations per second",
         "Gabriel Registration Throughput vs Number of Workers",
         "throughput_vs_workers.png", "#1f4e79"),
        ("wall_time_seconds", "Wall-clock time (seconds)",
         "Gabriel Registration Wall-clock Time vs Number of Workers",
         "wall_time_vs_workers.png", "#c45a18"),
    ):
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ax.plot(workers, [r[field] for r in summary], marker="o", lw=2, color=color)
        ax.set_xticks(workers)
        ax.set(xlabel="Number of workers", ylabel=ylabel, title=title)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", default="4,6,8,10")
    args = parser.parse_args()
    worker_counts = tuple(int(x) for x in args.workers.split(","))
    OUT.mkdir(parents=True, exist_ok=False)
    _, contours, filenames, templates, assignments, _, _ = production.prepare()
    indices = np.linspace(0, len(contours)-1, 40, dtype=int)
    if len(np.unique(indices)) != 40:
        raise RuntimeError("benchmark indices are not unique")
    np.save(OUT / "target_indices.npy", indices)
    (OUT / "benchmark_config.json").write_text(json.dumps({
        "workers": worker_counts,
        "target_indices": indices.tolist(),
        "K": 2,
        "scientific_config": production.asdict(production.CONFIG),
        "diagnostic_resolution": production.DIAGNOSTIC_RESOLUTION,
        "thread_environment": {name: os.environ.get(name) for name in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "worker_torch_threads": 1,
        "worker_torch_interop_threads": 1,
    }, indent=2) + "\n", encoding="utf-8")
    summary, all_rows = [], {}
    for workers in worker_counts:
        rows, result = benchmark_one(workers, contours, filenames, templates,
                                     assignments, indices)
        all_rows[workers] = rows
        summary.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    base = summary[0]
    for result in summary:
        result["speedup_vs_4"] = base["wall_time_seconds"] / result["wall_time_seconds"]
        result["parallel_efficiency_vs_4"] = result["speedup_vs_4"] / (result["workers"] / 4)
    atomic_csv(OUT / "parallel_benchmark.csv", summary)
    consistency = numerical_consistency(all_rows)
    make_figures(summary)
    (OUT / "validation.json").write_text(json.dumps({
        "same_40_indices_all_runs": True,
        "same_K2_templates_and_assignments": True,
        "production_checkpoint_directory_untouched_by_benchmark": True,
        "numerical_consistency": consistency,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
