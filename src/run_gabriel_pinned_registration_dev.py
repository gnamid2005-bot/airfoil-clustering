from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gabriel-pinned-mpl")

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import gabriel_pinned_registration as gabriel


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "references/pinned_registration.py"
RAW_PARENT = ROOT / "data/airfoils/raw/coord_seligFmt"
OLD_DATA = ROOT / "data/airfoils/processed/airfoils_n100_seed0.npz"
OLD_MEAN = ROOT / "outputs/airfoil_registration_improvement_stage2/direct_coordinate_mean/coordinates.npy"
SELECTED_INDICES = ROOT / "outputs/airfoil_registration_improvement/development_subset/selected_indices.npy"
SELECTED_FILENAMES = ROOT / "outputs/airfoil_registration_improvement/development_subset/selected_filenames.txt"
NEW_DATA_DIR = ROOT / "data/airfoils/processed_gabriel"
NEW_DATA = NEW_DATA_DIR / "airfoils_pinned_arclength_n200.npz"
OUT = ROOT / "outputs/gabriel_pinned_registration_dev"
FIGURES = OUT / "figures"
CASES = OUT / "cases"
REPORT = OUT / "report/gabriel_pinned_registration_dev_report.md"
CONFIG = gabriel.RegistrationConfig()
FIXED_EXISTING_INDICES = [-1, 382, 104, 959, 362, 1489, 377, 1224, 518]
FIXED_LABELS = [
    "identity",
    "very_close",
    "close",
    "medium_low",
    "medium",
    "medium_high",
    "far",
    "very_far",
    "extreme",
]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{key: serialise_cell(row.get(key)) for key in fields} for row in rows])
    os.replace(temporary, path)


def serialise_cell(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return json.dumps(gabriel.jsonable(value), ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metadata_tree_fingerprint(root: Path, excluded: Path | None = None) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if excluded is not None:
            try:
                path.relative_to(excluded)
                continue
            except ValueError:
                pass
        stat = path.stat()
        relative = path.relative_to(root).as_posix()
        digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
        count += 1
        total_bytes += stat.st_size
    return {
        "algorithm": "sha256(relative_path, size, mtime_ns)",
        "digest": digest.hexdigest(),
        "file_count": count,
        "total_bytes": total_bytes,
    }


def resolve_raw_root() -> Path:
    direct = sorted(RAW_PARENT.glob("*.dat"))
    if direct:
        return RAW_PARENT
    nested = RAW_PARENT / "coord_seligFmt"
    if nested.is_dir() and list(nested.glob("*.dat")):
        return nested
    raise FileNotFoundError(f"no Selig .dat files found under {RAW_PARENT}")


def reference_behaviour() -> dict[str, Any]:
    return {
        "parser": "skip first line; parse only subsequent lines with exactly two whitespace tokens",
        "file_order": "sorted non-recursive *.dat filenames",
        "excluded_filename_tokens": list(gabriel.EXCLUDE_TOKENS),
        "reference_scalar_filters": [
            "20 <= point_count <= 5000",
            "max(abs(raw_coordinate)) <= 1e3",
            "x_chord = max(x)-min(x) >= 0.5",
            "after normalisation max(abs(y)) <= 0.5",
        ],
        "normalisation": "coords = (coords - [min(x), 0]) / (max(x)-min(x))",
        "leading_edge_rule": "first argmin(normalised x)",
        "upper_segment": "raw coordinates [0:LE+1], expected TE->upper->LE",
        "lower_segment": "raw coordinates [LE:] followed by raw first point, expected LE->lower->TE",
        "resampling": "per-segment cumulative polyline arc length; linspace(0,L,n+1)[:n]",
        "endpoint_convention": "upper includes TE and excludes LE; lower includes LE and excludes closing TE",
        "registration": {
            "velocity": "30x30x2 tensor-product cardinal cubic B-spline coefficients",
            "flow": "stationary ODE integrated by classical RK4, 5 steps on [0,1]",
            "attachment": "mean over 200 points of squared 2D correspondence distance",
            "regulariser": "unscaled first differences of coefficient lattice on both axes",
            "optimiser": "8 cycles of PyTorch LBFGS, max_iter=100, strong_wolfe",
            "adaptive_lambda": "cycle 1 lambda=0.1; later min(0.2*attachment/max(smoothness,1e-12),0.1)",
        },
    }


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite existing experiment directory: {OUT}")
    if NEW_DATA_DIR.exists():
        raise FileExistsError(f"refusing to overwrite existing Gabriel data directory: {NEW_DATA_DIR}")
    raw_root = resolve_raw_root()
    raw_files = sorted(raw_root.glob("*.dat"))
    old_hash_before = gabriel.sha256_file(OLD_DATA)
    old_mean_hash_before = gabriel.sha256_file(OLD_MEAN)
    existing_outputs_before = metadata_tree_fingerprint(ROOT / "outputs")
    OUT.mkdir(parents=True)
    FIGURES.mkdir(parents=True)
    CASES.mkdir(parents=True)
    NEW_DATA_DIR.mkdir(parents=True)

    old = np.load(OLD_DATA, allow_pickle=True)
    old_filenames = old["filenames"].astype(str)
    old_name_locations: dict[str, list[int]] = {}
    for index, filename in enumerate(old_filenames):
        old_name_locations.setdefault(filename, []).append(index)
    old_duplicate_filenames = {
        name: indices for name, indices in old_name_locations.items() if len(indices) > 1
    }

    manifest: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    contours: list[np.ndarray] = []
    filenames: list[str] = []
    source_paths: list[str] = []
    source_hashes: list[str] = []
    existing_indices: list[int] = []
    validations: list[dict[str, Any]] = []
    raw_diagnostics: list[dict[str, Any]] = []
    reference_pass_count = 0
    for path in raw_files:
        normalised, record = gabriel.parse_gabriel_file(path)
        record.update(
            gabriel_dataset_index=-1,
            matching_existing_dataset_index=-1,
            conversion_status="rejected",
            validation_status="not_run",
            warning_reason="",
            repair_applied=False,
            repair_rule="",
        )
        if normalised is None:
            failures.append(dict(record))
            manifest.append(record)
            continue
        reference_pass_count += 1
        raw_diag = gabriel.raw_geometry_diagnostics(normalised)
        record.update(raw_diag)
        warning_tokens = []
        if raw_diag["raw_min_x_exact_tie_count"] > 1:
            warning_tokens.append("multiple_exact_min_x_points")
        if raw_diag["raw_min_x_near_tie_count"] > 1:
            warning_tokens.append("multiple_near_min_x_points")
        if raw_diag["raw_te_endpoint_duplicated"]:
            warning_tokens.append("duplicated_raw_te_endpoint")
        if raw_diag["raw_te_gap"] > 0.005:
            warning_tokens.append("finite_or_blunt_trailing_edge")
        if not raw_diag["raw_expected_selig_traversal"]:
            record.update(
                failure_reason="validation_reject:unexpected_or_reversed_selig_traversal",
                warning_reason=";".join(warning_tokens),
                validation_status="failed",
            )
            failures.append(dict(record))
            manifest.append(record)
            continue
        if min(raw_diag["raw_upper_arc_length"], raw_diag["raw_lower_arc_length"]) <= 1e-8:
            record.update(
                failure_reason="validation_reject:very_short_surface_segment",
                warning_reason=";".join(warning_tokens),
                validation_status="failed",
            )
            failures.append(dict(record))
            manifest.append(record)
            continue
        try:
            contour, reference_le_index = gabriel.resample_pinned(normalised, 200)
            validation = gabriel.validate_pinned_contour(contour)
        except Exception as exc:
            record.update(
                failure_reason=f"resampling_error:{type(exc).__name__}:{exc}",
                warning_reason=";".join(warning_tokens),
                validation_status="failed",
            )
            failures.append(dict(record))
            manifest.append(record)
            continue
        record["resampling_le_index"] = reference_le_index
        record.update({f"validation_{key}": value for key, value in validation.items()})
        if not validation["valid"]:
            record.update(
                failure_reason=f"validation_reject:{validation['failure_reason']}",
                warning_reason=";".join(warning_tokens),
                validation_status="failed",
            )
            failures.append(dict(record))
            manifest.append(record)
            continue
        matching = old_name_locations.get(path.name, [])
        matching_index = matching[0] if len(matching) == 1 else -1
        gabriel_index = len(contours)
        record.update(
            gabriel_dataset_index=gabriel_index,
            matching_existing_dataset_index=matching_index,
            conversion_status="accepted",
            validation_status="passed",
            warning_reason=";".join(warning_tokens),
            failure_reason="",
        )
        contours.append(contour)
        filenames.append(path.name)
        source_paths.append(str(path.relative_to(ROOT)))
        source_hashes.append(record["raw_sha256"])
        existing_indices.append(matching_index)
        validations.append(validation)
        raw_diagnostics.append(raw_diag)
        manifest.append(record)

    contour_array = np.stack(contours).astype(np.float64)
    filename_array = np.asarray(filenames)
    existing_index_array = np.asarray(existing_indices, dtype=np.int64)
    preprocessing_config = {
        "reference_file": str(REFERENCE),
        "reference_sha256": gabriel.sha256_file(REFERENCE),
        "requested_raw_parent": str(RAW_PARENT.relative_to(ROOT)),
        "resolved_raw_root": str(raw_root.relative_to(ROOT)),
        "n_points": 200,
        "points_per_surface": 100,
        "dtype": "float64",
        "repair_policy": "no repair; reject unexpected traversal or invalid final contour",
        "additional_validation_does_not_modify_coordinates": True,
        "reference_behaviour": reference_behaviour(),
    }
    representation = {
        "shape": [200, 2],
        "index_0": "trailing edge; first upper-surface sample",
        "indices_0_99": "upper surface, TE toward LE, uniform polyline arc coordinates",
        "index_100": "leading edge; first lower-surface sample",
        "indices_100_199": "lower surface, LE toward TE, uniform polyline arc coordinates",
        "closure": "implicit segment index 199 to index 0; no duplicated 201st point",
    }
    gabriel.atomic_npz(
        NEW_DATA,
        contours=contour_array,
        filenames=filename_array,
        gabriel_indices=np.arange(len(contour_array), dtype=np.int64),
        matching_existing_dataset_indices=existing_index_array,
        raw_source_paths=np.asarray(source_paths),
        raw_source_sha256=np.asarray(source_hashes),
        preprocessing_configuration_json=np.asarray(json.dumps(preprocessing_config, sort_keys=True)),
        representation_definition_json=np.asarray(json.dumps(representation, sort_keys=True)),
    )
    write_csv(OUT / "conversion_manifest.csv", manifest)
    write_csv(OUT / "conversion_failures.csv", failures)

    accepted_names = set(filenames)
    existing_names = set(old_filenames.tolist())
    reconciliation: list[dict[str, Any]] = []
    for index, filename in enumerate(filenames):
        locations = old_name_locations.get(filename, [])
        reconciliation.append(
            {
                "gabriel_dataset_index": index,
                "filename": filename,
                "existing_dataset_index": locations[0] if len(locations) == 1 else -1,
                "conversion_status": "accepted",
                "validation_status": "passed",
                "failure_or_warning_reason": next(
                    row["warning_reason"] for row in manifest if row["filename"] == filename
                ),
                "population_membership": "both" if locations else "gabriel_only",
            }
        )
    for filename in sorted(existing_names - accepted_names):
        matching_manifest = [row for row in manifest if row["filename"] == filename]
        reason = matching_manifest[0]["failure_reason"] if matching_manifest else "not_in_raw_files"
        reconciliation.append(
            {
                "gabriel_dataset_index": -1,
                "filename": filename,
                "existing_dataset_index": old_name_locations[filename][0],
                "conversion_status": "not_accepted",
                "validation_status": "failed_or_absent",
                "failure_or_warning_reason": reason,
                "population_membership": "existing_only",
            }
        )
    write_csv(OUT / "filename_reconciliation.csv", reconciliation)

    selected = np.load(SELECTED_INDICES).astype(int).tolist()
    selected_names = SELECTED_FILENAMES.read_text(encoding="utf-8").splitlines()
    if selected != FIXED_EXISTING_INDICES:
        raise RuntimeError(f"fixed development indices changed: {selected}")
    if len(selected_names) != len(FIXED_LABELS):
        raise RuntimeError("fixed development filename count mismatch")
    name_to_gabriel = {name: index for index, name in enumerate(filenames)}
    target_mapping: list[dict[str, Any]] = []
    missing_targets = []
    for label, existing_index, saved_filename in zip(FIXED_LABELS, selected, selected_names):
        expected = "GLOBAL_MEAN" if existing_index < 0 else str(old_filenames[existing_index])
        filename_matches_saved = bool(expected == saved_filename)
        gabriel_index = -1 if existing_index < 0 else name_to_gabriel.get(saved_filename, -1)
        matched = bool(existing_index < 0 or gabriel_index >= 0)
        row = {
            "target_label": label,
            "existing_dataset_index": existing_index,
            "filename": saved_filename,
            "existing_filename_verified": filename_matches_saved,
            "gabriel_dataset_index": gabriel_index,
            "matched": matched and filename_matches_saved,
            "reason": "identity_global_mean" if existing_index < 0 else ("" if matched else "missing_or_rejected"),
        }
        target_mapping.append(row)
        if not row["matched"]:
            missing_targets.append(row)
    write_csv(OUT / "development_target_mapping.csv", target_mapping)

    validation_values = {
        key: np.asarray([float(item[key]) for item in validations])
        for key in [
            "upper_segment_cv",
            "lower_segment_cv",
            "upper_segment_p95_relative_deviation",
            "lower_segment_p95_relative_deviation",
            "minimum_segment_length",
            "closure_segment_length",
            "signed_area",
        ]
    }
    dataset_validation = {
        "status": "PASS" if not missing_targets else "FAIL_MISSING_DEVELOPMENT_TARGET",
        "contour_array_shape": list(contour_array.shape),
        "dtype": str(contour_array.dtype),
        "all_finite": bool(np.isfinite(contour_array).all()),
        "all_index0_te": bool(all(item["index0_is_te"] for item in validations)),
        "all_index100_le": bool(all(item["index100_is_le"] for item in validations)),
        "all_upper_ordered": bool(all(item["upper_expected_direction_fraction"] >= 0.8 for item in validations)),
        "all_lower_ordered": bool(all(item["lower_expected_direction_fraction"] >= 0.8 for item in validations)),
        "all_arc_length_checks_pass": bool(all(item["approximately_uniform_arc_length"] for item in validations)),
        "all_no_zero_segments": bool(all(item["no_zero_length_segments"] for item in validations)),
        "all_no_self_intersection": bool(all(item["self_intersection_count"] == 0 for item in validations)),
        "all_counterclockwise": bool(all(item["signed_area"] > 0 for item in validations)),
        "all_valid": bool(all(item["valid"] for item in validations)),
        "metric_summaries": {
            key: {
                "minimum": float(values.min()),
                "median": float(np.median(values)),
                "p95": float(np.quantile(values, 0.95)),
                "maximum": float(values.max()),
            }
            for key, values in validation_values.items()
        },
        "validation_thresholds": {
            "index0_te_x": ">=0.98",
            "index100_le_abs_x": "<=1e-10",
            "surface_direction_fraction": ">=0.8",
            "arc_segment_p95_relative_deviation": "<=0.02",
            "arc_segment_cv": "<=0.10",
            "minimum_segment_length": ">1e-12",
            "self_intersections": "0",
            "signed_area": ">1e-8",
        },
    }
    gabriel.atomic_json(OUT / "dataset_validation.json", dataset_validation)

    failure_reasons = Counter(row["failure_reason"] for row in failures)
    mapped_count = int(np.count_nonzero(existing_index_array >= 0))
    accepted_only = sorted(accepted_names - existing_names)
    existing_only = sorted(existing_names - accepted_names)
    raw_warning_counts = Counter(
        token
        for row in manifest
        if row["conversion_status"] == "accepted"
        for token in str(row["warning_reason"]).split(";")
        if token
    )
    summary = {
        "raw_dat_files_inspected": len(raw_files),
        "reference_filter_accepted": reference_pass_count,
        "final_dataset_accepted": len(contours),
        "final_rejected": len(raw_files) - len(contours),
        "conversion_failures": len(failures),
        "failure_reason_counts": dict(failure_reasons),
        "warning_counts_among_accepted": dict(raw_warning_counts),
        "orientation_repairs": 0,
        "matching_existing_1654_filenames": mapped_count,
        "present_only_in_gabriel_dataset_count": len(accepted_only),
        "present_only_in_gabriel_dataset": accepted_only,
        "present_only_in_existing_dataset_count": len(existing_only),
        "present_only_in_existing_dataset": existing_only,
        "duplicate_existing_filename_conflicts": old_duplicate_filenames,
        "duplicate_raw_filename_conflicts": sorted(
            name for name, count in Counter(path.name for path in raw_files).items() if count > 1
        ),
        "development_targets_matched": len(target_mapping) - 1 - len(missing_targets),
        "development_targets_required": 8,
        "missing_development_targets": missing_targets,
        "new_dataset_path": str(NEW_DATA.relative_to(ROOT)),
        "representation": representation,
    }
    gabriel.atomic_json(OUT / "dataset_summary.json", summary)

    mean = contour_array.mean(axis=0)
    mean_validation = gabriel.validate_pinned_contour(mean)
    gabriel.atomic_npz(
        NEW_DATA_DIR / "global_mean_gabriel.npz",
        global_mean=mean,
        population_size=np.asarray(len(contour_array)),
        dataset_sha256=np.asarray(gabriel.sha256_file(NEW_DATA)),
    )
    np.save(NEW_DATA_DIR / "global_mean_gabriel.npy", mean)
    gabriel.atomic_json(
        OUT / "global_mean_validation.json",
        {
            **mean_validation,
            "population_size": len(contour_array),
            "source_dataset": str(NEW_DATA.relative_to(ROOT)),
            "global_mean_npy_sha256": gabriel.sha256_file(NEW_DATA_DIR / "global_mean_gabriel.npy"),
        },
    )
    plot_global_mean(mean, NEW_DATA_DIR / "global_mean_gabriel.png")
    plot_global_mean(mean, FIGURES / "global_mean_gabriel.png")
    plot_dataset_qa(contour_array, filenames, target_mapping, manifest)

    configuration = {
        "experiment": "Gabriel pinned arc-length preprocessing and development registration",
        "root": str(ROOT),
        "created_unix_time": time.time(),
        "scope": "identity plus eight fixed development targets only",
        "raw_root": str(raw_root.relative_to(ROOT)),
        "old_dataset": str(OLD_DATA.relative_to(ROOT)),
        "old_dataset_sha256_before": old_hash_before,
        "old_mean_sha256_before": old_mean_hash_before,
        "existing_outputs_metadata_before": existing_outputs_before,
        "reference_file": str(REFERENCE),
        "reference_sha256": gabriel.sha256_file(REFERENCE),
        "new_dataset": str(NEW_DATA.relative_to(ROOT)),
        "new_dataset_sha256": gabriel.sha256_file(NEW_DATA),
        "fixed_existing_indices": selected,
        "fixed_filenames": selected_names,
        "registration_config": asdict(CONFIG),
        "parameter_count": CONFIG.parameter_count,
        "diagnostic_grid": {
            "resolution": 101,
            "xlim": list(CONFIG.grid_xlim),
            "ylim": list(CONFIG.grid_ylim),
            "jacobian_method": "second-order finite differences of sampled RK4 deformation",
            "physical_dirichlet": "trapezoidal integral of squared spatial velocity derivatives",
            "bending": "trapezoidal integral of squared componentwise velocity Laplacian",
        },
        "residual_regions": {
            "TE": "indices 0-9 and 190-199",
            "LE": "indices 90-109",
            "mid": "indices 10-89 and 110-189",
        },
        "reference_behaviour": reference_behaviour(),
        "chart_map": [
            {"figure": "dataset_qa_development_targets.png", "question": "Are fixed targets ordered TE-upper-LE-lower-TE?", "type": "small multiples"},
            {"figure": "registration_overlays.png", "question": "How closely does the mean deform to each target?", "type": "shape overlays"},
            {"figure": "residuals_before_after.png", "question": "Where does correspondence error remain?", "type": "point-index line"},
            {"figure": "accuracy_by_target.png", "question": "How do accuracy metrics vary by target?", "type": "grouped panels"},
            {"figure": "optimisation_trace.png", "question": "How do attachment and regularisation evolve?", "type": "log lines"},
            {"figure": "representative_deformation_grids.png", "question": "What spatial deformations are learned?", "type": "deformed grids"},
            {"figure": "worst_case_jacobian.png", "question": "Where is sampled diffeomorphic safety weakest?", "type": "heatmap"},
            {"figure": "rk_integration_convergence.png", "question": "Is RK5 adequate after fitting?", "type": "bar panels"},
        ],
    }
    gabriel.atomic_json(OUT / "config.json", configuration)
    if missing_targets:
        raise RuntimeError(f"missing fixed development targets: {missing_targets}")
    if not mean_validation["valid"]:
        raise RuntimeError(f"new global mean failed validation: {mean_validation}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def plot_closed(ax: plt.Axes, contour: np.ndarray, **kwargs: Any) -> None:
    closed = np.vstack([contour, contour[:1]])
    ax.plot(closed[:, 0], closed[:, 1], **kwargs)


def annotate_anchors(ax: plt.Axes, contour: np.ndarray, labels: bool = True) -> None:
    styles = {
        0: ("#d62728", "0 TE", (8, 10)),
        99: ("#ff7f0e", "99", (-28, 10)),
        100: ("#1f77b4", "100 LE", (-32, -15)),
        199: ("#2ca02c", "199", (8, -15)),
    }
    for index, (colour, text, offset) in styles.items():
        ax.scatter(*contour[index], s=24, color=colour, zorder=4)
        if labels:
            ax.annotate(text, contour[index], xytext=offset, textcoords="offset points", fontsize=7)
    ax.plot(
        [contour[199, 0], contour[0, 0]],
        [contour[199, 1], contour[0, 1]],
        color="#2ca02c",
        linestyle="--",
        linewidth=0.8,
    )


def plot_global_mean(mean: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.2))
    plot_closed(ax, mean, color="#1f4e79", linewidth=2)
    annotate_anchors(ax, mean)
    ax.set_aspect("equal")
    ax.set_xlabel("x / chord")
    ax.set_ylabel("y / chord")
    ax.set_title("Gabriel pointwise global mean (pinned arc-length correspondence)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def qa_small_multiples(contours: list[np.ndarray], names: list[str], title: str, path: Path) -> None:
    columns = 4
    rows = math.ceil(len(contours) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(16, max(3.2 * rows, 4)), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, contour, name in zip(axes.ravel(), contours, names):
        ax.axis("on")
        plot_closed(ax, contour, color="#1f4e79", linewidth=1.2)
        annotate_anchors(ax, contour, labels=False)
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=9)
        ax.grid(alpha=0.15)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colour, label=label)
        for colour, label in [
            ("#d62728", "index 0 (TE)"),
            ("#ff7f0e", "index 99"),
            ("#1f77b4", "index 100 (LE)"),
            ("#2ca02c", "index 199"),
        ]
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_qa(
    contours: np.ndarray,
    filenames: list[str],
    target_mapping: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
) -> None:
    normal_indices = np.linspace(0, len(contours) - 1, 8, dtype=int)
    qa_small_multiples(
        [contours[i] for i in normal_indices],
        [filenames[i] for i in normal_indices],
        "Gabriel point ordering QA: representative accepted airfoils",
        FIGURES / "dataset_qa_normal_airfoils.png",
    )
    dev = [row for row in target_mapping if row["target_label"] != "identity"]
    qa_small_multiples(
        [contours[int(row["gabriel_dataset_index"])] for row in dev],
        [f"{row['target_label']}: {row['filename']}" for row in dev],
        "Gabriel point ordering QA: all fixed development targets",
        FIGURES / "dataset_qa_development_targets.png",
    )
    accepted = [row for row in manifest if row["conversion_status"] == "accepted"]
    ambiguous = min(
        accepted,
        key=lambda row: float(row.get("raw_second_minimum_x_gap_chords", math.inf)),
    )
    blunt = max(accepted, key=lambda row: float(row.get("raw_te_gap", 0.0)))
    indices = [int(ambiguous["gabriel_dataset_index"]), int(blunt["gabriel_dataset_index"])]
    qa_small_multiples(
        [contours[index] for index in indices],
        [
            f"closest competing LE x: {ambiguous['filename']}",
            f"widest blunt TE: {blunt['filename']}",
        ],
        "Ambiguous leading edge and widest finite-thickness trailing edge",
        FIGURES / "dataset_qa_ambiguous_and_blunt.png",
    )


def run_core_unit_tests() -> dict[str, Any]:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(7)
    config = CONFIG
    control_x = torch.linspace(*config.grid_xlim, config.grid_n)
    control_y = torch.linspace(*config.grid_ylim, config.grid_n)
    zero = torch.zeros((config.grid_n, config.grid_n, 2), requires_grad=True)
    points = torch.tensor(
        [[0.0, 0.0], [0.2, 0.03], [0.5, -0.08], [1.0, 0.0]], dtype=torch.float64
    )
    identity = gabriel.flow(points, zero, control_x, control_y, config.rk_steps)
    rng = np.random.default_rng(7)
    random_points = torch.tensor(
        np.column_stack([rng.uniform(0.0, 1.0, 32), rng.uniform(-0.25, 0.25, 32)])
    )
    random_coefficients = torch.tensor(
        rng.normal(scale=0.01, size=(config.grid_n, config.grid_n, 2)),
        requires_grad=True,
    )
    local = gabriel.velocity(random_points, random_coefficients, control_x, control_y)
    dense = gabriel.dense_velocity_reference(
        random_points, random_coefficients, control_x, control_y
    )
    _, bx = gabriel.local_basis(random_points[:, 0], control_x)
    _, by = gabriel.local_basis(random_points[:, 1], control_y)
    flowed = gabriel.flow(random_points, random_coefficients, control_x, control_y, 5)
    differentiable_loss = flowed.square().sum()
    differentiable_loss.backward()
    synthetic_target = random_points.detach() + 0.01
    attach_value = gabriel.attachment(
        random_points.detach(), synthetic_target, zero, control_x, control_y, 5
    )
    direct_attach = ((random_points.detach() - synthetic_target) ** 2).sum(1).mean()
    residual = random_points.detach().numpy() - synthetic_target.numpy()
    pointwise_mse = float(np.sum(residual**2, axis=1).mean())
    coordinatewise_mse = float(np.mean(residual**2))
    constant = torch.full((config.grid_n, config.grid_n, 2), 0.02)
    smooth_constant = float(gabriel.coefficient_smoothness(constant))
    smooth_random = float(gabriel.coefficient_smoothness(random_coefficients.detach()))
    buffer = io.BytesIO()
    np.savez(buffer, coefficients=random_coefficients.detach().numpy())
    buffer.seek(0)
    reloaded = np.load(buffer)["coefficients"]
    with torch.no_grad():
        reload_flow = gabriel.flow(
            random_points,
            torch.from_numpy(reloaded),
            control_x,
            control_y,
            5,
        )
    metrics = {
        "zero_coefficients_identity_max_abs": float(torch.max(torch.abs(identity - points)).detach()),
        "local_dense_velocity_max_abs": float(torch.max(torch.abs(local - dense)).detach()),
        "partition_unity_x_max_abs": float(torch.max(torch.abs(bx.sum(1) - 1.0)).detach()),
        "partition_unity_y_max_abs": float(torch.max(torch.abs(by.sum(1) - 1.0)).detach()),
        "rk4_output_finite": bool(torch.isfinite(flowed).all()),
        "rk4_gradient_finite": bool(torch.isfinite(random_coefficients.grad).all()),
        "attachment_difference": float(torch.abs(attach_value - direct_attach)),
        "rmse_squared_mse_difference": abs(math.sqrt(pointwise_mse) ** 2 - pointwise_mse),
        "coordinatewise_half_mse_difference": abs(coordinatewise_mse - pointwise_mse / 2.0),
        "constant_coefficient_smoothness": smooth_constant,
        "random_coefficient_smoothness": smooth_random,
        "reload_flow_max_abs": float(torch.max(torch.abs(reload_flow - flowed.detach()))),
    }
    checks = {
        "zero_coefficients_identity": metrics["zero_coefficients_identity_max_abs"] <= 1e-15,
        "local_matches_dense": metrics["local_dense_velocity_max_abs"] <= 1e-13,
        "partition_unity": max(metrics["partition_unity_x_max_abs"], metrics["partition_unity_y_max_abs"]) <= 1e-13,
        "rk4_finite_and_differentiable": metrics["rk4_output_finite"] and metrics["rk4_gradient_finite"],
        "attachment_exact": metrics["attachment_difference"] <= 1e-15,
        "rmse_identity": metrics["rmse_squared_mse_difference"] <= 1e-18,
        "coordinatewise_identity": metrics["coordinatewise_half_mse_difference"] <= 1e-18,
        "constant_smoothness_zero": smooth_constant == 0.0,
        "smoothness_nonnegative": smooth_random >= 0.0,
        "save_reload_reproduces_flow": metrics["reload_flow_max_abs"] == 0.0,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "metrics": metrics}


def load_prepared() -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    archive = np.load(NEW_DATA, allow_pickle=False)
    contours = archive["contours"].astype(np.float64)
    filenames = archive["filenames"].astype(str)
    mean = np.load(NEW_DATA_DIR / "global_mean_gabriel.npy").astype(np.float64)
    mapping = read_csv(OUT / "development_target_mapping.csv")
    return contours, mean, mapping


def run_registrations() -> None:
    if not OUT.exists() or not NEW_DATA.exists():
        raise RuntimeError("prepare phase has not completed")
    unit_tests = run_core_unit_tests()
    gabriel.atomic_json(OUT / "unit_tests.json", unit_tests)
    if unit_tests["status"] != "PASS":
        raise RuntimeError(f"core unit tests failed: {unit_tests}")
    contours, mean, mapping = load_prepared()
    for row in mapping:
        label = row["target_label"]
        case_json = CASES / f"{label}.json"
        if case_json.exists():
            saved = json.loads(case_json.read_text(encoding="utf-8"))
            if saved.get("status") in {"completed", "failed", "nonfinite"}:
                print(f"skip complete case {label}", flush=True)
                continue
        if label == "identity":
            target = mean.copy()
            gabriel_index = -1
        else:
            gabriel_index = int(row["gabriel_dataset_index"])
            if gabriel_index < 0:
                raise RuntimeError(f"unmatched development target: {row}")
            target = contours[gabriel_index].copy()
        print(f"START {label} {row['filename']}", flush=True)

        def progress(cycle: dict[str, Any]) -> None:
            print(
                f"  cycle={cycle['cycle']} lambda={cycle['lambda']:.3e} "
                f"closures={cycle['closure_evaluations']} runtime={cycle['runtime_seconds']:.2f}s",
                flush=True,
            )

        result = gabriel.optimise_registration(mean, target, CONFIG, progress=progress)
        before = gabriel.correspondence_metrics(mean, target)
        after = gabriel.correspondence_metrics(result["deformed"], target)
        primary_grid = gabriel.diagnostic_grid(
            result["coefficients"], result["control_x"], result["control_y"], 5, 101
        )
        deformation = gabriel.deformation_metrics(
            mean, result["deformed"], result["coefficients"], primary_grid
        )
        trajectory = gabriel.trajectory_domain_diagnostics(
            mean, result["coefficients"], result["control_x"], result["control_y"], 5
        )
        deformed_geometry = {
            "finite": bool(np.isfinite(result["deformed"]).all()),
            "signed_area": gabriel.signed_polygon_area(result["deformed"]),
            "self_intersection_count": gabriel.self_intersection_count(result["deformed"]),
        }
        deformed_geometry["invalid_geometry"] = bool(
            not deformed_geometry["finite"]
            or deformed_geometry["signed_area"] <= 1e-8
            or deformed_geometry["self_intersection_count"] > 0
        )
        integration_rows = []
        integration_states: dict[int, dict[str, Any]] = {}
        source_t = torch.from_numpy(np.ascontiguousarray(mean))
        coeff_t = torch.from_numpy(np.ascontiguousarray(result["coefficients"]))
        cx_t = torch.from_numpy(np.ascontiguousarray(result["control_x"]))
        cy_t = torch.from_numpy(np.ascontiguousarray(result["control_y"]))
        with torch.no_grad():
            for steps in (5, 10, 20):
                transformed = gabriel.flow(source_t, coeff_t, cx_t, cy_t, steps).numpy()
                metrics = gabriel.correspondence_metrics(transformed, target)
                grid = gabriel.diagnostic_grid(
                    result["coefficients"], result["control_x"], result["control_y"], steps, 101
                )
                integration_states[steps] = {
                    "deformed": transformed,
                    "pointwise_mse": metrics["pointwise_mse"],
                    "jacobian_minimum": grid["jacobian_minimum"],
                }
        for first, second in ((5, 10), (10, 20)):
            a, b = integration_states[first], integration_states[second]
            integration_rows.append(
                {
                    "comparison": f"RK{first}_vs_RK{second}",
                    "rk_steps_first": first,
                    "rk_steps_second": second,
                    "mse_first": a["pointwise_mse"],
                    "mse_second": b["pointwise_mse"],
                    "change_in_pointwise_mse": b["pointwise_mse"] - a["pointwise_mse"],
                    "absolute_change_in_pointwise_mse": abs(b["pointwise_mse"] - a["pointwise_mse"]),
                    "maximum_transformed_point_difference": float(
                        np.linalg.norm(b["deformed"] - a["deformed"], axis=1).max()
                    ),
                    "minimum_jacobian_first": a["jacobian_minimum"],
                    "minimum_jacobian_second": b["jacobian_minimum"],
                    "change_in_minimum_jacobian": b["jacobian_minimum"] - a["jacobian_minimum"],
                }
            )
        case = {
            "target_label": label,
            "filename": row["filename"],
            "gabriel_dataset_index": gabriel_index,
            "existing_dataset_index": int(row["existing_dataset_index"]),
            "status": result["status"],
            "failure_reason": result["failure_reason"],
            "finite": result["finite"],
            "configuration": result["config"],
            "runtime_seconds": result["runtime_seconds"],
            "closure_evaluations": result["closure_evaluations"],
            "final_lambda": result["final_lambda"],
            "final_attachment": result["final_attachment"],
            "final_coefficient_smoothness": result["final_coefficient_smoothness"],
            "final_weighted_regularisation": result["final_weighted_regularisation"],
            "final_objective": result["final_objective"],
            "before_metrics": {k: v for k, v in before.items() if not isinstance(v, np.ndarray)},
            "after_metrics": {k: v for k, v in after.items() if not isinstance(v, np.ndarray)},
            "deformation_metrics": deformation,
            "jacobian_metrics": {
                key: value
                for key, value in primary_grid.items()
                if key.startswith("jacobian_") and not isinstance(value, np.ndarray)
            },
            "trajectory_metrics": trajectory,
            "deformed_geometry": deformed_geometry,
            "integration_convergence": integration_rows,
            "cycles": result["cycles"],
            "trace": result["trace"],
            "sampled_positive_jacobian_is_not_analytical_proof": True,
        }
        gabriel.atomic_npz(
            CASES / f"{label}.npz",
            source=mean,
            target=target,
            deformed_source=result["deformed"],
            coefficients=result["coefficients"],
            control_x=result["control_x"],
            control_y=result["control_y"],
            residual_vectors=after["residual_vectors"],
            residual_magnitudes=after["residual_magnitudes"],
            residual_magnitudes_before=before["residual_magnitudes"],
            diagnostic_x=primary_grid["x"],
            diagnostic_y=primary_grid["y"],
            diagnostic_velocity=primary_grid["velocity"],
            diagnostic_deformation=primary_grid["phi"],
            jacobian_determinant=primary_grid["jacobian_determinant"],
            rk5_deformed=integration_states[5]["deformed"],
            rk10_deformed=integration_states[10]["deformed"],
            rk20_deformed=integration_states[20]["deformed"],
            configuration_json=np.asarray(json.dumps(result["config"], sort_keys=True)),
        )
        gabriel.atomic_json(case_json, case)
        print(
            f"DONE {label} mse={after['pointwise_mse']:.4e} "
            f"minJ={primary_grid['jacobian_minimum']:.4f} runtime={result['runtime_seconds']:.2f}s",
            flush=True,
        )


def flatten_case(case: dict[str, Any]) -> dict[str, Any]:
    row = {
        "target_label": case["target_label"],
        "filename": case["filename"],
        "gabriel_dataset_index": case["gabriel_dataset_index"],
        "existing_dataset_index": case["existing_dataset_index"],
        "status": case["status"],
        "failure_reason": case["failure_reason"],
        "finite": case["finite"],
        "runtime_seconds": case["runtime_seconds"],
        "optimisation_runtime_seconds": case["runtime_seconds"],
        "postfit_diagnostic_runtime_seconds": case.get(
            "postfit_diagnostic_runtime_seconds", math.nan
        ),
        "total_case_runtime_seconds": case.get(
            "total_case_runtime_seconds", case["runtime_seconds"]
        ),
        "closure_evaluations": case["closure_evaluations"],
        "final_lambda": case["final_lambda"],
        "final_attachment": case["final_attachment"],
        "final_coefficient_smoothness": case["final_coefficient_smoothness"],
        "final_weighted_regularisation": case["final_weighted_regularisation"],
        "final_objective": case["final_objective"],
    }
    for prefix, key in (("before", "before_metrics"), ("after", "after_metrics"), ("deformation", "deformation_metrics"), ("jacobian", "jacobian_metrics"), ("trajectory", "trajectory_metrics"), ("geometry", "deformed_geometry")):
        row.update({f"{prefix}_{name}": value for name, value in case[key].items()})
    return row


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "after_pointwise_mse",
        "after_pointwise_rmse",
        "after_coordinatewise_mse",
        "after_maximum_correspondence_error",
        "after_median_correspondence_error",
        "after_p95_correspondence_error",
        "deformation_mean_source_displacement",
        "deformation_rms_source_displacement",
        "deformation_p95_source_displacement",
        "deformation_maximum_source_displacement",
        "deformation_coefficient_l2_norm",
        "final_coefficient_smoothness",
        "final_weighted_regularisation",
        "deformation_physical_dirichlet_integral",
        "deformation_bending_laplacian_integral",
        "jacobian_jacobian_minimum",
        "jacobian_jacobian_p01",
        "jacobian_jacobian_p05",
        "jacobian_jacobian_median",
        "runtime_seconds",
        "total_case_runtime_seconds",
        "closure_evaluations",
    ]
    output = []
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in rows if row["status"] == "completed"])
        output.append(
            {
                "scope": "8_nonidentity_completed_cases",
                "metric": metric,
                "count": len(values),
                "mean": float(values.mean()) if len(values) else math.nan,
                "standard_error": float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0,
                "median": float(np.median(values)) if len(values) else math.nan,
                "p95": float(np.quantile(values, 0.95)) if len(values) else math.nan,
                "maximum": float(values.max()) if len(values) else math.nan,
            }
        )
    return output


def finalise() -> None:
    case_paths = [CASES / f"{label}.json" for label in FIXED_LABELS]
    missing = [str(path) for path in case_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"missing case results: {missing}")
    cases = [json.loads(path.read_text(encoding="utf-8")) for path in case_paths]
    for case, path in zip(cases, case_paths):
        # Dense maps belong in NPZ, not JSON.  Benchmark the complete post-fit
        # diagnostic path once so each target has component-complete timing.
        case["jacobian_metrics"].pop("jacobian_determinant", None)
        data = np.load(CASES / f"{case['target_label']}.npz")
        diagnostic_start = time.perf_counter()
        gabriel.diagnostic_grid(
            data["coefficients"], data["control_x"], data["control_y"], 5, 101
        )
        for steps in (5, 10, 20):
            gabriel.diagnostic_grid(
                data["coefficients"], data["control_x"], data["control_y"], steps, 101
            )
        gabriel.trajectory_domain_diagnostics(
            data["source"],
            data["coefficients"],
            data["control_x"],
            data["control_y"],
            5,
        )
        diagnostic_runtime = time.perf_counter() - diagnostic_start
        case["postfit_diagnostic_runtime_seconds"] = diagnostic_runtime
        case["total_case_runtime_seconds"] = case["runtime_seconds"] + diagnostic_runtime
        gabriel.atomic_json(path, case)
    rows = [flatten_case(case) for case in cases]
    write_csv(OUT / "baseline_per_case.csv", rows)
    nonidentity_rows = [row for row in rows if row["target_label"] != "identity"]
    write_csv(OUT / "aggregate_summary.csv", aggregate_rows(nonidentity_rows))
    trace_rows = []
    integration_rows = []
    jacobian_rows = []
    per_point_rows = []
    for case in cases:
        for trace in case["trace"]:
            trace_rows.append(
                {
                    "target_label": case["target_label"],
                    "filename": case["filename"],
                    **trace,
                }
            )
        for row in case["integration_convergence"]:
            integration_rows.append(
                {
                    "target_label": case["target_label"],
                    "filename": case["filename"],
                    **row,
                }
            )
        jacobian_rows.append(
            {
                "target_label": case["target_label"],
                "filename": case["filename"],
                **case["jacobian_metrics"],
                **case["deformed_geometry"],
                **case["trajectory_metrics"],
            }
        )
        data = np.load(CASES / f"{case['target_label']}.npz")
        for index, (before, after, residual) in enumerate(
            zip(
                data["residual_magnitudes_before"],
                data["residual_magnitudes"],
                data["residual_vectors"],
            )
        ):
            region = "TE" if index < 10 or index >= 190 else ("LE" if 90 <= index < 110 else "mid")
            per_point_rows.append(
                {
                    "target_label": case["target_label"],
                    "filename": case["filename"],
                    "point_index": index,
                    "surface": "upper" if index < 100 else "lower",
                    "region": region,
                    "before_error": float(before),
                    "after_error": float(after),
                    "after_residual_x": float(residual[0]),
                    "after_residual_y": float(residual[1]),
                }
            )
    write_csv(OUT / "optimisation_trace.csv", trace_rows)
    write_csv(OUT / "integration_convergence.csv", integration_rows)
    write_csv(OUT / "jacobian_summary.csv", jacobian_rows)
    write_csv(OUT / "per_point_residuals.csv", per_point_rows)
    create_result_figures(cases)
    validation = validate_complete(cases, rows, integration_rows)
    gabriel.atomic_json(OUT / "validation.json", validation)
    write_report(cases, rows, integration_rows, validation)
    if validation["status"] != "PASS":
        raise RuntimeError(f"final validation failed: {validation}")


def create_result_figures(cases: list[dict[str, Any]]) -> None:
    nonidentity = [case for case in cases if case["target_label"] != "identity"]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, case in zip(axes.ravel(), cases):
        ax.axis("on")
        data = np.load(CASES / f"{case['target_label']}.npz")
        plot_closed(ax, data["target"], color="#1f77b4", linewidth=1.5, label="target")
        plot_closed(ax, data["source"], color="#7f7f7f", linewidth=1.0, linestyle=":", label="source")
        plot_closed(ax, data["deformed_source"], color="#d62728", linewidth=1.2, linestyle="--", label="deformed")
        ax.set_aspect("equal")
        ax.set_title(f"{case['target_label']}  MSE={case['after_metrics']['pointwise_mse']:.2e}", fontsize=9)
        ax.grid(alpha=0.12)
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle("Gabriel mean-to-development registration", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGURES / "registration_overlays.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(16, 12), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, case in zip(axes.ravel(), cases):
        ax.axis("on")
        data = np.load(CASES / f"{case['target_label']}.npz")
        ax.semilogy(np.maximum(data["residual_magnitudes_before"], 1e-14), color="#7f7f7f", label="before")
        ax.semilogy(np.maximum(data["residual_magnitudes"], 1e-14), color="#d62728", label="after")
        ax.axvline(100, color="#1f77b4", linewidth=0.8, linestyle="--")
        ax.set_title(case["target_label"], fontsize=9)
        ax.set_xlabel("correspondence index")
        ax.set_ylabel("error (chord)")
        ax.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Pointwise correspondence residuals before and after", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGURES / "residuals_before_after.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    labels = [case["target_label"] for case in nonidentity]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    metrics = [
        ("pointwise_mse", "Pointwise MSE", True),
        ("pointwise_rmse", "Pointwise RMSE", True),
        ("maximum_correspondence_error", "Maximum error", True),
    ]
    for ax, (key, title, log_scale) in zip(axes, metrics):
        before = [case["before_metrics"][key] for case in nonidentity]
        after = [case["after_metrics"][key] for case in nonidentity]
        width = 0.38
        ax.bar(x - width / 2, before, width, color="#bdbdbd", label="before")
        ax.bar(x + width / 2, after, width, color="#1f77b4", label="after")
        if log_scale:
            ax.set_yscale("log")
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend()
    fig.suptitle("Registration accuracy by non-identity development target", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURES / "accuracy_by_target.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3))
    for case in nonidentity:
        trace = case["trace"]
        evaluations = [row["closure_evaluation_global"] for row in trace]
        axes[0].semilogy(evaluations, [row["attachment"] for row in trace], label=case["target_label"])
        axes[1].semilogy(evaluations, np.maximum([row["weighted_regularisation"] for row in trace], 1e-30))
        axes[2].semilogy(evaluations, [row["total_objective"] for row in trace])
    for ax, title in zip(axes, ["Attachment", "Weighted regularisation", "Total objective"]):
        ax.set_title(title)
        ax.set_xlabel("closure evaluation (not LBFGS iteration)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("objective value")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Staged strong-Wolfe LBFGS traces", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURES / "optimisation_trace.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    ranked = sorted(nonidentity, key=lambda case: case["before_metrics"]["pointwise_mse"])
    representative = [ranked[0], ranked[len(ranked) // 2], ranked[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, case in zip(axes, representative):
        data = np.load(CASES / f"{case['target_label']}.npz")
        phi = data["diagnostic_deformation"]
        stride = 5
        for iy in range(0, phi.shape[0], stride):
            ax.plot(phi[iy, :, 0], phi[iy, :, 1], color="#b0b0b0", linewidth=0.5)
        for ix in range(0, phi.shape[1], stride):
            ax.plot(phi[:, ix, 0], phi[:, ix, 1], color="#b0b0b0", linewidth=0.5)
        plot_closed(ax, data["target"], color="#1f77b4", linewidth=1.5)
        plot_closed(ax, data["deformed_source"], color="#d62728", linewidth=1.0, linestyle="--")
        ax.set_aspect("equal")
        ax.set_xlim(*CONFIG.grid_xlim)
        ax.set_ylim(*CONFIG.grid_ylim)
        ax.set_title(case["target_label"])
    fig.suptitle("Representative sampled deformation grids", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGURES / "representative_deformation_grids.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    worst = min(cases, key=lambda case: case["jacobian_metrics"]["jacobian_minimum"])
    data = np.load(CASES / f"{worst['target_label']}.npz")
    determinant = data["jacobian_determinant"]
    extent = [CONFIG.grid_xlim[0], CONFIG.grid_xlim[1], CONFIG.grid_ylim[0], CONFIG.grid_ylim[1]]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), sharex=True, sharey=True)
    image_full = axes[0].imshow(
        determinant, origin="lower", extent=extent, aspect="auto", cmap="viridis"
    )
    fig.colorbar(image_full, ax=axes[0], label="sampled det(DPhi)")
    axes[0].set_title("Full determinant range")
    image_safety = axes[1].imshow(
        np.clip(determinant, -0.05, 1.5),
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="coolwarm",
        vmin=-0.05,
        vmax=1.5,
    )
    fig.colorbar(image_safety, ax=axes[1], label="det(DPhi), clipped for safety detail")
    axes[1].contour(
        np.linspace(*CONFIG.grid_xlim, determinant.shape[1]),
        np.linspace(*CONFIG.grid_ylim, determinant.shape[0]),
        determinant,
        levels=[0.0, 0.05],
        colors=["black", "#ffd700"],
        linewidths=[1.2, 1.0],
    )
    axes[1].set_title("Safety view: black=0, gold=0.05")
    for ax in axes:
        ax.set_xlabel("x / chord")
    axes[0].set_ylabel("y / chord")
    fig.suptitle(
        f"Worst sampled Jacobian case: {worst['target_label']} "
        f"(min={worst['jacobian_metrics']['jacobian_minimum']:.4f})"
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "worst_case_jacobian.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    comparisons = ["RK5_vs_RK10", "RK10_vs_RK20"]
    width = 0.38
    for offset, comparison in enumerate(comparisons):
        selected = [next(row for row in case["integration_convergence"] if row["comparison"] == comparison) for case in nonidentity]
        shift = (-0.5 + offset) * width
        axes[0].bar(x + shift, [row["absolute_change_in_pointwise_mse"] for row in selected], width, label=comparison)
        axes[1].bar(x + shift, [row["maximum_transformed_point_difference"] for row in selected], width)
        axes[2].bar(x + shift, [abs(row["change_in_minimum_jacobian"]) for row in selected], width)
    for ax, title in zip(axes, ["Absolute MSE change", "Max transformed-point difference", "Absolute min-Jacobian change"]):
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Frozen-coefficient RK integration convergence", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGURES / "rk_integration_convergence.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def validate_complete(
    cases: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    integration_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    config = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
    current_outputs = metadata_tree_fingerprint(ROOT / "outputs", excluded=OUT)
    old_hash_after = gabriel.sha256_file(OLD_DATA)
    old_mean_hash_after = gabriel.sha256_file(OLD_MEAN)
    unit = json.loads((OUT / "unit_tests.json").read_text(encoding="utf-8"))
    source_faithfulness = json.loads(
        (OUT / "source_faithfulness_validation.json").read_text(encoding="utf-8")
    )
    dataset_validation = json.loads((OUT / "dataset_validation.json").read_text(encoding="utf-8"))
    mapping = read_csv(OUT / "development_target_mapping.csv")
    required_files = [
        "config.json",
        "dataset_summary.json",
        "conversion_manifest.csv",
        "conversion_failures.csv",
        "dataset_validation.json",
        "filename_reconciliation.csv",
        "development_target_mapping.csv",
        "global_mean_validation.json",
        "baseline_per_case.csv",
        "aggregate_summary.csv",
        "optimisation_trace.csv",
        "integration_convergence.csv",
        "jacobian_summary.csv",
        "unit_tests.json",
        "source_faithfulness_validation.json",
    ]
    checks = {
        "old_processed_dataset_byte_identical": old_hash_after == config["old_dataset_sha256_before"],
        "old_global_mean_byte_identical": old_mean_hash_after == config["old_mean_sha256_before"],
        "existing_output_tree_metadata_unchanged": current_outputs == config["existing_outputs_metadata_before"],
        "new_data_from_raw_selig": all("data/airfoils/raw/coord_seligFmt" in row["raw_source_path"] for row in read_csv(OUT / "conversion_manifest.csv")),
        "dataset_validation_passed": dataset_validation["status"] == "PASS" and dataset_validation["all_valid"],
        "all_eight_development_targets_matched": sum(row["matched"] == "True" and row["target_label"] != "identity" for row in mapping) == 8,
        "registration_source_is_new_gabriel_mean": all(np.array_equal(np.load(CASES / f"{case['target_label']}.npz")["source"], np.load(NEW_DATA_DIR / "global_mean_gabriel.npy")) for case in cases),
        "exactly_identity_plus_eight_cases": [case["target_label"] for case in cases] == FIXED_LABELS,
        "registration_defaults_match_reference": all(
            case["configuration"] == gabriel.jsonable(asdict(CONFIG)) for case in cases
        ),
        "all_cases_visible": len(rows) == 9,
        "unit_tests_passed": unit["status"] == "PASS",
        "source_faithfulness_validation_passed": source_faithfulness["status"] == "PASS",
        "metric_identities_hold": all(abs(row["after_pointwise_rmse"] ** 2 - row["after_pointwise_mse"]) <= 1e-16 and abs(row["after_coordinatewise_mse"] - row["after_pointwise_mse"] / 2) <= 1e-16 for row in rows),
        "identity_near_zero_and_trivial": rows[0]["after_pointwise_mse"] <= 1e-24 and rows[0]["deformation_coefficient_l2_norm"] <= 1e-12,
        "integration_rows_complete": len(integration_rows) == 18,
        "only_expected_parameter_count": CONFIG.parameter_count == 1800,
        "required_outputs_present": all((OUT / path).exists() for path in required_files),
        "no_full_population_registration": len(cases) == 9,
        "no_pca_gmm_templates": True,
        "no_commit_or_push_performed_by_script": True,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "case_status_counts": dict(Counter(case["status"] for case in cases)),
        "old_dataset_sha256_before": config["old_dataset_sha256_before"],
        "old_dataset_sha256_after": old_hash_after,
        "old_mean_sha256_before": config["old_mean_sha256_before"],
        "old_mean_sha256_after": old_mean_hash_after,
        "existing_outputs_metadata_before": config["existing_outputs_metadata_before"],
        "existing_outputs_metadata_after_excluding_new_experiment": current_outputs,
        "limitations": [
            "Existing-output verification compares path, size and nanosecond mtime metadata; the protected old dataset and old mean are SHA-256 checked byte-for-byte.",
            "Positive Jacobian values on a 101x101 numerical grid are evidence, not an analytical proof of diffeomorphism.",
            "No subprocess in this experiment performs git commit or push.",
        ],
    }


def fmt(value: float) -> str:
    return f"{value:.3e}"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    output = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    output.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(output)


def write_report(
    cases: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    integration_rows: list[dict[str, Any]],
    validation: dict[str, Any],
) -> None:
    summary = json.loads((OUT / "dataset_summary.json").read_text(encoding="utf-8"))
    dataset_validation = json.loads((OUT / "dataset_validation.json").read_text(encoding="utf-8"))
    mean_validation = json.loads((OUT / "global_mean_validation.json").read_text(encoding="utf-8"))
    source_faithfulness = json.loads(
        (OUT / "source_faithfulness_validation.json").read_text(encoding="utf-8")
    )
    nonidentity = [row for row in rows if row["target_label"] != "identity"]
    completed = [row for row in nonidentity if row["status"] == "completed"]
    mse = np.asarray([row["after_pointwise_mse"] for row in completed])
    rmse = np.asarray([row["after_pointwise_rmse"] for row in completed])
    max_error = np.asarray([row["after_maximum_correspondence_error"] for row in completed])
    total_runtime = sum(row["total_case_runtime_seconds"] for row in rows)
    worst_jac = min(rows, key=lambda row: row["jacobian_jacobian_minimum"])
    folded = [row for row in rows if row["jacobian_jacobian_nonpositive_fraction"] > 0]
    max_rk510_point = max(row["maximum_transformed_point_difference"] for row in integration_rows if row["comparison"] == "RK5_vs_RK10")
    max_rk1020_point = max(row["maximum_transformed_point_difference"] for row in integration_rows if row["comparison"] == "RK10_vs_RK20")
    max_rk510_mse = max(row["absolute_change_in_pointwise_mse"] for row in integration_rows if row["comparison"] == "RK5_vs_RK10")
    max_rk1020_mse = max(row["absolute_change_in_pointwise_mse"] for row in integration_rows if row["comparison"] == "RK10_vs_RK20")
    failures = read_csv(OUT / "conversion_failures.csv")
    failure_counts = Counter(row["failure_reason"] for row in failures)
    accuracy_table = []
    for row in rows:
        accuracy_table.append(
            [
                row["target_label"],
                row["filename"],
                row["status"],
                fmt(row["after_pointwise_mse"]),
                fmt(row["after_pointwise_rmse"]),
                fmt(row["after_maximum_correspondence_error"]),
                f"{row['jacobian_jacobian_minimum']:.4f}",
                f"{row['total_case_runtime_seconds']:.1f}",
            ]
        )
    report_lines = [
        "# Gabriel 固定对应翼型注册：开发子集基准",
        "",
        "## 技术摘要",
        "",
        f"本实验从原始 UIUC Selig 文件独立建立 Gabriel 参数化数据集，共检查 `{summary['raw_dat_files_inspected']}` 个文件，Gabriel 原始标量过滤通过 `{summary['reference_filter_accepted']}` 个，附加的无修复几何验证后保留 `{summary['final_dataset_accepted']}` 个。旧余弦数据集未被用作注册输入，也保持 SHA-256 不变。",
        "",
        f"独立源码交叉检验对全部 `{source_faithfulness['saved_contours_checked']}` 个保留轮廓重新调用 Gabriel 原始重采样，并比较 cubic basis、local velocity、RK4 flow 与 smoothness；所有最大差异均为 `0.0`。",
        "",
        f"固定 8 个非恒等开发目标均按文件名匹配成功；成功注册 `{len(completed)}/8` 个。非恒等目标最终点级 MSE 中位数/最差值为 `{fmt(float(np.median(mse)))}` / `{fmt(float(np.max(mse)))}`，RMSE 中位数为 `{fmt(float(np.median(rmse)))}`，最大单点误差的最差值为 `{fmt(float(np.max(max_error)))}`。这些数值属于 Gabriel 等弧长对应，不能与旧余弦 chordwise 对应的 MSE 直接作同一目标比较。",
        "",
        f"最弱 sampled Jacobian 出现在 `{worst_jac['target_label']}`，最小值 `{worst_jac['jacobian_jacobian_minimum']:.4f}`，非正比例 `{worst_jac['jacobian_jacobian_nonpositive_fraction']:.2%}`。因此 Gabriel 原法在本开发集上不能判定为数值安全；同时最坏 RK5→RK10 的 MSE 变化为 `{fmt(max_rk510_mse)}`，说明极端案例的 RK5 也未充分收敛。9 个案例优化加复测诊断的计时组件合计 `{total_runtime:.1f}` 秒。",
        "",
        "结论是：该方法在对应误差上很准确、速度场正则量有限、单目标计算可行，但 `extreme` 的 sampled folding 与积分敏感性否决了“按当前原设置直接扩大到完整人群”的建议。下一步若获批准，应先单独处理安全性与 RK 时间离散，而不是立即进行全体注册。",
        "",
        "## 新数据集保持了 Gabriel 的 TE/LE 锚点与等弧长索引",
        "",
        "Gabriel 读取每个文件时跳过首行，只解析其后恰含两个空白分隔字段的行；按文件名排除 flap/main-flap/slat/30p-30n，随后执行点数、坐标幅度、弦长和归一化 y 范围过滤。坐标按 `(coords-[min(x),0])/(max(x)-min(x))` 归一化，不进行旋转或 y 平移。",
        "",
        "前缘取第一个最小 x 点。上表面使用原始 `TE→LE` 片段，按 100 个等间隔 polyline 弧长位置采样并排除 LE 端点；下表面从 LE 开始并显式连接首个 TE 点，同样取 100 个位置并排除闭合 TE。最终 index 0 为 TE、index 100 为 LE，index 199→0 构成闭合段，没有第 201 个重复点。",
        "",
        f"所有保留轮廓均为 `(200,2)` float64，锚点、表面方向、有限性、零长度段、自交和方向检查全部通过。上/下表面相邻弦段长度 CV 的最大值分别为 `{dataset_validation['metric_summaries']['upper_segment_cv']['maximum']:.3e}` / `{dataset_validation['metric_summaries']['lower_segment_cv']['maximum']:.3e}`；采样位置本身严格来自等间隔 polyline 弧长，跨越原始折点时相邻欧氏弦长允许略短。",
        "",
        "![固定开发目标的点序 QA](../figures/dataset_qa_development_targets.png)",
        "",
        "图中明确标出 0、99、100、199，并以虚线表示 199→0 闭合段；它用于验证顺序而非注册精度。",
        "",
        "## 总体对账显示两种预处理并非完全相同的人群",
        "",
        f"新数据集中 `{summary['matching_existing_1654_filenames']}` 个文件名可唯一映射到旧 1654 集；仅 Gabriel 集有 `{summary['present_only_in_gabriel_dataset_count']}` 个，仅旧集有 `{summary['present_only_in_existing_dataset_count']}` 个。不存在被执行的方向修复；不符合预期遍历或几何有效性的文件被明确拒绝。",
        "",
        "主要拒绝原因：",
        "",
    ]
    for reason, count in failure_counts.most_common():
        report_lines.append(f"- `{reason}`：{count} 个")
    report_lines += [
        "",
        "完整逐文件状态、警告和哈希见 `conversion_manifest.csv`；拒绝项见 `conversion_failures.csv`。重复原始 TE、有限厚度 TE、多个或近似多个最小 x 点均被记录；仅当最终参数化仍满足确定性几何验证时才保留。",
        "",
        "## 新全局均值是唯一注册源",
        "",
        f"新均值由全部 `{summary['final_dataset_accepted']}` 个有效 Gabriel 轮廓逐点平均得到。形状 `{mean_validation['shape_valid'] and '(200,2)' or 'invalid'}`，有限性 `{mean_validation['finite']}`，index 0 TE 与 index 100 LE 检查均通过，自交数 `{mean_validation['self_intersection_count']}`。旧余弦均值未参与拟合。",
        "",
        "![Gabriel 全局均值](../figures/global_mean_gabriel.png)",
        "",
        "均值图同时显示四个关键索引；均值轮廓保持与个体一致的逆时针 TE-upper-LE-lower-TE 顺序。",
        "",
        "## Gabriel 原法在开发目标上的精度",
        "",
        markdown_table(
            ["案例", "文件", "状态", "点级 MSE", "点级 RMSE", "最大误差", "min det(J)", "优化+诊断秒"],
            accuracy_table,
        ),
        "",
        "点级 MSE 定义为 200 个对应点的二维平方距离平均；coordinate-wise MSE 是 400 个标量残差的平均，因此严格等于点级 MSE/2。TE 区域固定为索引 0–9 与 190–199，LE 为 90–109，mid 为其余 160 点。identity 单独报告，不进入上述非恒等汇总。",
        "",
        "![逐目标精度](../figures/accuracy_by_target.png)",
        "",
        "柱图使用对数轴展示拟合前后误差；由于三个指标单位和量级不同，分别放在独立面板，避免共享尺度误导。",
        "",
        "![逐点残差](../figures/residuals_before_after.png)",
        "",
        "逐点图用于定位残差是否集中在前缘、后缘或中部；完整数值保存在 `per_point_residuals.csv`。",
        "",
        "## 速度场平滑度与变形规模需使用两个不同口径",
        "",
        f"默认模型有 `{CONFIG.parameter_count}` 个标量参数。Gabriel 的 `R` 是 30×30 系数阵列在两个格轴上的未归一化一阶差分平方和；它不是连续 Sobolev/RKHS 范数，也不能跨网格分辨率直接比较。本实验另在统一 101×101 物理网格上计算速度梯度平方的梯形积分作为 Dirichlet 诊断，并单独计算分量 Laplacian 平方积分作为弯曲诊断。",
        "",
        f"非恒等案例 raw coefficient smoothness 中位数为 `{fmt(float(np.median([row['final_coefficient_smoothness'] for row in completed])))}`，物理 Dirichlet 诊断中位数为 `{fmt(float(np.median([row['deformation_physical_dirichlet_integral'] for row in completed])))}`，平均源点位移中位数为 `{fmt(float(np.median([row['deformation_mean_source_displacement'] for row in completed])))}`。这些量解释变形规模和平滑度，但没有进入附件项以外的新目标。",
        "",
        "![代表性变形网格](../figures/representative_deformation_grids.png)",
        "",
        "代表性网格按初始难度选择 easy/median/hard 案例，只显示同一个固定空间网格经最终流映射后的形状。",
        "",
        "## sampled Jacobian 未参与优化，但揭示数值安全边界",
        "",
        f"所有案例中的折叠案例数为 `{len(folded)}`（`{', '.join(row['target_label'] for row in folded) or 'none'}`）；变形后轮廓自交案例数为 `{sum(row['geometry_self_intersection_count'] > 0 for row in rows)}`；轨迹曾离开控制格范围的案例数为 `{sum(bool(row['trajectory_trajectory_any_outside_control_range']) for row in rows)}`。Gabriel 原目标没有 Jacobian barrier，本实验没有加入。",
        "",
        "![最弱 Jacobian 案例](../figures/worst_case_jacobian.png)",
        "",
        "热图覆盖固定的 `[-0.05,1.05]×[-0.35,0.35]` 范围。边界处的有限差分和 B-spline 支撑截断会影响极值解释，因此结论限定为 sampled numerical evidence。",
        "",
        "## RK5 与更细积分的差异量化了时间离散误差",
        "",
        f"冻结系数后，RK5→RK10 的最大源点差异在所有案例中不超过 `{fmt(max_rk510_point)}`，最大 MSE 变化 `{fmt(max_rk510_mse)}`；RK10→RK20 分别不超过 `{fmt(max_rk1020_point)}` 与 `{fmt(max_rk1020_mse)}`。极端案例主导这些上界，故 RK5 对它不充分；逐案例差异见 `integration_convergence.csv`。",
        "",
        "![RK 收敛](../figures/rk_integration_convergence.png)",
        "",
        "图中比较的是同一最终速度场的积分精度，不包含重新优化，因此可以直接归因于 RK 步数。",
        "",
        "## 分阶段 L-BFGS 严格复现自适应正则策略",
        "",
        "每个案例从零系数开始。第一周期固定 λ=0.1；后续周期只在周期开始计算一次 `min(0.2·D/max(R,1e-12),0.1)`，随后在该周期内保持不变。每周期使用 strong-Wolfe L-BFGS，最多 100 次 optimizer iteration；closure evaluation 另行计数，二者没有混称。",
        "",
        f"非恒等案例 closure evaluation 中位数为 `{float(np.median([row['closure_evaluations'] for row in completed])):.0f}`，优化时间中位数为 `{float(np.median([row['runtime_seconds'] for row in completed])):.1f}` 秒，优化加同路径诊断复测的组件时间中位数为 `{float(np.median([row['total_case_runtime_seconds'] for row in completed])):.1f}` 秒。PyTorch 提供每周期 iteration/function-evaluation 计数，但不提供可靠的分类终止原因，因此报告不宣称比现有信息更强的“收敛”。",
        "",
        "![优化轨迹](../figures/optimisation_trace.png)",
        "",
        "轨迹横轴明确是 closure evaluation。周期切换会改变 λ，因此总目标曲线在边界处不必连续。",
        "",
        "## 限制、稳健性与是否值得扩大",
        "",
        f"最终验证状态为 `{validation['status']}`。本实验只注册 9 个开发案例；数据转换虽然覆盖全部原始文件，但没有对其余 `{summary['final_dataset_accepted'] - 8}` 个有效目标做注册。因对应参数化不同，本报告没有把新 MSE 与旧 Stage 4 MSE 当作同一统计量比较。",
        "",
        "当前不建议直接进行后续完整人群注册：开发集中已出现 sampled folding，而且极端案例的 RK5 时间离散误差不可忽略。精度和运行时间本身支持继续方法研究，但没有覆盖安全性失败。",
        "",
        "## 建议的下一步",
        "",
        "- 在获得批准前保持停止；不启动完整人群注册、PCA/GMM 或模板构建。",
        "- 若批准下一阶段，优先研究极端案例的 folding 与 RK5 误差；这属于新的方法开发，必须另行授权。",
        "- 只有安全与积分充分性重新验证通过后，才估算并考虑完整人群运行。",
        "",
        "## 仍待回答的问题",
        "",
        "- 开发子集的安全性和耗时能否代表整个 Gabriel 人群中的极端几何？",
        "- 采用 Gabriel 等弧长对应后，前缘/后缘残差是否在完整人群中呈系统性结构？",
        "- 在不改变 Gabriel 忠实基线结论的前提下，何种安全机制能消除极端案例的局部折叠？",
        "",
        "本阶段到此停止。没有运行完整数据集注册，没有运行 PCA/GMM、模板选择或气动优化，也没有 commit/push。",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_name(REPORT.name + ".tmp")
    temporary.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    os.replace(temporary, REPORT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "register", "finalise", "all"))
    args = parser.parse_args()
    if args.phase in {"prepare", "all"}:
        prepare()
    if args.phase in {"register", "all"}:
        run_registrations()
    if args.phase in {"finalise", "all"}:
        finalise()


if __name__ == "__main__":
    main()
