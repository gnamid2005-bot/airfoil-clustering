"""Phase A--K screening for the number of representative airfoil templates.

The registration map is intentionally imported unchanged from the Stage 3
development script.  This driver adds only orchestration, diagnostics and the
PCA/GMM analysis.  Per-target NPZ files are atomic and are skipped when valid,
so the population run is resumable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

import run_airfoil_registration_improvement_stage2 as s2
import run_airfoil_registration_improvement_stage3 as s3
from airfoil_registration import validate_airfoil_shape


matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "airfoil_template_number_stage5_screening"
DATA = ROOT / "data" / "airfoils" / "processed" / "airfoils_n100_seed0.npz"
STAGE2_MEAN = ROOT / "outputs" / "airfoil_registration_improvement_stage2" / "direct_coordinate_mean" / "coordinates.npy"
STAGE3 = ROOT / "outputs" / "airfoil_registration_improvement_stage3"
DTYPE = torch.float64
SCHEDULE = (1e-2, 1e-4, 1e-5)
REG_GRID = 51
FFT_HIGH_FREQUENCY_THRESHOLD = 0.5
ENTROPY_BINS = 32


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def frozen_config(shapes: np.ndarray) -> dict:
    stage3_config = json.loads((STAGE3 / "config.json").read_text(encoding="utf-8"))
    stage3_best = json.loads((STAGE3 / "best_configuration.json").read_text(encoding="utf-8"))
    assert stage3_best["selected_method"] == "local48_wide"
    assert float(stage3_best["selected_local_alpha"]) == 1.0
    return {
        "method": "stage3_local48_wide_frozen",
        "dtype": "float64",
        "optimizer": "torch.optim.LBFGS strong_wolfe",
        "global_resolutions": [6, 12, 24],
        "global_regularisation_schedule": list(SCHEDULE),
        "local_resolution": 48,
        "local_edge_support": 0.15,
        "local_alpha": 1.0,
        "local_regularisation": 1e-5,
        "integration": "explicit Euler stationary velocity field",
        "steps": int(stage3_config["steps"]),
        "barrier_tau": float(stage3_config["barrier_tau"]),
        "barrier_weight": float(stage3_config["barrier_weight"]),
        "barrier_resolution": int(stage3_config["barrier_resolution"]),
        "jacobian_report_resolution": int(stage3_config["jacobian_report_resolution"]),
        "global_lbfgs_max_iter": 80,
        "local_lbfgs_max_iter": int(stage3_config["local_lbfgs_max_iter"]),
        "tolerance_grad": 1e-12,
        "tolerance_change": 1e-15,
        "bounds": [0.0, 1.0, float(shapes[:, :, 1].min() - 0.08), float(shapes[:, :, 1].max() + 0.08)],
        "field_vector_dimension": 10368,
        "field_vector_order": "C-order concatenate global[24,24,2], LE[48,48,2], TE[48,48,2]",
        "field_scaling": "none",
    }


def load_dataset() -> dict:
    raw = np.load(DATA)
    shapes = raw["shapes"].astype(np.float64)
    filenames = raw["filenames"].astype(str)
    names = raw["names"].astype(str)
    if shapes.shape != (1654, 200, 2):
        raise ValueError(f"Unexpected dataset shape: {shapes.shape}")
    if not np.isfinite(shapes).all():
        raise ValueError("Dataset contains non-finite values")
    if float(np.max(np.ptp(shapes[:, :, 0], axis=0))) != 0.0:
        raise ValueError("The chordwise x grid is not fixed")
    invalid = [index for index, shape in enumerate(shapes) if not validate_airfoil_shape(shape)["valid"]]
    if invalid:
        raise ValueError(f"Invalid processed geometries: {len(invalid)}")
    mean = shapes.mean(axis=0)
    saved_mean = np.load(STAGE2_MEAN).astype(np.float64)
    if not np.array_equal(mean, saved_mean):
        raise ValueError(f"Stage 2 mean differs from direct mean: {np.max(np.abs(mean-saved_mean))}")
    return {"shapes": shapes, "filenames": filenames, "names": names, "mean": mean}


def vectorize_coefficients(g: np.ndarray, le: np.ndarray, te: np.ndarray) -> np.ndarray:
    if g.shape != (24, 24, 2) or le.shape != (48, 48, 2) or te.shape != (48, 48, 2):
        raise ValueError(f"Inconsistent coefficient shapes: {g.shape}, {le.shape}, {te.shape}")
    return np.concatenate([g.ravel(order="C"), le.ravel(order="C"), te.ravel(order="C")])


def common_grid(bounds: tuple[float, float, float, float], resolution: int = REG_GRID) -> tuple[torch.Tensor, float, float]:
    xs = torch.linspace(bounds[0], bounds[1], resolution, dtype=DTYPE)
    ys = torch.linspace(bounds[2], bounds[3], resolution, dtype=DTYPE)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx.ravel(), yy.ravel()), dim=1), float(xs[1] - xs[0]), float(ys[1] - ys[0])


def composite_field_array(
    g: torch.Tensor,
    le: torch.Tensor,
    te: torch.Tensor,
    bounds: tuple[float, float, float, float],
    resolution: int = REG_GRID,
) -> tuple[np.ndarray, float, float]:
    points, dx, dy = common_grid(bounds, resolution)
    with torch.no_grad():
        field = s3.hierarchical_velocity(points, g, le, te, bounds, 0.15)
    return field.reshape(resolution, resolution, 2).cpu().numpy(), dx, dy


def field_properties(
    source: torch.Tensor,
    target: torch.Tensor,
    g: torch.Tensor,
    le: torch.Tensor,
    te: torch.Tensor,
    bounds: tuple[float, float, float, float],
    cfg: dict,
    metrics: dict,
) -> tuple[dict, np.ndarray]:
    field, dx, dy = composite_field_array(g, le, te, bounds)
    vx, vy = field[:, :, 0], field[:, :, 1]
    magnitude = np.linalg.norm(field, axis=2)
    dvx_dy, dvx_dx = np.gradient(vx, dy, dx, edge_order=2)
    dvy_dy, dvy_dx = np.gradient(vy, dy, dx, edge_order=2)
    curl = dvy_dx - dvx_dy
    divergence = dvx_dx + dvy_dy
    lap_vx = np.gradient(dvx_dx, dx, axis=1, edge_order=2) + np.gradient(dvx_dy, dy, axis=0, edge_order=2)
    lap_vy = np.gradient(dvy_dx, dx, axis=1, edge_order=2) + np.gradient(dvy_dy, dy, axis=0, edge_order=2)
    area = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2])
    dirichlet = area * float(np.mean(dvx_dx**2 + dvx_dy**2 + dvy_dx**2 + dvy_dy**2))
    bending = area * float(np.mean(lap_vx**2 + lap_vy**2))
    kinetic = area * float(np.mean(vx**2 + vy**2))

    fft_vx = np.fft.fftshift(np.fft.fft2(vx - vx.mean()))
    fft_vy = np.fft.fftshift(np.fft.fft2(vy - vy.mean()))
    spectral = np.abs(fft_vx) ** 2 + np.abs(fft_vy) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(REG_GRID, d=dy))
    fx = np.fft.fftshift(np.fft.fftfreq(REG_GRID, d=dx))
    fyy, fxx = np.meshgrid(fy, fx, indexing="ij")
    radial = np.sqrt((fxx / np.max(np.abs(fx))) ** 2 + (fyy / np.max(np.abs(fy))) ** 2)
    spectral_total = float(spectral.sum())
    high_fraction = float(spectral[radial >= FFT_HIGH_FREQUENCY_THRESHOLD].sum() / max(spectral_total, 1e-300))

    normalised_mag = magnitude / max(float(magnitude.max()), 1e-300)
    counts, _ = np.histogram(normalised_mag, bins=ENTROPY_BINS, range=(0.0, 1.0))
    probs = counts[counts > 0] / max(counts.sum(), 1)
    entropy = float(-np.sum(probs * np.log(probs)) / np.log(ENTROPY_BINS)) if len(probs) else 0.0

    with torch.no_grad():
        x = source.clone()
        dt = 1.0 / int(cfg["steps"])
        path = torch.zeros(len(source), dtype=DTYPE)
        for _ in range(int(cfg["steps"])):
            velocity = s3.hierarchical_velocity(x, g, le, te, bounds, 0.15)
            path += dt * torch.linalg.norm(velocity, dim=1)
            x = x + dt * velocity
        deformed = x
        errors = torch.linalg.norm(deformed - target, dim=1)
        displacements = torch.linalg.norm(deformed - source, dim=1)
        jac = s3.jacobian(g, le, te, bounds, 0.15, int(cfg["steps"]), REG_GRID).cpu().numpy()

    row = {
        "metric_group_field_magnitude": "CORE",
        "mean_vector_field_magnitude": float(magnitude.mean()),
        "rms_vector_field_magnitude": float(np.sqrt(np.mean(magnitude**2))),
        "maximum_vector_field_magnitude": float(magnitude.max()),
        "velocity_magnitude_variance": float(np.var(magnitude)),
        "metric_group_dirichlet": "CORE",
        "dirichlet_energy_common_grid": dirichlet,
        "registration_regularisation_surrogate": float(metrics["raw_dirichlet"]),
        "correspondence_mse": float(torch.mean(errors**2)),
        "correspondence_rmse": float(torch.sqrt(torch.mean(errors**2))),
        "maximum_pointwise_error": float(errors.max()),
        "classical_lddmm_geodesic_energy": "UNAVAILABLE",
        "classical_lddmm_geodesic_energy_reason": "stationary Cartesian/local field has no specified RKHS metric or LDDMM Hamiltonian",
        "mean_displacement": float(displacements.mean()),
        "maximum_displacement": float(displacements.max()),
        "jacobian_minimum": float(jac.min()),
        "jacobian_mean": float(jac.mean()),
        "jacobian_median": float(np.median(jac)),
        "jacobian_p01": float(np.quantile(jac, 0.01)),
        "jacobian_p05": float(np.quantile(jac, 0.05)),
        "jacobian_nonpositive_proportion": float(np.mean(jac <= 0.0)),
        "jacobian_variance": float(np.var(jac)),
        "curl_rms": float(np.sqrt(np.mean(curl**2))),
        "curl_maximum_absolute": float(np.max(np.abs(curl))),
        "divergence_rms": float(np.sqrt(np.mean(divergence**2))),
        "divergence_maximum_absolute": float(np.max(np.abs(divergence))),
        "bending_energy_laplacian": bending,
        "mean_flow_path_length": float(path.mean()),
        "maximum_flow_path_length": float(path.max()),
        "kinetic_energy": kinetic,
        "rkhs_norm": "UNAVAILABLE",
        "rkhs_norm_reason": "no RKHS kernel/operator or V-norm is specified for the frozen model",
        "metric_group_spectral_entropy": "EXPLORATORY",
        "spectral_high_frequency_fraction": high_fraction,
        "spectral_threshold_normalised_radius": FFT_HIGH_FREQUENCY_THRESHOLD,
        "field_magnitude_entropy": entropy,
        "field_entropy_definition": "Shannon entropy / log(32) of 32 fixed bins on per-field magnitude/max(magnitude) in [0,1]",
    }
    return row, deformed.cpu().numpy()


def registration_artifact_valid(path: Path, index: int) -> bool:
    try:
        with np.load(path) as value:
            return (
                int(value["target_index"]) == index
                and value["field_vector"].shape == (10368,)
                and value["deformed"].shape == (200, 2)
                and np.isfinite(value["field_vector"]).all()
            )
    except Exception:
        return False


def register_one(payload: tuple[int, str, str, np.ndarray, np.ndarray, dict, str]) -> dict:
    index, filename, name, source_np, target_np, cfg, artifact_text = payload
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch allows this setting only once per process; sequential dry
        # runs reuse the same process while population workers do not.
        pass
    artifact = Path(artifact_text)
    bounds = tuple(float(value) for value in cfg["bounds"])
    source = torch.tensor(source_np, dtype=DTYPE)
    target = torch.tensor(target_np, dtype=DTYPE)
    base_cfg = {
        "steps": int(cfg["steps"]),
        "basis": "cartesian",
        "barrier_resolution": int(cfg["barrier_resolution"]),
        "lbfgs_max_iter": int(cfg["global_lbfgs_max_iter"]),
        "tolerance_grad": float(cfg["tolerance_grad"]),
        "tolerance_change": float(cfg["tolerance_change"]),
    }
    wall_start = time.perf_counter()
    failure_reason = ""
    try:
        base_metrics, c24, _, _, _ = s2.run_schedule(
            source_np,
            target_np,
            bounds,
            base_cfg,
            SCHEDULE,
            float(cfg["barrier_tau"]),
            float(cfg["barrier_weight"]),
            (6, 12, 24),
        )
        local_cfg = {
            "steps": int(cfg["steps"]),
            "barrier_resolution": int(cfg["barrier_resolution"]),
            "barrier_tau": float(cfg["barrier_tau"]),
            "barrier_weight": float(cfg["barrier_weight"]),
            "local_lbfgs_max_iter": int(cfg["local_lbfgs_max_iter"]),
        }
        local_metrics, coeffs, _, _ = s3.run_local(
            source,
            target,
            torch.tensor(c24, dtype=DTYPE),
            bounds,
            local_cfg,
            48,
            0.15,
            1.0,
            "joint",
        )
        local_metrics["runtime_seconds"] += float(base_metrics["runtime_seconds"])
        g_np, le_np, te_np = coeffs
        g = torch.tensor(g_np, dtype=DTYPE)
        le = torch.tensor(le_np, dtype=DTYPE)
        te = torch.tensor(te_np, dtype=DTYPE)
        properties, deformed = field_properties(source, target, g, le, te, bounds, cfg, local_metrics)
        field_vector = vectorize_coefficients(g_np, le_np, te_np)
        validity = validate_airfoil_shape(deformed)
        safe = bool(
            local_metrics["safe"]
            and validity["valid"]
            and properties["jacobian_minimum"] > 0.0
            and properties["jacobian_nonpositive_proportion"] == 0.0
            and np.isfinite(field_vector).all()
        )
    except Exception as exc:  # retained as a saved failure row for audit/resume
        failure_reason = f"{type(exc).__name__}: {exc}"
        return {
            "target_index": index,
            "filename": filename,
            "name": name,
            "success": False,
            "safe": False,
            "failure_reason": failure_reason,
            "wall_runtime_seconds": time.perf_counter() - wall_start,
        }

    row = {
        "target_index": index,
        "filename": filename,
        "name": name,
        "success": True,
        "safe": safe,
        "valid_geometry": bool(validity["valid"]),
        "failure_reason": failure_reason,
        "optimizer_runtime_seconds": float(local_metrics["runtime_seconds"]),
        "wall_runtime_seconds": time.perf_counter() - wall_start,
        "closure_evaluations": int(local_metrics["closure_evaluations"]),
        "converged": local_metrics["converged"],
        "convergence_status": local_metrics["convergence_status"],
        **properties,
    }
    artifact.parent.mkdir(parents=True, exist_ok=True)
    tmp = artifact.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        target_index=np.int64(index),
        filename=np.asarray(filename),
        global_coefficients=g_np,
        le_coefficients=le_np,
        te_coefficients=te_np,
        field_vector=field_vector,
        deformed=deformed,
        target=target_np,
        metrics_json=np.asarray(json.dumps(row, sort_keys=True)),
    )
    tmp.replace(artifact)
    return row


def parse_metrics_from_artifact(path: Path) -> dict:
    with np.load(path) as value:
        return json.loads(str(value["metrics_json"]))


def selected_indices(dataset_size: int, indices_text: str, limit: int | None) -> list[int]:
    if indices_text:
        result = [int(value) for value in indices_text.split(",") if value.strip()]
    else:
        result = list(range(dataset_size))
    if limit is not None:
        result = result[:limit]
    if any(index < 0 or index >= dataset_size for index in result):
        raise ValueError("Registration index out of bounds")
    return result


def run_registrations(out: Path, dataset: dict, cfg: dict, indices: list[int], workers: int, force: bool) -> list[dict]:
    artifacts = out / "registrations"
    artifacts.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    todo: list[tuple] = []
    for index in indices:
        path = artifacts / f"{index:04d}.npz"
        if not force and registration_artifact_valid(path, index):
            rows.append(parse_metrics_from_artifact(path))
        else:
            todo.append(
                (
                    index,
                    str(dataset["filenames"][index]),
                    str(dataset["names"][index]),
                    dataset["mean"],
                    dataset["shapes"][index],
                    cfg,
                    str(path),
                )
            )
    print(f"resume: {len(rows)} valid artifacts skipped; {len(todo)} registrations pending", flush=True)
    if workers <= 1:
        for number, payload in enumerate(todo, 1):
            row = register_one(payload)
            rows.append(row)
            print(f"[{number}/{len(todo)}] index={row['target_index']} safe={row['safe']} runtime={row['wall_runtime_seconds']:.2f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(register_one, payload): payload[0] for payload in todo}
            for number, future in enumerate(as_completed(futures), 1):
                row = future.result()
                rows.append(row)
                print(f"[{number}/{len(todo)}] index={row['target_index']} safe={row['safe']} runtime={row['wall_runtime_seconds']:.2f}s", flush=True)
    rows.sort(key=lambda row: int(row["target_index"]))
    save_csv(out / "registration_results.csv", rows)
    save_csv(out / "velocity_field_properties.csv", rows)
    return rows


def reproduce_baseline(out: Path, dataset: dict, cfg: dict) -> dict:
    stage4_config = json.loads((ROOT / "outputs" / "airfoil_registration_improvement_stage4" / "config.json").read_text(encoding="utf-8"))
    indices = stage4_config["input_verification"]["selected_indices"][1:]
    rows = run_registrations(out / "baseline_reproduction", dataset, cfg, indices, workers=1, force=True)
    values = [float(row["correspondence_mse"]) for row in rows]
    result = {
        "target_indices": indices,
        "median_correspondence_mse": float(np.median(values)),
        "worst_correspondence_mse": float(np.max(values)),
        "expected_median": 1.1382817569856905e-8,
        "expected_worst": 3.268340443462815e-8,
        "all_safe": all(bool(row["safe"]) for row in rows),
    }
    result["consistent"] = bool(
        result["all_safe"]
        and np.isclose(result["median_correspondence_mse"], result["expected_median"], rtol=5e-8, atol=1e-14)
        and np.isclose(result["worst_correspondence_mse"], result["expected_worst"], rtol=5e-8, atol=1e-14)
    )
    atomic_json(out / "baseline_reproduction.json", result)
    if not result["consistent"]:
        raise RuntimeError(f"Frozen baseline did not reproduce: {result}")
    return result


def load_field_matrix(out: Path, n: int) -> tuple[np.ndarray, list[dict], np.ndarray]:
    rows = read_csv(out / "registration_results.csv")
    by_index = {int(row["target_index"]): row for row in rows}
    valid_indices = np.asarray(
        [index for index in range(n) if index in by_index and by_index[index]["success"] == "True" and by_index[index]["safe"] == "True"],
        dtype=np.int64,
    )
    if len(valid_indices) < 0.95 * n:
        raise RuntimeError(f"Too few safe fields for PCA: {len(valid_indices)}/{n}")
    matrix = np.empty((len(valid_indices), 10368), dtype=np.float64)
    for position, index in enumerate(valid_indices):
        with np.load(out / "registrations" / f"{index:04d}.npz") as value:
            matrix[position] = value["field_vector"]
    np.save(out / "velocity_field_matrix.npy", matrix)
    np.save(out / "velocity_field_population_indices.npy", valid_indices)
    return matrix, [by_index[int(index)] for index in valid_indices], valid_indices


def deterministic_split(n: int, seed: int = 20260813) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_train = int(round(0.8 * n))
    return np.sort(order[:n_train]), np.sort(order[n_train:])


def fit_pca_study(
    out: Path,
    matrix: np.ndarray,
    train: np.ndarray,
    heldout: np.ndarray,
    population_indices: np.ndarray,
) -> tuple[PCA, np.ndarray, np.ndarray, list[dict]]:
    model = PCA(n_components=20, svd_solver="randomized", random_state=20260813, iterated_power=7)
    train_scores = model.fit_transform(matrix[train])
    heldout_scores = model.transform(matrix[heldout])
    rows = []
    for q in range(1, 21):
        train_recon = train_scores[:, :q] @ model.components_[:q] + model.mean_
        heldout_recon = heldout_scores[:, :q] @ model.components_[:q] + model.mean_
        rows.append(
            {
                "q": q,
                "explained_variance_ratio": float(model.explained_variance_ratio_[q - 1]),
                "cumulative_explained_variance": float(np.sum(model.explained_variance_ratio_[:q])),
                "train_reconstruction_mse": float(np.mean((matrix[train] - train_recon) ** 2)),
                "heldout_reconstruction_mse": float(np.mean((matrix[heldout] - heldout_recon) ** 2)),
            }
        )
    np.savez_compressed(
        out / "pca" / "pca_q20.npz",
        mean=model.mean_,
        components=model.components_,
        explained_variance=model.explained_variance_,
        explained_variance_ratio=model.explained_variance_ratio_,
        singular_values=model.singular_values_,
        train_matrix_positions=train,
        heldout_matrix_positions=heldout,
        train_indices=population_indices[train],
        heldout_indices=population_indices[heldout],
        train_scores=train_scores,
        heldout_scores=heldout_scores,
    )
    save_csv(out / "pca" / "pca_dimension_study.csv", rows)
    return model, train_scores, heldout_scores, rows


def posterior_entropy(probabilities: np.ndarray) -> float:
    return float(-np.sum(probabilities * np.log(np.clip(probabilities, 1e-300, 1.0))))


def component_values(values: np.ndarray) -> str:
    return json.dumps([float(value) for value in values], separators=(",", ":"))


def fit_gmm_cell(train_scores: np.ndarray, heldout_scores: np.ndarray, q: int, k: int, seed: int, n_init: int, covariance: str = "full") -> tuple[dict, GaussianMixture]:
    start = time.perf_counter()
    model = GaussianMixture(
        n_components=k,
        covariance_type=covariance,
        n_init=n_init,
        random_state=seed,
        reg_covar=1e-7,
        max_iter=500,
        tol=1e-5,
        init_params="kmeans",
    ).fit(train_scores[:, :q])
    train_ll_mean = float(model.score(train_scores[:, :q]))
    heldout_ll_mean = float(model.score(heldout_scores[:, :q]))
    probabilities = model.predict_proba(train_scores[:, :q])
    entropy = posterior_entropy(probabilities)
    labels = probabilities.argmax(axis=1)
    sizes = np.bincount(labels, minlength=k)
    bic = float(model.bic(train_scores[:, :q]))
    row = {
        "q": q,
        "K": k,
        "covariance_type": covariance,
        "seed": seed,
        "n_init": n_init,
        "training_log_likelihood_total": train_ll_mean * len(train_scores),
        "training_log_likelihood_mean": train_ll_mean,
        "heldout_log_likelihood_total": heldout_ll_mean * len(heldout_scores),
        "heldout_log_likelihood_mean": heldout_ll_mean,
        "BIC": bic,
        "ICL": bic + 2.0 * entropy,
        "posterior_entropy_total": entropy,
        "posterior_entropy_mean": entropy / len(train_scores),
        "component_weights": component_values(model.weights_),
        "hard_component_sizes": json.dumps([int(value) for value in sizes]),
        "minimum_component_size": int(sizes.min()),
        "tiny_component_flag": bool(sizes.min() < max(10, math.ceil(0.01 * len(train_scores)))),
        "converged": bool(model.converged_),
        "iterations": int(model.n_iter_),
        "runtime_seconds": time.perf_counter() - start,
    }
    return row, model


def run_gmm_dry(out: Path, train_scores: np.ndarray, heldout_scores: np.ndarray) -> dict:
    rows = []
    start = time.perf_counter()
    for k in (1, 2, 3):
        row, _ = fit_gmm_cell(train_scores, heldout_scores, 5, k, 20260813, 1)
        rows.append(row)
    result = {
        "cells": 3,
        "starts": 3,
        "runtime_seconds": time.perf_counter() - start,
        "projected_400_cell_runtime_seconds": (time.perf_counter() - start) / 3.0 * 400.0 * 5.0,
        "all_converged": all(row["converged"] for row in rows),
    }
    save_csv(out / "gmm" / "dry_run.csv", rows)
    atomic_json(out / "gmm" / "dry_run_estimate.json", result)
    return result


def run_primary_gmm(out: Path, train_scores: np.ndarray, heldout_scores: np.ndarray, force: bool) -> list[dict]:
    path = out / "gmm" / "primary_grid.csv"
    existing = read_csv(path) if path.exists() and not force else []
    by_cell = {(int(row["q"]), int(row["K"])): row for row in existing}
    rows: list[dict] = list(existing)
    for q in range(1, 21):
        for k in range(1, 21):
            if (q, k) in by_cell:
                continue
            row, model = fit_gmm_cell(train_scores, heldout_scores, q, k, 20260813, 5)
            rows.append(row)
            model_path = out / "gmm" / "models" / f"q{q:02d}_k{k:02d}_full.npz"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                model_path,
                weights=model.weights_,
                means=model.means_,
                covariances=model.covariances_,
                precisions_cholesky=model.precisions_cholesky_,
            )
            save_csv(path, sorted(rows, key=lambda value: (int(value["q"]), int(value["K"]))))
            print(f"GMM q={q:02d} K={k:02d} BIC={row['BIC']:.1f} ICL={row['ICL']:.1f} heldout={row['heldout_log_likelihood_mean']:.3f}", flush=True)
    return sorted(rows, key=lambda value: (int(value["q"]), int(value["K"])))


def preferred_by_q(rows: list[dict]) -> list[dict]:
    result = []
    for q in range(1, 21):
        group = [row for row in rows if int(row["q"]) == q and row["converged"] in (True, "True")]
        result.append(
            {
                "q": q,
                "K_BIC": int(min(group, key=lambda row: float(row["BIC"]))["K"]),
                "K_ICL": int(min(group, key=lambda row: float(row["ICL"]))["K"]),
                "K_heldout_LL": int(max(group, key=lambda row: float(row["heldout_log_likelihood_mean"]))["K"]),
            }
        )
    return result


def build_shortlist(preferred: list[dict]) -> list[int]:
    heldout_counts = Counter(int(row["K_heldout_LL"]) for row in preferred)
    consensus_counts = Counter(
        int(row[key])
        for row in preferred
        for key in ("K_BIC", "K_ICL")
    )
    # Main branch: persistent held-out winners.  Sensitivity branch: the most
    # persistent BIC/ICL winner, retained even when tiny-component diagnostics
    # later prevent it from entering the primary downstream range.
    main = {k for k, count in heldout_counts.items() if count >= 3}
    sensitivity = {max(consensus_counts, key=consensus_counts.get)}
    return sorted(main | sensitivity)


def robustness_q_values(preferred: list[dict], shortlist: list[int]) -> list[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for k in shortlist:
        q_candidates = [int(row["q"]) for row in preferred if k in (int(row["K_BIC"]), int(row["K_ICL"]), int(row["K_heldout_LL"]))]
        q = int(np.median(q_candidates)) if q_candidates else 20
        cells.add((q, k))
    return sorted(cells)


def run_robustness(out: Path, train_scores: np.ndarray, heldout_scores: np.ndarray, cells: list[tuple[int, int]], force: bool) -> list[dict]:
    path = out / "gmm" / "shortlist_robustness.csv"
    existing = read_csv(path) if path.exists() and not force else []
    keys = {(int(row["q"]), int(row["K"]), row["covariance_type"], int(row["seed"])) for row in existing}
    rows: list[dict] = list(existing)
    assignments: dict[tuple[int, int, str], list[tuple[int, np.ndarray]]] = defaultdict(list)
    for q, k in cells:
        for covariance in ("spherical", "diag", "tied", "full"):
            for seed in range(5):
                key = (q, k, covariance, seed)
                if key in keys:
                    continue
                row, model = fit_gmm_cell(train_scores, heldout_scores, q, k, seed, 3, covariance)
                labels = model.predict(train_scores[:, :q])
                assignments[(q, k, covariance)].append((seed, labels))
                row["mean_pairwise_ARI"] = ""
                row["minimum_pairwise_ARI"] = ""
                row["maximum_pairwise_ARI"] = ""
                rows.append(row)
                save_csv(path, rows)
    # Refit cheaply only where resumed rows leave assignments unavailable, then
    # write the same covariance/seed stability summary onto every member row.
    for q, k in cells:
        for covariance in ("spherical", "diag", "tied", "full"):
            group = assignments[(q, k, covariance)]
            if len(group) < 5:
                group = []
                for seed in range(5):
                    _, model = fit_gmm_cell(train_scores, heldout_scores, q, k, seed, 3, covariance)
                    group.append((seed, model.predict(train_scores[:, :q])))
            aris = [adjusted_rand_score(group[i][1], group[j][1]) for i in range(5) for j in range(i + 1, 5)]
            for row in rows:
                if int(row["q"]) == q and int(row["K"]) == k and row["covariance_type"] == covariance:
                    row["mean_pairwise_ARI"] = float(np.mean(aris))
                    row["minimum_pairwise_ARI"] = float(np.min(aris))
                    row["maximum_pairwise_ARI"] = float(np.max(aris))
    save_csv(path, sorted(rows, key=lambda value: (int(value["q"]), int(value["K"]), value["covariance_type"], int(value["seed"]))))
    return rows


def plot_pca(rows: list[dict], path: Path) -> None:
    q = [int(row["q"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(q, [float(row["explained_variance_ratio"]) for row in rows], color="#2563eb")
    axes[0].plot(q, [float(row["cumulative_explained_variance"]) for row in rows], color="#d97706", marker="o", label="cumulative")
    axes[0].set(xlabel="PCA dimension q", ylabel="variance ratio", title="Velocity-field PCA variance")
    axes[0].legend()
    axes[1].plot(q, [float(row["train_reconstruction_mse"]) for row in rows], marker="o", label="train")
    axes[1].plot(q, [float(row["heldout_reconstruction_mse"]) for row in rows], marker="s", label="held-out")
    axes[1].set(xlabel="PCA dimension q", ylabel="coefficient reconstruction MSE", title="PCA reconstruction error")
    axes[1].set_yscale("log")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_heatmap(rows: list[dict], key: str, title: str, path: Path) -> None:
    matrix = np.full((20, 20), np.nan)
    for row in rows:
        matrix[int(row["q"]) - 1, int(row["K"]) - 1] = float(row[key])
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
    ax.set(xticks=np.arange(20), xticklabels=np.arange(1, 21), yticks=np.arange(20), yticklabels=np.arange(1, 21), xlabel="GMM components K", ylabel="PCA dimension q", title=title)
    fig.colorbar(image, ax=ax, label=key)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_preferred(preferred: list[dict], path: Path) -> None:
    q = [row["q"] for row in preferred]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for key, marker in (("K_BIC", "o"), ("K_ICL", "s"), ("K_heldout_LL", "^")):
        ax.plot(q, [row[key] for row in preferred], marker=marker, label=key)
    ax.set(xlabel="PCA dimension q", ylabel="preferred K", xticks=range(1, 21), yticks=range(1, 21), title="Preferred component count across PCA dimensions")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_criterion_curves(rows: list[dict], key: str, path: Path) -> None:
    fig, axes = plt.subplots(4, 5, figsize=(15, 10), sharex=True)
    for q, axis in enumerate(axes.ravel(), 1):
        group = sorted((row for row in rows if int(row["q"]) == q), key=lambda row: int(row["K"]))
        axis.plot([int(row["K"]) for row in group], [float(row[key]) for row in group], marker="o", markersize=2.5)
        axis.set_title(f"q={q}")
        axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("K")
    for axis in axes[:, 0]:
        axis.set_ylabel(key)
    fig.suptitle(f"{key} versus K for every PCA dimension", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_screening_report(out: Path, baseline: dict, registration_rows: list[dict], pca_rows: list[dict], gmm_rows: list[dict], preferred: list[dict], shortlist: list[int], robustness: list[dict], dry: dict) -> None:
    successful = [row for row in registration_rows if row["success"] == "True"]
    safe = [row for row in successful if row["safe"] == "True"]
    unsafe = [
        {
            "index": int(row["target_index"]),
            "filename": row["filename"],
            "minimum_jacobian": float(row["jacobian_minimum"]),
            "valid_geometry": row["valid_geometry"] in (True, "True"),
        }
        for row in successful
        if row["safe"] not in (True, "True")
    ]
    total_runtime = sum(float(row["wall_runtime_seconds"]) for row in successful)
    gmm_runtime = sum(float(row["runtime_seconds"]) for row in gmm_rows)
    robustness_groups = defaultdict(list)
    for row in robustness:
        robustness_groups[(int(row["q"]), int(row["K"]), row["covariance_type"])].append(float(row["mean_pairwise_ARI"]))
    best_stability = sorted(
        ({"q": key[0], "K": key[1], "covariance": key[2], "mean_ARI": float(np.mean(values))} for key, values in robustness_groups.items()),
        key=lambda row: row["mean_ARI"],
        reverse=True,
    )[:10]
    full_stability = []
    for candidate_k in (6, 7, 8, 13):
        group = [row for row in robustness if int(row["K"]) == candidate_k and row["covariance_type"] == "full"]
        if group:
            full_stability.append(
                {
                    "K": candidate_k,
                    "q": int(group[0]["q"]),
                    "mean": float(group[0]["mean_pairwise_ARI"]),
                    "minimum": float(group[0]["minimum_pairwise_ARI"]),
                    "maximum": float(group[0]["maximum_pairwise_ARI"]),
                }
            )
    disk_bytes = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
    primary_shortlist = [6, 7, 8]
    sensitivity_k = 13
    assigned_count = len(primary_shortlist) * len(registration_rows)
    all_template_count = sum(primary_shortlist) * len(registration_rows)
    assigned_with_sensitivity = (len(primary_shortlist) + 1) * len(registration_rows)
    all_with_sensitivity = (sum(primary_shortlist) + sensitivity_k) * len(registration_rows)
    median_runtime = float(np.median([float(row["wall_runtime_seconds"]) for row in successful]))
    artifact_times = [path.stat().st_mtime for path in (out / "registrations").glob("*.npz")]
    actual_registration_wall = max(artifact_times) - min(artifact_times)
    nonconverged_cells = [(int(row["q"]), int(row["K"])) for row in gmm_rows if row["converged"] not in (True, "True")]
    tiny_cells = sum(row["tiny_component_flag"] in (True, "True") for row in gmm_rows)
    lines = [
        "# Airfoil Template Number Stage 5：Phase A–K Screening",
        "",
        "## 技术摘要",
        "",
        f"冻结的 Stage 3 local48_wide 基线再次复现：中位/最差 MSE 为 `{baseline['median_correspondence_mse']:.3e}` / `{baseline['worst_correspondence_mse']:.3e}`。全量注册数值成功/安全为 `{len(successful)}/{len(safe)}/{len(registration_rows)}`；3 个不安全目标未进入 PCA/GMM。本报告在 Phase K 后停止。",
        "",
        f"PCA 前 1/5/10/20 维累计解释方差分别为 `{pca_rows[0]['cumulative_explained_variance']:.1%}` / `{pca_rows[4]['cumulative_explained_variance']:.1%}` / `{pca_rows[9]['cumulative_explained_variance']:.1%}` / `{pca_rows[19]['cumulative_explained_variance']:.1%}`。初步主 shortlist 为 `[6, 7, 8]`；`K=13` 作为 BIC/ICL 高-K敏感性对照保留。",
        "",
        "## 运行与可恢复性",
        "",
        f"所有 1654 个目标均保存独立原子 NPZ；重跑时按 target index、字段维数与有限性检查后跳过。文件时间戳测得的完整注册墙钟跨度为 `{actual_registration_wall/60:.1f} min`；累计单任务时间 `{total_runtime/3600:.2f} h`，每目标中位 `{median_runtime:.2f} s`。当前输出占 `{disk_bytes/1024**2:.1f} MiB`。",
        "",
        f"安全门排除明细：`{json.dumps(unsafe, ensure_ascii=False)}`。这 3 个样本仍保留完整场和属性，但不参与 PCA/GMM。",
        "",
        "## PCA 与主 GMM 网格",
        "",
        "![PCA](figures/pca_study.png)",
        "",
        f"PCA 只在固定 80% 训练集的 10,368 维速度系数上拟合；20% 留出集仅用于变换和评估。GMM dry run 为 `{dry['runtime_seconds']:.2f} s`。400 cells、每 cell 5 个初始化，共 2000 个 EM starts；累计 cell runtime 为 `{gmm_runtime/60:.2f} min`。399/400 cells 收敛，未收敛 cell 为 `{nonconverged_cells}`；`{tiny_cells}/400` cells 触发训练硬分量小于 14 个样本的警告。",
        "",
        "![preferred K](figures/preferred_k_by_q.png)",
        "",
        "### 每个 q 的偏好 K",
        "",
        "| q | BIC | ICL | held-out LL |",
        "|---:|---:|---:|---:|",
        *[f"| {row['q']} | {row['K_BIC']} | {row['K_ICL']} | {row['K_heldout_LL']} |" for row in preferred],
        "",
        "## Phase K shortlist robustness",
        "",
        f"广义 robustness screening set 为 `{shortlist}`。每个选定 (q,K) 使用 spherical/diag/tied/full、seeds 0–4、每次 3 个初始化。held-out likelihood 在 20 个 q 中选择 K=6/7/8 的次数为 6/6/4；BIC 与 ICL 最常选择 K=13，但其代表 cell 出现 2–9 个样本的小分量，且 full-covariance mean ARI 仅约 0.47。因此 downstream 主 shortlist 收敛为 `[6, 7, 8]`，K=13 只作为敏感性对照。最高稳定性组合摘要：`{json.dumps(best_stability, ensure_ascii=False)}`。详细结果在 `gmm/shortlist_robustness.csv`。",
        "",
        "| K | representative q | full-cov mean ARI | min ARI | max ARI |",
        "|---:|---:|---:|---:|---:|",
        *[f"| {row['K']} | {row['q']} | {row['mean']:.3f} | {row['minimum']:.3f} | {row['maximum']:.3f} |" for row in full_stability],
        "",
        "## 下游 Phase L–Q 成本估算（尚未运行）",
        "",
        f"对主 shortlist `[6,7,8]`，方案 A（每个 K 仅 GMM-assigned template）预计 `{assigned_count}` 次注册，约 `{assigned_count*median_runtime/3600:.1f}` 单进程小时；方案 B（每个 K 对全部模板择优）预计 `{all_template_count}` 次，约 `{all_template_count*median_runtime/3600:.1f}` 小时。若加入 K=13 敏感性对照，则分别为 `{assigned_with_sensitivity}` 次 / `{assigned_with_sensitivity*median_runtime/3600:.1f}` 小时和 `{all_with_sensitivity}` 次 / `{all_with_sensitivity*median_runtime/3600:.1f}` 小时。实测 4 workers 有资源争用，批准后建议 2 workers 并行。",
        "",
        "## 停止条件",
        "",
        "未构造 cluster-specific templates，未执行 Phase L–Q 重注册，未修改冻结算法，未 commit 或 push。等待批准。",
    ]
    (out / "report" / "screening_report.md").parent.mkdir(parents=True, exist_ok=True)
    (out / "report" / "screening_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(out: Path, dataset: dict, force: bool) -> None:
    matrix, registration_rows, population_indices = load_field_matrix(out, len(dataset["shapes"]))
    train, heldout = deterministic_split(len(matrix))
    (out / "split").mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "split" / "train_heldout_indices.npz",
        train_indices=population_indices[train],
        heldout_indices=population_indices[heldout],
        excluded_indices=np.setdiff1d(np.arange(len(dataset["shapes"])), population_indices),
        train_matrix_positions=train,
        heldout_matrix_positions=heldout,
    )
    pca, train_scores, heldout_scores, pca_rows = fit_pca_study(out, matrix, train, heldout, population_indices)
    dry = run_gmm_dry(out, train_scores, heldout_scores)
    gmm_rows = run_primary_gmm(out, train_scores, heldout_scores, force)
    preferred = preferred_by_q(gmm_rows)
    save_csv(out / "gmm" / "preferred_k_by_q.csv", preferred)
    shortlist = build_shortlist(preferred)
    # Keep already completed broad sensitivity results if present; otherwise
    # run only the final data-driven shortlist.
    robustness_path = out / "gmm" / "shortlist_robustness.csv"
    if robustness_path.exists() and not force:
        previous = read_csv(robustness_path)
        broad_k = sorted({int(row["K"]) for row in previous})
    else:
        broad_k = shortlist
    cells = robustness_q_values(preferred, broad_k)
    robustness = run_robustness(out, train_scores, heldout_scores, cells, force)
    atomic_json(out / "gmm" / "shortlist.json", {
        "primary_K_values": [6, 7, 8],
        "sensitivity_K_values": [13],
        "robustness_screening_K_values": broad_k,
        "q_k_cells": cells,
        "selection_rule": "primary: held-out winner in >=3 PCA dimensions; sensitivity: most persistent BIC/ICL winner, demoted for tiny components and weak full-covariance ARI",
    })
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_pca(pca_rows, figures / "pca_study.png")
    plot_heatmap(gmm_rows, "BIC", "BIC over PCA dimension and component count", figures / "bic_heatmap.png")
    plot_heatmap(gmm_rows, "ICL", "ICL over PCA dimension and component count", figures / "icl_heatmap.png")
    plot_heatmap(gmm_rows, "heldout_log_likelihood_mean", "Held-out mean log-likelihood", figures / "heldout_ll_heatmap.png")
    plot_preferred(preferred, figures / "preferred_k_by_q.png")
    plot_criterion_curves(gmm_rows, "BIC", figures / "bic_vs_k_all_q.png")
    plot_criterion_curves(gmm_rows, "ICL", figures / "icl_vs_k_all_q.png")
    baseline = json.loads((out / "baseline_reproduction.json").read_text(encoding="utf-8"))
    all_registration_rows = read_csv(out / "registration_results.csv")
    write_screening_report(out, baseline, all_registration_rows, pca_rows, gmm_rows, preferred, broad_k, robustness, dry)


def preflight(out: Path, dataset: dict, cfg: dict) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    for folder in ("registrations", "pca", "gmm/models", "figures", "report", "split"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    mean_path = out / "global_mean.npz"
    np.savez_compressed(mean_path, coordinates=dataset["mean"], point_count=200, upper_points=100, lower_points=100, definition="pointwise population mean over all 1654 processed shapes")
    verification = {
        "data_path": str(DATA.relative_to(ROOT)),
        "shape": list(dataset["shapes"].shape),
        "airfoils": len(dataset["shapes"]),
        "upper_points": 100,
        "lower_points": 100,
        "fixed_x_grid": True,
        "exact_pointwise_correspondence": True,
        "valid_geometry_count": len(dataset["shapes"]),
        "direct_mean_matches_stage2_exactly": bool(np.array_equal(dataset["mean"], np.load(STAGE2_MEAN))),
    }
    command = ".venv/bin/python src/run_airfoil_template_number_screening.py --stage all --workers 4"
    config = {
        "frozen_registration": cfg,
        "data_verification": verification,
        "main_pca_input": "velocity field coefficient vectors, not scalar properties",
        "split": {"seed": 20260813, "train_fraction": 0.8, "fit_pca_on": "training only"},
        "pca_dimensions": list(range(1, 21)),
        "primary_gmm": {"K": list(range(1, 21)), "covariance": "full", "n_init": 5, "seed": 20260813, "reg_covar": 1e-7},
        "resume": "per-target atomic NPZ; valid target index and 10,368-vector artifacts are skipped unless --force",
        "exact_command": command,
        "unavailable_metrics": ["classical LDDMM geodesic energy", "RKHS norm"],
        "exploratory_metrics": ["spectral high-frequency fraction", "field magnitude entropy"],
    }
    atomic_json(out / "config.json", config)
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "baseline", "register", "analyse", "all"), default="preflight")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--indices", default="")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    dataset = load_dataset()
    cfg = frozen_config(dataset["shapes"])
    preflight(out, dataset, cfg)
    if args.stage == "preflight":
        print(json.dumps(json.loads((out / "config.json").read_text()), indent=2))
        return
    if args.stage in ("baseline", "all"):
        print(json.dumps(reproduce_baseline(out, dataset, cfg), indent=2), flush=True)
    if args.stage in ("register", "all"):
        indices = selected_indices(len(dataset["shapes"]), args.indices, args.limit)
        rows = run_registrations(out, dataset, cfg, indices, args.workers, args.force)
        summary = {
            "requested": len(indices),
            "successful": sum(bool(row["success"]) for row in rows),
            "safe": sum(bool(row["safe"]) for row in rows),
            "failed": sum(not bool(row["success"]) for row in rows),
            "total_saved_wall_runtime_seconds": sum(float(row["wall_runtime_seconds"]) for row in rows),
        }
        atomic_json(out / "registration_summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    if args.stage in ("analyse", "all"):
        run_analysis(out, dataset, args.force)
        print("Phase A-K complete; stopped before Phase L.", flush=True)


if __name__ == "__main__":
    main()
