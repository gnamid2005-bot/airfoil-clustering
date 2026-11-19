import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.use("Agg")


@dataclass
class ParsedAirfoil:
    name: str
    filename: str
    coordinates: np.ndarray
    original_point_count: int


@dataclass
class ProcessedAirfoil:
    name: str
    filename: str
    original_point_count: int
    raw: np.ndarray
    normalized: np.ndarray
    shape: np.ndarray


def parse_dat_file(path: Path, min_points: int = 20) -> ParsedAirfoil:
    name = path.stem
    coordinates = []
    saw_name = False

    for raw_line in path.read_text(encoding="latin-1", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.replace(",", " ").split()
        numeric = None
        if len(parts) >= 2:
            try:
                numeric = (float(parts[0]), float(parts[1]))
            except ValueError:
                numeric = None

        if numeric is None:
            if not saw_name:
                name = line
                saw_name = True
            continue

        coordinates.append(numeric)
        saw_name = True

    points = np.asarray(coordinates, dtype=np.float64)
    if points.shape[0] < min_points:
        raise ValueError(f"too few valid coordinate points: {points.shape[0]}")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"invalid coordinate array shape: {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("coordinates contain NaN or infinity")

    return ParsedAirfoil(
        name=name,
        filename=path.name,
        coordinates=points,
        original_point_count=int(points.shape[0]),
    )


def remove_consecutive_duplicates(points: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    if len(points) == 0:
        return points
    keep = [True]
    deltas = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep.extend(deltas > tol)
    return points[np.asarray(keep, dtype=bool)]


def normalize_airfoil(points: np.ndarray) -> np.ndarray:
    points = remove_consecutive_duplicates(points)
    if points.shape[0] < 20:
        raise ValueError("too few unique coordinate points")

    le_index = int(np.argmin(points[:, 0]))
    leading = points[le_index]
    trailing = 0.5 * (points[0] + points[-1])
    chord = trailing - leading
    chord_length = float(np.linalg.norm(chord))
    if chord_length <= 1e-8:
        raise ValueError("degenerate chord length")

    angle = math.atan2(chord[1], chord[0])
    rotation = np.asarray(
        [
            [math.cos(-angle), -math.sin(-angle)],
            [math.sin(-angle), math.cos(-angle)],
        ]
    )
    normalized = (points - leading) @ rotation.T / chord_length

    if normalized[:, 0].max() < 0.5:
        normalized[:, 0] *= -1.0

    x_min = float(normalized[:, 0].min())
    x_max = float(normalized[:, 0].max())
    normalized[:, 0] = (normalized[:, 0] - x_min) / max(x_max - x_min, 1e-12)

    if normalized[:, 0].min() > 1e-6 or normalized[:, 0].max() < 1.0 - 1e-6:
        raise ValueError("normalisation did not span the chord")

    return normalized


def split_surfaces(normalized: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    le_index = int(np.argmin(normalized[:, 0]))
    if le_index < 2 or le_index > normalized.shape[0] - 3:
        raise ValueError("leading edge split leaves an undersized surface")

    upper = normalized[: le_index + 1][::-1]
    lower = normalized[le_index:]
    if upper.shape[0] < 5 or lower.shape[0] < 5:
        raise ValueError("too few points on one surface")

    return upper, lower


def monotone_surface(surface: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(surface[:, 0], kind="mergesort")
    sorted_surface = surface[order]
    x = sorted_surface[:, 0]
    y = sorted_surface[:, 1]

    unique_x = []
    mean_y = []
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and abs(x[end] - x[start]) <= 1e-8:
            end += 1
        unique_x.append(float(np.mean(x[start:end])))
        mean_y.append(float(np.mean(y[start:end])))
        start = end

    x_unique = np.asarray(unique_x)
    y_unique = np.asarray(mean_y)
    if len(x_unique) < 5:
        raise ValueError("surface has too few unique x locations")
    if np.any(np.diff(x_unique) <= 0):
        raise ValueError("surface x coordinates are not strictly monotone")

    return x_unique, y_unique


def cosine_grid(n_surface_points: int) -> np.ndarray:
    if n_surface_points < 5:
        raise ValueError("--n-surface-points must be at least 5")
    theta = np.linspace(0.0, math.pi, n_surface_points)
    return (1.0 - np.cos(theta)) / 2.0


def resample_airfoil(normalized: np.ndarray, n_surface_points: int) -> np.ndarray:
    grid = cosine_grid(n_surface_points)
    upper, lower = split_surfaces(normalized)
    upper_x, upper_y = monotone_surface(upper)
    lower_x, lower_y = monotone_surface(lower)

    eps = 2e-3
    if upper_x[0] > eps or lower_x[0] > eps:
        raise ValueError("surface does not reach the leading edge")
    if upper_x[-1] < 1.0 - eps or lower_x[-1] < 1.0 - eps:
        raise ValueError("surface does not reach the trailing edge")

    upper_resampled_y = np.interp(grid, upper_x, upper_y)
    lower_resampled_y = np.interp(grid, lower_x, lower_y)

    if np.nanmean(upper_resampled_y - lower_resampled_y) < 0:
        upper_resampled_y, lower_resampled_y = lower_resampled_y, upper_resampled_y

    upper_points = np.column_stack([grid, upper_resampled_y])
    lower_points = np.column_stack([grid, lower_resampled_y])
    shape = np.vstack([upper_points, lower_points])
    if not np.isfinite(shape).all():
        raise ValueError("resampled shape contains NaN or infinity")
    if np.any(upper_resampled_y < lower_resampled_y - 5e-3):
        raise ValueError("inconsistent upper/lower orientation after resampling")
    return shape


def load_raw_airfoils(raw_dir: Path, min_points: int = 20) -> tuple[list[ParsedAirfoil], list[dict]]:
    files = sorted(raw_dir.rglob("*.dat"))
    parsed = []
    rejected = []
    for path in files:
        try:
            parsed.append(parse_dat_file(path, min_points=min_points))
        except Exception as exc:
            rejected.append(
                {
                    "filename": path.name,
                    "stage": "parse",
                    "reason": str(exc),
                }
            )
    return parsed, rejected


def preprocess_airfoils(
    raw_dir: Path,
    n_surface_points: int,
    max_airfoils: int | None = None,
    seed: int = 0,
) -> tuple[list[ProcessedAirfoil], list[dict], dict]:
    parsed, rejected = load_raw_airfoils(raw_dir)
    selected = parsed
    if max_airfoils is not None and len(parsed) > max_airfoils:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(parsed), size=max_airfoils, replace=False))
        selected = [parsed[int(index)] for index in indices]

    processed = []
    for airfoil in selected:
        try:
            normalized = normalize_airfoil(airfoil.coordinates)
            shape = resample_airfoil(normalized, n_surface_points)
            processed.append(
                ProcessedAirfoil(
                    name=airfoil.name,
                    filename=airfoil.filename,
                    original_point_count=airfoil.original_point_count,
                    raw=airfoil.coordinates,
                    normalized=normalized,
                    shape=shape,
                )
            )
        except Exception as exc:
            rejected.append(
                {
                    "filename": airfoil.filename,
                    "stage": "preprocess",
                    "reason": str(exc),
                }
            )

    counts = [airfoil.original_point_count for airfoil in parsed]
    summary = {
        "files_found": len(list(raw_dir.rglob("*.dat"))),
        "successfully_parsed": len(parsed),
        "selected_for_preprocessing": len(selected),
        "survived_preprocessing": len(processed),
        "rejected": len(rejected),
        "n_surface_points": n_surface_points,
        "min_original_points": int(np.min(counts)) if counts else 0,
        "median_original_points": float(np.median(counts)) if counts else 0.0,
        "max_original_points": int(np.max(counts)) if counts else 0,
    }
    return processed, rejected, summary


def flatten_shapes(shapes: np.ndarray) -> np.ndarray:
    if shapes.ndim != 3 or shapes.shape[2] != 2:
        raise ValueError(f"expected shape array (n, m, 2), got {shapes.shape}")
    return shapes.reshape(shapes.shape[0], -1)


def validate_processed_shapes(shapes: np.ndarray) -> None:
    if shapes.ndim != 3 or shapes.shape[2] != 2:
        raise ValueError(f"processed shapes must be (n, m, 2), got {shapes.shape}")
    if not np.isfinite(shapes).all():
        raise ValueError("processed shapes contain NaN or infinity")
    if not np.allclose(shapes[:, :, 0].min(axis=1), 0.0, atol=1e-8):
        raise ValueError("not all processed shapes have x_min approximately 0")
    if not np.allclose(shapes[:, :, 0].max(axis=1), 1.0, atol=1e-8):
        raise ValueError("not all processed shapes have x_max approximately 1")

    half = shapes.shape[1] // 2
    if np.any(shapes[:, :half, 1] < shapes[:, half:, 1] - 5e-3):
        raise ValueError("upper surface falls below lower surface")

    if shapes.shape[0] >= 2:
        flat = flatten_shapes(shapes[:2])
        flat_distance = float(np.sum((flat[0] - flat[1]) ** 2))
        pointwise = float(np.sum((shapes[0] - shapes[1]) ** 2))
        if not np.allclose(flat_distance, pointwise):
            raise ValueError("flattened and pointwise distances do not match")


def save_preprocessing_tables(
    output_dir: Path,
    processed: list[ProcessedAirfoil],
    rejected: list[dict],
    summary: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "preprocessing_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    with (output_dir / "processed_airfoils.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "airfoil_name", "original_point_count"],
        )
        writer.writeheader()
        for airfoil in processed:
            writer.writerow(
                {
                    "filename": airfoil.filename,
                    "airfoil_name": airfoil.name,
                    "original_point_count": airfoil.original_point_count,
                }
            )

    with (output_dir / "rejected_airfoils.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "stage", "reason"])
        writer.writeheader()
        writer.writerows(rejected)


def plot_shape(ax, shape: np.ndarray, **kwargs) -> None:
    half = shape.shape[0] // 2
    closed = np.vstack([shape[:half], shape[half:][::-1], shape[0:1]])
    ax.plot(closed[:, 0], closed[:, 1], **kwargs)


def save_sample_raw_vs_normalized(
    airfoils: list[ProcessedAirfoil],
    save_path: Path,
    seed: int,
    sample_size: int = 6,
) -> None:
    if not airfoils:
        return
    rng = np.random.default_rng(seed)
    count = min(sample_size, len(airfoils))
    indices = rng.choice(len(airfoils), size=count, replace=False)
    fig, axes = plt.subplots(count, 2, figsize=(9, 2.0 * count))
    if count == 1:
        axes = np.asarray([axes])

    for row, index in enumerate(indices):
        airfoil = airfoils[int(index)]
        raw = airfoil.raw
        axes[row, 0].plot(raw[:, 0], raw[:, 1], color="tab:gray", linewidth=1.0)
        axes[row, 0].set_title(f"{airfoil.filename} raw")
        axes[row, 1].plot(airfoil.shape[:, 0], airfoil.shape[:, 1], ".", color="tab:blue", markersize=2)
        plot_shape(axes[row, 1], airfoil.shape, color="tab:orange", linewidth=1.0)
        axes[row, 1].set_title("normalised and resampled")
        for ax in axes[row]:
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.25)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def save_sample_resampled(
    shapes: np.ndarray,
    save_path: Path,
    seed: int,
    sample_size: int = 30,
) -> None:
    if shapes.shape[0] == 0:
        return
    rng = np.random.default_rng(seed)
    count = min(sample_size, shapes.shape[0])
    indices = rng.choice(shapes.shape[0], size=count, replace=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    for index in indices:
        plot_shape(ax, shapes[int(index)], color="tab:blue", alpha=0.22, linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Random resampled airfoils (n={count})")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
