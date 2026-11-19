import json
from pathlib import Path

import numpy as np


FAMILY_NAMES_TWO = np.asarray(["ellipse", "rectangle"])
FAMILY_NAMES_THREE = np.asarray(["ellipse", "rectangle", "octagon"])


def polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def polygon_perimeter(points: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)))


def is_convex_polygon(points: np.ndarray) -> bool:
    edges_a = np.roll(points, -1, axis=0) - points
    edges_b = np.roll(points, -2, axis=0) - np.roll(points, -1, axis=0)
    cross = edges_a[:, 0] * edges_b[:, 1] - edges_a[:, 1] * edges_b[:, 0]
    return bool(np.all(cross > 1e-10) or np.all(cross < -1e-10))


def center(points: np.ndarray) -> np.ndarray:
    return points - points.mean(axis=0, keepdims=True)


def ensure_ccw(points: np.ndarray) -> np.ndarray:
    return points if polygon_area(points) > 0.0 else points[::-1].copy()


def rotate_to_rightmost_start(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    max_x = np.max(x)
    candidates = np.flatnonzero(np.isclose(x, max_x, rtol=0.0, atol=1e-12))
    if len(candidates) > 1:
        scores = np.abs(points[candidates, 1])
        start = int(candidates[np.argmin(scores)])
    else:
        start = int(candidates[0])
    return np.roll(points, -start, axis=0)


def resample_closed_loop(points: np.ndarray, m: int) -> np.ndarray:
    points = ensure_ccw(center(points))
    points = rotate_to_rightmost_start(points)
    closed = np.vstack([points, points[0]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    perimeter = cumulative[-1]
    if perimeter <= 0.0:
        raise ValueError("Cannot resample a zero-perimeter loop")

    targets = np.linspace(0.0, perimeter, m, endpoint=False)
    sampled = np.empty((m, 2), dtype=np.float64)
    for dim in range(2):
        sampled[:, dim] = np.interp(targets, cumulative, closed[:, dim])
    sampled = center(ensure_ccw(sampled))
    return rotate_to_rightmost_start(sampled)


def scale_to_perimeter(points: np.ndarray, target_perimeter: float) -> np.ndarray:
    perimeter = polygon_perimeter(points)
    if perimeter <= 0.0:
        raise ValueError("Cannot scale a zero-perimeter loop")
    return points * (target_perimeter / perimeter)


def ellipse_dense(aspect_ratio: float, dense_points: int) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, dense_points, endpoint=False)
    a = aspect_ratio
    b = 1.0
    return np.column_stack([a * np.cos(theta), b * np.sin(theta)])


def rectangle_dense(aspect_ratio: float, dense_per_side: int) -> np.ndarray:
    width = 1.0
    length = aspect_ratio
    half_l = 0.5 * length
    half_w = 0.5 * width
    edges = [
        np.column_stack([np.linspace(half_l, -half_l, dense_per_side, endpoint=False), np.full(dense_per_side, half_w)]),
        np.column_stack([np.full(dense_per_side, -half_l), np.linspace(half_w, -half_w, dense_per_side, endpoint=False)]),
        np.column_stack([np.linspace(-half_l, half_l, dense_per_side, endpoint=False), np.full(dense_per_side, -half_w)]),
        np.column_stack([np.full(dense_per_side, half_l), np.linspace(-half_w, half_w, dense_per_side, endpoint=False)]),
    ]
    return np.vstack(edges)


def octagon_dense(aspect_ratio: float, rng: np.random.Generator, jitter: float) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    for _ in range(100):
        radii = np.clip(1.0 + rng.normal(0.0, jitter, size=8), 0.90, 1.10)
        points = np.column_stack([radii * np.cos(theta) * aspect_ratio, radii * np.sin(theta)])
        order = np.argsort(np.arctan2(points[:, 1], points[:, 0]))
        points = points[order]
        if is_convex_polygon(points):
            return points
    points = np.column_stack([np.cos(theta) * aspect_ratio, np.sin(theta)])
    return points[np.argsort(np.arctan2(points[:, 1], points[:, 0]))]


def make_sample(family: str, rng: np.random.Generator, m: int, target_perimeter: float, dense_points: int) -> tuple[np.ndarray, dict]:
    aspect_ratio = float(rng.uniform(1.1, 3.0))
    if family == "ellipse":
        dense = ellipse_dense(aspect_ratio, dense_points)
        params = {"family": family, "aspect_ratio": aspect_ratio}
    elif family == "rectangle":
        dense = rectangle_dense(aspect_ratio, max(16, dense_points // 4))
        params = {"family": family, "aspect_ratio": aspect_ratio}
    elif family == "octagon":
        dense = octagon_dense(aspect_ratio, rng, jitter=0.03)
        params = {"family": family, "aspect_ratio": aspect_ratio, "corner_jitter_std": 0.03}
    else:
        raise ValueError(f"Unknown synthetic family: {family}")

    dense = scale_to_perimeter(center(ensure_ccw(dense)), target_perimeter)
    sampled = resample_closed_loop(dense, m)
    sampled = scale_to_perimeter(center(ensure_ccw(sampled)), target_perimeter)
    sampled = rotate_to_rightmost_start(center(ensure_ccw(sampled)))
    params["perimeter_after_sampling"] = polygon_perimeter(sampled)
    return sampled, params


def generate_dataset(
    counts: dict[str, int],
    seed: int,
    m: int = 200,
    target_perimeter: float = 4.0,
    dense_points: int = 2000,
) -> dict:
    rng = np.random.default_rng(seed)
    shapes = []
    labels = []
    params = []
    family_names = np.asarray(list(counts.keys()))
    for label, family in enumerate(family_names):
        for _ in range(counts[str(family)]):
            shape, sample_params = make_sample(str(family), rng, m, target_perimeter, dense_points)
            shapes.append(shape)
            labels.append(label)
            params.append(sample_params)

    shapes = np.asarray(shapes, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    order = rng.permutation(len(labels))
    return {
        "shapes": shapes[order],
        "true_labels": labels[order],
        "family_names": family_names,
        "generation_parameters": np.asarray([json.dumps(params[i], sort_keys=True) for i in order]),
        "perimeter": np.asarray(target_perimeter, dtype=np.float64),
        "seed": np.asarray(seed, dtype=np.int64),
    }


def save_dataset(path: Path, dataset: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dataset)


def load_dataset(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}
