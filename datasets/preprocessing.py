"""Point cloud preprocessing utilities.

All functions operate on numpy arrays for simplicity. Shapes are [N, 3] for
coordinates and [N, C] for features. The dataset class converts to tensors
right before returning.
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple


def normalize_points(points: np.ndarray, return_center_scale: bool = False):
    """Center at origin and scale to unit ball.

    Args:
        points: [N, 3] xyz.
        return_center_scale: if True also return (center, scale) for inversion.
    """
    center = points.mean(axis=0, keepdims=True)
    centered = points - center
    # max distance from origin after centering
    scale = np.max(np.linalg.norm(centered, axis=1))
    if scale < 1e-8:
        scale = 1.0
    normed = centered / scale
    if return_center_scale:
        return normed, center.squeeze(0), float(scale)
    return normed


def random_sample(points: np.ndarray, num_points: int,
                  features: Optional[np.ndarray] = None,
                  labels: Optional[np.ndarray] = None,
                  replace_if_short: bool = True,
                  rng: Optional[np.random.Generator] = None):
    """Randomly sample `num_points` rows from `points` (and parallel arrays).

    If the cloud has fewer points than requested, sample with replacement (unless
    `replace_if_short=False`, in which case we raise).
    """
    rng = rng or np.random.default_rng()
    n = points.shape[0]
    if n >= num_points:
        idx = rng.choice(n, size=num_points, replace=False)
    else:
        if not replace_if_short:
            raise ValueError(f"Cloud has {n} < {num_points} points.")
        idx = rng.choice(n, size=num_points, replace=True)
    out_pts = points[idx]
    out_feats = features[idx] if features is not None else None
    out_labels = labels[idx] if labels is not None else None
    return out_pts, out_feats, out_labels, idx


def farthest_point_sample(points: np.ndarray, num_points: int,
                          rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Farthest Point Sampling (FPS). Returns indices of shape [num_points].

    Pure-numpy implementation — fine for toy/medium clouds. For production use,
    swap this with a CUDA FPS op.
    """
    rng = rng or np.random.default_rng()
    n = points.shape[0]
    if n <= num_points:
        # not enough points — pad by resampling
        pad = rng.choice(n, size=num_points - n, replace=True)
        return np.concatenate([np.arange(n), pad])

    selected = np.empty(num_points, dtype=np.int64)
    # initialize distances to +inf, pick a random starting point
    dist = np.full(n, np.inf, dtype=np.float32)
    start = int(rng.integers(0, n))
    selected[0] = start
    last = points[start]
    for i in range(1, num_points):
        new_dist = np.sum((points - last) ** 2, axis=1)
        dist = np.minimum(dist, new_dist)
        next_idx = int(np.argmax(dist))
        selected[i] = next_idx
        last = points[next_idx]
    return selected


def pad_cloud(points: np.ndarray, num_points: int,
              features: Optional[np.ndarray] = None,
              labels: Optional[np.ndarray] = None,
              pad_value: float = 0.0):
    """Pad or truncate a cloud to exactly `num_points`. Used when batching
    variable-sized clouds without a collate_fn."""
    n = points.shape[0]
    if n == num_points:
        return points, features, labels
    if n > num_points:
        return points[:num_points], (features[:num_points] if features is not None else None), \
               (labels[:num_points] if labels is not None else None)
    # n < num_points → pad
    pad_pts = np.full((num_points - n, points.shape[1]), pad_value, dtype=points.dtype)
    points = np.concatenate([points, pad_pts], axis=0)
    if features is not None:
        pad_f = np.full((num_points - n, features.shape[1]), pad_value, dtype=features.dtype)
        features = np.concatenate([features, pad_f], axis=0)
    if labels is not None:
        pad_l = np.full((num_points - n,), -1, dtype=labels.dtype)  # -1 = ignore
        labels = np.concatenate([labels, pad_l], axis=0)
    return points, features, labels
