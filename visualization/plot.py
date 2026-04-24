"""Point cloud plotting with matplotlib.

All functions accept numpy arrays. Points are [N, 3], labels are [N] ints
or [N, 3] colors.

Usage:
    plot_points(xyz, save_path='cloud.png')
    plot_points_labeled(xyz, labels, class_names=[...], save_path='seg.png')
    plot_pred_vs_gt(xyz, pred_labels, gt_labels, save_path='comp.png')
    plot_before_after(xyz_before, xyz_after, title_before='raw', title_after='augmented')
"""
from __future__ import annotations
from typing import List, Optional, Sequence, Tuple, Union
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _as_numpy(x) -> np.ndarray:
    """Accept torch tensor or numpy array."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def _set_equal_aspect(ax) -> None:
    """3D matplotlib doesn't have set_aspect('equal'); fake it by setting equal
    box limits."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    sizes = limits[:, 1] - limits[:, 0]
    center = limits.mean(axis=1)
    max_size = sizes.max()
    for setter, c in zip(("set_xlim3d", "set_ylim3d", "set_zlim3d"), center):
        getattr(ax, setter)(c - max_size / 2, c + max_size / 2)


def _finish(fig, save_path: Optional[str], show: bool) -> None:
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _colors_from_labels(labels: np.ndarray,
                        num_classes: Optional[int] = None) -> np.ndarray:
    """Map integer labels to RGB colors via tab20."""
    labels = labels.astype(int)
    if num_classes is None:
        num_classes = int(labels.max()) + 1
    cmap = cm.get_cmap("tab20", max(num_classes, 2))
    return cmap(labels % cmap.N)[:, :3]


# --------------------------------------------------------------------------- #
# Public plotting functions
# --------------------------------------------------------------------------- #
def plot_points(points, colors: Optional[np.ndarray] = None,
                title: str = "Points", save_path: Optional[str] = None,
                show: bool = False, point_size: float = 1.0) -> None:
    """Plot raw points, optionally colored."""
    pts = _as_numpy(points)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    if colors is None:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=point_size, c=pts[:, 2])
    else:
        colors = _as_numpy(colors)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=point_size, c=colors)
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    _set_equal_aspect(ax)
    _finish(fig, save_path, show)


def plot_points_labeled(points, labels, num_classes: Optional[int] = None,
                        class_names: Optional[Sequence[str]] = None,
                        title: str = "Labeled Points",
                        save_path: Optional[str] = None,
                        show: bool = False, point_size: float = 1.5) -> None:
    """Plot points colored by integer labels."""
    pts = _as_numpy(points)
    labs = _as_numpy(labels).astype(int)
    colors = _colors_from_labels(labs, num_classes=num_classes)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=point_size, c=colors)
    ax.set_title(title)
    _set_equal_aspect(ax)

    # Legend (only if class names provided, and not too many)
    if class_names is not None and len(class_names) <= 20:
        present = np.unique(labs)
        palette = _colors_from_labels(np.arange(len(class_names)),
                                      num_classes=len(class_names))
        handles = [plt.Line2D([], [], marker="o", linestyle="",
                              color=palette[c], label=class_names[c])
                   for c in present if 0 <= c < len(class_names)]
        ax.legend(handles=handles, loc="upper right", fontsize=8)
    _finish(fig, save_path, show)


def plot_pred_vs_gt(points, pred_labels, gt_labels,
                    num_classes: Optional[int] = None,
                    class_names: Optional[Sequence[str]] = None,
                    save_path: Optional[str] = None, show: bool = False,
                    point_size: float = 1.5) -> None:
    """Side-by-side comparison of prediction vs ground truth."""
    pts = _as_numpy(points)
    pred = _as_numpy(pred_labels).astype(int)
    gt = _as_numpy(gt_labels).astype(int)

    nc = num_classes or int(max(pred.max(), gt.max()) + 1)
    pred_colors = _colors_from_labels(pred, num_classes=nc)
    gt_colors = _colors_from_labels(gt, num_classes=nc)

    fig = plt.figure(figsize=(13, 6))
    for i, (labs, cols, title) in enumerate(
            [(pred, pred_colors, "Prediction"), (gt, gt_colors, "Ground Truth")]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=point_size, c=cols)
        ax.set_title(title)
        _set_equal_aspect(ax)
    _finish(fig, save_path, show)


def plot_before_after(points_before, points_after,
                      title_before: str = "Before",
                      title_after: str = "After",
                      save_path: Optional[str] = None, show: bool = False,
                      point_size: float = 1.5) -> None:
    """Side-by-side comparison for preprocessing / augmentation visualization."""
    before = _as_numpy(points_before)
    after = _as_numpy(points_after)
    fig = plt.figure(figsize=(13, 6))
    for i, (pts, title) in enumerate(
            [(before, title_before), (after, title_after)]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=point_size, c=pts[:, 2])
        ax.set_title(title)
        _set_equal_aspect(ax)
    _finish(fig, save_path, show)
