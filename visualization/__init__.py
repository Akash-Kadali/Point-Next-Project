"""Matplotlib-based point cloud visualization.

Open3D is a nicer tool but not always easy to install, so this module uses
matplotlib 3D scatter by default. All functions have a `show=False, save_path=None`
pattern: pass a path to save to disk without opening a window.
"""
from .plot import (
    plot_points,
    plot_points_labeled,
    plot_pred_vs_gt,
    plot_before_after,
)

__all__ = [
    "plot_points",
    "plot_points_labeled",
    "plot_pred_vs_gt",
    "plot_before_after",
]
