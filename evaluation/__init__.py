"""Evaluation metrics for classification and segmentation.

Everything here is NumPy-based so it's easy to inspect in isolation.
Core tools:
    - ConfusionMatrix class: accumulate predictions → derive OA, mAcc, IoU, mIoU
    - classification_metrics: OA + mean-class accuracy for classification
    - segmentation_metrics: OA + mean-class accuracy + per-class IoU + mIoU
    - part_segmentation_metrics: instance mIoU, per-object-class mean IoU
"""
from .metrics import (
    ConfusionMatrix,
    classification_metrics,
    segmentation_metrics,
    part_segmentation_metrics,
)

__all__ = [
    "ConfusionMatrix",
    "classification_metrics",
    "segmentation_metrics",
    "part_segmentation_metrics",
]
