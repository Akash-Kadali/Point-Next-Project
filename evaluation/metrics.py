"""Classification / segmentation metrics.

We handle the "missing class" case everywhere: if a class has zero ground-truth
samples in the evaluation pool, per-class IoU is returned as NaN and the mean
skips NaNs.

Shapes (all NumPy):
    - Classification:
        preds, labels: [B] — one label per object.
    - Segmentation (scene):
        preds, labels: [B, N] or concatenated [B*N] — one label per point.
    - Part segmentation:
        preds, labels, obj_labels: [B, N], [B, N], [B]

Ignore index:
    Any label == `ignore_index` (default -1) is excluded from accounting.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Sequence
import numpy as np


# --------------------------------------------------------------------------- #
# Confusion matrix — the common backbone for OA / mAcc / IoU / mIoU
# --------------------------------------------------------------------------- #
class ConfusionMatrix:
    """Accumulate a KxK confusion matrix across batches.

    Rows = ground truth, columns = predictions.
    """

    def __init__(self, num_classes: int, ignore_index: int = -1) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self) -> None:
        self.matrix.fill(0)

    def update(self, preds: np.ndarray, labels: np.ndarray) -> None:
        """Add predictions to the running matrix. Both arrays are flattened."""
        preds = np.asarray(preds).reshape(-1)
        labels = np.asarray(labels).reshape(-1)
        mask = labels != self.ignore_index
        preds = preds[mask]
        labels = labels[mask]
        # bincount trick: 2D index = label * K + pred
        idx = labels * self.num_classes + preds
        counts = np.bincount(idx, minlength=self.num_classes ** 2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes)

    # -- derived metrics --------------------------------------------------- #
    def overall_accuracy(self) -> float:
        total = self.matrix.sum()
        if total == 0:
            return float("nan")
        return float(np.trace(self.matrix)) / float(total)

    def per_class_accuracy(self) -> np.ndarray:
        """Accuracy per ground-truth class. NaN if that class never appeared."""
        row_sums = self.matrix.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            acc = np.diag(self.matrix) / row_sums
        acc[row_sums == 0] = np.nan
        return acc

    def mean_class_accuracy(self) -> float:
        return float(np.nanmean(self.per_class_accuracy()))

    def per_class_iou(self) -> np.ndarray:
        """Jaccard per class: TP / (TP + FP + FN). NaN if union is empty."""
        tp = np.diag(self.matrix).astype(np.float64)
        fp = self.matrix.sum(axis=0) - tp
        fn = self.matrix.sum(axis=1) - tp
        union = tp + fp + fn
        with np.errstate(invalid="ignore", divide="ignore"):
            iou = tp / union
        iou[union == 0] = np.nan
        return iou

    def mean_iou(self) -> float:
        return float(np.nanmean(self.per_class_iou()))


# --------------------------------------------------------------------------- #
# Convenience wrappers per task
# --------------------------------------------------------------------------- #
def classification_metrics(preds: np.ndarray, labels: np.ndarray,
                           num_classes: int) -> Dict[str, float]:
    """OA + mean-class accuracy for object classification."""
    cm = ConfusionMatrix(num_classes=num_classes)
    cm.update(preds, labels)
    return {
        "overall_accuracy": cm.overall_accuracy(),
        "mean_class_accuracy": cm.mean_class_accuracy(),
    }


def segmentation_metrics(preds: np.ndarray, labels: np.ndarray,
                         num_classes: int,
                         ignore_index: int = -1) -> Dict[str, float]:
    """OA + mAcc + per-class IoU + mIoU for semantic segmentation."""
    cm = ConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index)
    cm.update(preds, labels)
    out = {
        "overall_accuracy": cm.overall_accuracy(),
        "mean_class_accuracy": cm.mean_class_accuracy(),
        "mean_iou": cm.mean_iou(),
    }
    per_iou = cm.per_class_iou()
    for c in range(num_classes):
        out[f"iou_class_{c}"] = float(per_iou[c]) if not np.isnan(per_iou[c]) else float("nan")
    return out


def part_segmentation_metrics(preds: Sequence[np.ndarray],
                              labels: Sequence[np.ndarray],
                              obj_labels: Sequence[int],
                              parts_per_obj: Dict[int, List[int]],
                              ignore_index: int = -1) -> Dict[str, float]:
    """Instance mIoU, following ShapeNetPart convention.

    For each object (one item in the lists), compute IoU per part that actually
    belongs to this object class (per `parts_per_obj[obj_class]`). Take mean
    over those parts to get the object's IoU. Report:
        - instance_miou: mean of per-object IoUs (this is "Ins. mIoU" in the paper).
        - class_miou: mean over object-class of mean-object-IoU.

    Args:
        preds: list of per-point part predictions, one array per object.
        labels: list of per-point ground-truth part labels, one per object.
        obj_labels: list of object class ids, one per object.
        parts_per_obj: {obj_class: [valid part ids for that class]}.
    """
    object_ious: List[float] = []
    per_class_ious: Dict[int, List[float]] = {c: [] for c in parts_per_obj}

    for pred, lbl, obj in zip(preds, labels, obj_labels):
        pred = np.asarray(pred).reshape(-1)
        lbl = np.asarray(lbl).reshape(-1)
        valid = lbl != ignore_index
        pred, lbl = pred[valid], lbl[valid]
        valid_parts = parts_per_obj[int(obj)]
        part_ious = []
        for p in valid_parts:
            pred_p = pred == p
            lbl_p = lbl == p
            if not pred_p.any() and not lbl_p.any():
                part_ious.append(1.0)  # absent part, perfect
                continue
            inter = np.logical_and(pred_p, lbl_p).sum()
            union = np.logical_or(pred_p, lbl_p).sum()
            part_ious.append(float(inter) / float(union) if union > 0 else 0.0)
        obj_iou = float(np.mean(part_ious)) if part_ious else float("nan")
        object_ious.append(obj_iou)
        per_class_ious[int(obj)].append(obj_iou)

    class_miou = float(np.nanmean(
        [np.nanmean(v) if v else float("nan") for v in per_class_ious.values()]
    ))
    return {
        "instance_miou": float(np.nanmean(object_ious)) if object_ious else float("nan"),
        "class_miou": class_miou,
        "num_objects": len(object_ious),
    }
