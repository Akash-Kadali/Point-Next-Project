"""Synthetic point cloud datasets so the project runs without external data.

Three classes are provided, one per task. Each mimics the interface a real
dataset class should expose; the only difference is `load_sample` generates
procedurally instead of reading from disk.

Replace `load_sample` in a subclass to use S3DIS / ScanObjectNN / ShapeNetPart.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import normalize_points, random_sample
from .augmentation import Compose


# --------------------------------------------------------------------------- #
# Shape generators
# --------------------------------------------------------------------------- #
def _sample_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    x = rng.standard_normal((n, 3)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x


def _sample_cube(n: int, rng: np.random.Generator) -> np.ndarray:
    pts = rng.uniform(-1, 1, size=(n, 3)).astype(np.float32)
    # project to surface (move the largest-abs axis to ±1)
    idx = np.argmax(np.abs(pts), axis=1)
    for i, j in enumerate(idx):
        pts[i, j] = np.sign(pts[i, j])
    return pts


def _sample_cylinder(n: int, rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(0, 2 * np.pi, size=n).astype(np.float32)
    z = rng.uniform(-1, 1, size=n).astype(np.float32)
    return np.stack([np.cos(theta), np.sin(theta), z], axis=1)


def _sample_cone(n: int, rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(0, 2 * np.pi, size=n).astype(np.float32)
    h = rng.uniform(0, 1, size=n).astype(np.float32)
    r = 1 - h
    return np.stack([r * np.cos(theta), r * np.sin(theta), h], axis=1)


_SHAPE_GENS = {
    0: _sample_sphere,
    1: _sample_cube,
    2: _sample_cylinder,
    3: _sample_cone,
}


# --------------------------------------------------------------------------- #
# Classification: one class label per object
# --------------------------------------------------------------------------- #
class ToyClassificationDataset(Dataset):
    """Each sample: one of 4 toy shapes + random xyz jitter. Label = shape id."""

    NUM_CLASSES = 4
    CLASS_NAMES = ["sphere", "cube", "cylinder", "cone"]

    def __init__(self, split: str = "train", num_samples: int = 256,
                 num_points: int = 1024, transform: Optional[Compose] = None,
                 seed: int = 0):
        self.split = split
        self.num_samples = num_samples
        self.num_points = num_points
        self.transform = transform
        # precompute class labels for determinism (train vs val use different seeds)
        rng = np.random.default_rng(seed + (0 if split == "train" else 1))
        self.class_labels = rng.integers(0, self.NUM_CLASSES, size=num_samples)
        self._base_seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def load_sample(self, idx: int) -> Dict[str, Any]:
        """Override in subclasses to load from disk."""
        rng = np.random.default_rng(self._base_seed + idx * 31 +
                                    (0 if self.split == "train" else 777))
        cls = int(self.class_labels[idx])
        pts = _SHAPE_GENS[cls](self.num_points, rng)
        pts += rng.standard_normal(pts.shape).astype(np.float32) * 0.01  # small noise
        pts = normalize_points(pts)
        return {
            "points": pts.astype(np.float32),
            "features": None,
            "labels": int(cls),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.load_sample(idx)
        if self.transform:
            sample = self.transform(sample)
        pts = torch.from_numpy(sample["points"]).float()  # [N, 3]
        feats = sample.get("features")
        feats_t = torch.from_numpy(feats).float() if feats is not None else pts.clone()
        return {
            "xyz": pts,           # [N, 3]
            "features": feats_t,  # [N, C]
            "label": torch.tensor(sample["labels"], dtype=torch.long),  # scalar
        }


# --------------------------------------------------------------------------- #
# Semantic segmentation: one class label per point
# --------------------------------------------------------------------------- #
class ToySegmentationDataset(Dataset):
    """Toy 'scene' = mixture of 3 shapes, each colored and labeled distinctly.

    Emulates S3DIS-style input: xyz + RGB features, per-point class labels.
    """

    NUM_CLASSES = 3
    CLASS_NAMES = ["ground", "wall", "furniture"]

    def __init__(self, split: str = "train", num_samples: int = 64,
                 num_points: int = 2048, transform: Optional[Compose] = None,
                 with_color: bool = True, seed: int = 0):
        self.split = split
        self.num_samples = num_samples
        self.num_points = num_points
        self.transform = transform
        self.with_color = with_color
        self._base_seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def load_sample(self, idx: int) -> Dict[str, Any]:
        rng = np.random.default_rng(self._base_seed + idx * 101 +
                                    (0 if self.split == "train" else 999))
        # ground: flat XY plane (z near 0)
        n_g = self.num_points // 3
        g = np.column_stack([
            rng.uniform(-1, 1, n_g),
            rng.uniform(-1, 1, n_g),
            rng.uniform(-0.02, 0.02, n_g),
        ]).astype(np.float32)
        # wall: vertical plane (x near -1)
        n_w = self.num_points // 3
        w = np.column_stack([
            rng.uniform(-1, -0.98, n_w),
            rng.uniform(-1, 1, n_w),
            rng.uniform(0, 1, n_w),
        ]).astype(np.float32)
        # furniture: blob somewhere in middle
        n_f = self.num_points - n_g - n_w
        c = rng.uniform(-0.5, 0.5, 3)
        f = c + rng.standard_normal((n_f, 3)).astype(np.float32) * 0.15
        f[:, 2] = np.abs(f[:, 2])

        pts = np.concatenate([g, w, f], axis=0)
        lbl = np.concatenate([
            np.zeros(n_g, dtype=np.int64),
            np.ones(n_w, dtype=np.int64),
            np.full(n_f, 2, dtype=np.int64),
        ])
        # shuffle
        perm = rng.permutation(pts.shape[0])
        pts, lbl = pts[perm], lbl[perm]

        feats = None
        if self.with_color:
            # color is a cue correlated with label, but noisy
            colors = np.zeros((pts.shape[0], 3), dtype=np.float32)
            colors[lbl == 0] = np.array([0.6, 0.4, 0.2])  # brown ground
            colors[lbl == 1] = np.array([0.9, 0.9, 0.9])  # white wall
            colors[lbl == 2] = np.array([0.2, 0.5, 0.8])  # blue furniture
            colors += rng.standard_normal(colors.shape).astype(np.float32) * 0.1
            colors = np.clip(colors, 0, 1)
            feats = colors

        pts = normalize_points(pts)
        return {
            "points": pts.astype(np.float32),
            "features": feats,
            "labels": lbl,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.load_sample(idx)
        if self.transform:
            sample = self.transform(sample)
        pts = torch.from_numpy(sample["points"]).float()
        feats = sample.get("features")
        feats_t = torch.from_numpy(feats).float() if feats is not None else pts.clone()
        labels = torch.from_numpy(sample["labels"]).long()
        return {"xyz": pts, "features": feats_t, "label": labels}


# --------------------------------------------------------------------------- #
# Part segmentation: object-level point labels, plus an object class
# --------------------------------------------------------------------------- #
class ToyPartSegmentationDataset(Dataset):
    """Toy part segmentation: a shape is split into 2 parts by a plane.

    Mimics ShapeNetPart: one object class per sample, per-point part label.
    """

    NUM_OBJECT_CLASSES = 4
    NUM_PARTS = 2

    def __init__(self, split: str = "train", num_samples: int = 128,
                 num_points: int = 1024, transform: Optional[Compose] = None,
                 seed: int = 0):
        self.split = split
        self.num_samples = num_samples
        self.num_points = num_points
        self.transform = transform
        rng = np.random.default_rng(seed + (0 if split == "train" else 3))
        self.obj_labels = rng.integers(0, self.NUM_OBJECT_CLASSES, size=num_samples)
        self._base_seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def load_sample(self, idx: int) -> Dict[str, Any]:
        rng = np.random.default_rng(self._base_seed + idx * 59 +
                                    (0 if self.split == "train" else 13))
        obj_cls = int(self.obj_labels[idx])
        pts = _SHAPE_GENS[obj_cls](self.num_points, rng)
        pts += rng.standard_normal(pts.shape).astype(np.float32) * 0.02
        pts = normalize_points(pts)

        # Part label: 0 if z < 0 else 1 (toy split)
        part_lbl = (pts[:, 2] >= 0).astype(np.int64)

        # Optional per-point normals (approx sphere normals = pts themselves)
        normals = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-8)

        return {
            "points": pts.astype(np.float32),
            "features": normals.astype(np.float32),
            "labels": part_lbl,
            "obj_label": obj_cls,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.load_sample(idx)
        if self.transform:
            sample = self.transform(sample)
        pts = torch.from_numpy(sample["points"]).float()
        feats = sample.get("features")
        feats_t = torch.from_numpy(feats).float() if feats is not None else pts.clone()
        labels = torch.from_numpy(sample["labels"]).long()
        return {
            "xyz": pts,
            "features": feats_t,
            "label": labels,
            "obj_label": torch.tensor(sample["obj_label"], dtype=torch.long),
        }
