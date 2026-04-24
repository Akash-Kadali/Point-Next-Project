"""Data augmentations studied in the PointNeXt paper.

Every augmentation takes and returns a `sample` dict with keys:
    - points: [N, 3] xyz (numpy)
    - features: [N, C] optional (numpy or None)
    - labels: [N] or scalar, optional
Each augmentation acts on the fields it needs and leaves others alone.

The `build_augmentation` factory takes a config block and returns a `Compose`.
This is how we toggle individual augmentations on/off for ablations.
"""
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


Sample = Dict[str, Any]


class Compose:
    """Apply a list of augmentations in order."""

    def __init__(self, transforms: List[Callable[[Sample], Sample]]):
        self.transforms = transforms

    def __call__(self, sample: Sample) -> Sample:
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self) -> str:
        names = ", ".join(type(t).__name__ for t in self.transforms)
        return f"Compose([{names}])"


# --------------------------------------------------------------------------- #
# Geometric augmentations
# --------------------------------------------------------------------------- #
class RandomRotation:
    """Rotate around the gravity (Z or Y) axis by a random angle."""

    def __init__(self, axis: str = "z", angle_range: Tuple[float, float] = (0.0, 2 * np.pi)):
        self.axis = axis.lower()
        self.lo, self.hi = angle_range

    def __call__(self, sample: Sample) -> Sample:
        theta = np.random.uniform(self.lo, self.hi)
        c, s = np.cos(theta), np.sin(theta)
        if self.axis == "z":
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        elif self.axis == "y":
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        else:  # x
            R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
        sample["points"] = sample["points"] @ R.T
        return sample


class RandomScaling:
    """Per-sample isotropic scaling factor in [lo, hi]."""

    def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.2)):
        self.lo, self.hi = scale_range

    def __call__(self, sample: Sample) -> Sample:
        s = np.random.uniform(self.lo, self.hi)
        sample["points"] = sample["points"] * s
        return sample


class RandomTranslation:
    """Per-sample xyz translation in [-t, t] each."""

    def __init__(self, shift: float = 0.1):
        self.shift = shift

    def __call__(self, sample: Sample) -> Sample:
        t = np.random.uniform(-self.shift, self.shift, size=(1, 3)).astype(np.float32)
        sample["points"] = sample["points"] + t
        return sample


class RandomJitter:
    """Add independent Gaussian noise to each point, clipped."""

    def __init__(self, sigma: float = 0.01, clip: float = 0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, sample: Sample) -> Sample:
        n = sample["points"].shape[0]
        noise = np.clip(np.random.randn(n, 3) * self.sigma, -self.clip, self.clip)
        sample["points"] = sample["points"] + noise.astype(np.float32)
        return sample


# --------------------------------------------------------------------------- #
# Feature-space augmentations
# --------------------------------------------------------------------------- #
class HeightAppending:
    """Append each point's height (distance along gravity axis from the min) as
    an extra feature channel.

    The paper shows this helps classification a lot (+1.1% OA on ScanObjectNN)
    because the network learns the real size of objects.
    """

    def __init__(self, axis: int = 2):
        self.axis = axis

    def __call__(self, sample: Sample) -> Sample:
        pts = sample["points"]
        height = pts[:, self.axis:self.axis + 1] - pts[:, self.axis].min()
        feats = sample.get("features")
        if feats is None:
            sample["features"] = height.astype(np.float32)
        else:
            sample["features"] = np.concatenate([feats, height.astype(np.float32)], axis=1)
        return sample


class ColorDropping:
    """With probability p, zero out color channels for the whole cloud.

    The paper finds this is surprisingly powerful on S3DIS: +5.9% mIoU alone.
    It forces the network to rely on geometry instead of color shortcut cues.

    Assumes the first 3 feature channels are RGB.
    """

    def __init__(self, p: float = 0.2, color_channels: Tuple[int, int] = (0, 3)):
        self.p = p
        self.color_channels = color_channels

    def __call__(self, sample: Sample) -> Sample:
        feats = sample.get("features")
        if feats is None or feats.shape[1] < self.color_channels[1]:
            return sample
        if np.random.rand() < self.p:
            feats = feats.copy()
            feats[:, self.color_channels[0]:self.color_channels[1]] = 0.0
            sample["features"] = feats
        return sample


class ColorAutoContrast:
    """Auto-contrast color channels to [0, 1] per sample, probabilistically."""

    def __init__(self, p: float = 0.2, color_channels: Tuple[int, int] = (0, 3)):
        self.p = p
        self.color_channels = color_channels

    def __call__(self, sample: Sample) -> Sample:
        feats = sample.get("features")
        if feats is None or feats.shape[1] < self.color_channels[1]:
            return sample
        if np.random.rand() < self.p:
            feats = feats.copy()
            c0, c1 = self.color_channels
            c = feats[:, c0:c1]
            lo = c.min(axis=0, keepdims=True)
            hi = c.max(axis=0, keepdims=True)
            rng = np.maximum(hi - lo, 1e-6)
            feats[:, c0:c1] = (c - lo) / rng
            sample["features"] = feats
        return sample


# --------------------------------------------------------------------------- #
# Structural augmentations (on the point set itself)
# --------------------------------------------------------------------------- #
class PointResampling:
    """Randomly resample a fixed number of points from the cloud.

    This is the "point resampling" augmentation the paper cites from Point-BERT,
    and it gives PointNet++ +2.5% OA on ScanObjectNN.
    """

    def __init__(self, num_points: int):
        self.num_points = num_points

    def __call__(self, sample: Sample) -> Sample:
        pts = sample["points"]
        n = pts.shape[0]
        if n == self.num_points:
            return sample
        if n >= self.num_points:
            idx = np.random.choice(n, size=self.num_points, replace=False)
        else:
            idx = np.random.choice(n, size=self.num_points, replace=True)
        sample["points"] = pts[idx]
        if sample.get("features") is not None:
            sample["features"] = sample["features"][idx]
        if isinstance(sample.get("labels"), np.ndarray):
            sample["labels"] = sample["labels"][idx]
        return sample


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_augmentation(cfg: Optional[Dict[str, Any]]) -> Compose:
    """Build a Compose pipeline from a config block.

    Each key is the augmentation name; if the value is truthy, it's added.
    Values that are dicts are passed as kwargs; `True` uses defaults.

    Example config block:
        augmentation:
          random_rotation: true
          random_scaling: {scale_range: [0.9, 1.1]}
          height_appending: true
          color_dropping: {p: 0.2}
    """
    if not cfg:
        return Compose([])

    name_to_cls = {
        "random_rotation": RandomRotation,
        "random_scaling": RandomScaling,
        "random_translation": RandomTranslation,
        "random_jitter": RandomJitter,
        "height_appending": HeightAppending,
        "color_dropping": ColorDropping,
        "color_auto_contrast": ColorAutoContrast,
        "point_resampling": PointResampling,
    }
    # stable, logical order: resample → geometric → feature-space
    order = [
        "point_resampling",
        "random_rotation",
        "random_scaling",
        "random_translation",
        "random_jitter",
        "color_auto_contrast",
        "color_dropping",
        "height_appending",
    ]

    transforms: List[Callable[[Sample], Sample]] = []
    for name in order:
        if name not in cfg:
            continue
        spec = cfg[name]
        if spec is False or spec is None:
            continue
        cls = name_to_cls[name]
        kwargs = spec if isinstance(spec, dict) else {}
        transforms.append(cls(**kwargs))
    return Compose(transforms)
