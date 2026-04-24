"""Point cloud dataset classes and preprocessing utilities.

Three task styles are supported, each with its own dataset class:
- classification (one label per object)
- semantic segmentation (one label per point, scene-level)
- part segmentation (one label per point, object-level, with object class)

All default to generating toy synthetic data so the project runs end-to-end
without external downloads. To plug in real datasets (S3DIS, ScanObjectNN,
ShapeNetPart), subclass the base class and override `load_sample`.
"""
from .preprocessing import (
    normalize_points,
    random_sample,
    farthest_point_sample,
)
from .augmentation import (
    Compose,
    RandomRotation,
    RandomScaling,
    RandomTranslation,
    RandomJitter,
    HeightAppending,
    ColorDropping,
    ColorAutoContrast,
    PointResampling,
    build_augmentation,
)
from .toy import (
    ToyClassificationDataset,
    ToySegmentationDataset,
    ToyPartSegmentationDataset,
)
from .loader import build_dataloader

__all__ = [
    "normalize_points",
    "random_sample",
    "farthest_point_sample",
    "Compose",
    "RandomRotation",
    "RandomScaling",
    "RandomTranslation",
    "RandomJitter",
    "HeightAppending",
    "ColorDropping",
    "ColorAutoContrast",
    "PointResampling",
    "build_augmentation",
    "ToyClassificationDataset",
    "ToySegmentationDataset",
    "ToyPartSegmentationDataset",
    "build_dataloader",
]
