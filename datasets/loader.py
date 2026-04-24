"""Build a DataLoader from a config block.

Dataset registry is tiny — toy datasets only by default. To add real datasets,
import your class and register it here.
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
from torch.utils.data import DataLoader, Dataset

from .augmentation import build_augmentation
from .toy import (
    ToyClassificationDataset,
    ToySegmentationDataset,
    ToyPartSegmentationDataset,
)


_DATASET_REGISTRY = {
    "toy_cls": ToyClassificationDataset,
    "toy_seg": ToySegmentationDataset,
    "toy_partseg": ToyPartSegmentationDataset,
}


def build_dataset(cfg: Dict[str, Any], split: str = "train") -> Dataset:
    """Build a single Dataset split from a config block.

    Expected cfg keys:
        name: one of the registered names
        num_samples, num_points: ints
        augmentation: dict — applied in train split by default.
        val_transform: dict (optional) — applied in val/test split. Typically
            used for deterministic feature-generation steps (like
            height_appending) that must exist in both splits so model input
            shapes match. If omitted, no transform is applied in val.
        kwargs: dict (extra args to the dataset class)
    """
    name = cfg["name"]
    if name not in _DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. "
                         f"Registered: {list(_DATASET_REGISTRY)}")
    cls = _DATASET_REGISTRY[name]

    transform = None
    if split == "train" and "augmentation" in cfg:
        transform = build_augmentation(cfg["augmentation"])
    elif split != "train" and "val_transform" in cfg:
        transform = build_augmentation(cfg["val_transform"])

    kwargs = dict(cfg.get("kwargs", {}) or {})
    return cls(
        split=split,
        num_samples=cfg.get("num_samples", 256),
        num_points=cfg.get("num_points", 1024),
        transform=transform,
        **kwargs,
    )


def build_dataloader(cfg: Dict[str, Any], split: str = "train",
                     batch_size: Optional[int] = None,
                     num_workers: Optional[int] = None,
                     shuffle: Optional[bool] = None) -> Tuple[DataLoader, Dataset]:
    """Build a DataLoader for a given split. Returns (loader, dataset)."""
    dataset = build_dataset(cfg, split=split)
    if batch_size is None:
        batch_size = cfg.get("batch_size", 16)
    if num_workers is None:
        num_workers = cfg.get("num_workers", 0)
    if shuffle is None:
        shuffle = (split == "train")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=(split == "train"),
        pin_memory=False,
    )
    return loader, dataset


def register_dataset(name: str, cls) -> None:
    """Register a new dataset class under a name (for real datasets)."""
    _DATASET_REGISTRY[name] = cls
