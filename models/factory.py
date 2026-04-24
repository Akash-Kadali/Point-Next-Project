"""Factory function to build models from config.

Config shape:
    model:
      name: pointnet2_cls | pointnet2_seg | pointnet2_partseg |
            pointnext_cls | pointnext_seg | pointnext_partseg
      # For PointNeXt, set preset OR explicit width/blocks:
      preset: s | b | l | xl
      # Common:
      in_channels: 3
      num_classes: 4
      normalize_dp: true / false
      ...  (any kwargs the model accepts)
"""
from __future__ import annotations
from typing import Any, Dict
import torch.nn as nn

from .pointnet2 import (
    PointNet2ClsModel,
    PointNet2SegModel,
    PointNet2PartSegModel,
)
from .pointnext import (
    PointNeXtClsModel,
    PointNeXtSegModel,
    PointNeXtPartSegModel,
    POINTNEXT_PRESETS,
)


_MODEL_REGISTRY = {
    "pointnet2_cls":     PointNet2ClsModel,
    "pointnet2_seg":     PointNet2SegModel,
    "pointnet2_partseg": PointNet2PartSegModel,
    "pointnext_cls":     PointNeXtClsModel,
    "pointnext_seg":     PointNeXtSegModel,
    "pointnext_partseg": PointNeXtPartSegModel,
}


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """Build a model from a model config block."""
    cfg = dict(cfg)
    name = cfg.pop("name")
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY)}")

    # Apply PointNeXt preset if asked
    preset = cfg.pop("preset", None)
    if preset is not None:
        if not name.startswith("pointnext"):
            raise ValueError(f"preset '{preset}' only valid for pointnext_* models.")
        preset_cfg = POINTNEXT_PRESETS[preset.lower()]
        cfg.setdefault("width", preset_cfg["width"])
        cfg.setdefault("blocks", tuple(preset_cfg["blocks"]))

    cls = _MODEL_REGISTRY[name]
    return cls(**cfg)
