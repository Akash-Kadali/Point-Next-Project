"""Loss factory.

The paper's improved training uses CrossEntropy with label smoothing. PyTorch
≥1.10 has `label_smoothing` built into `nn.CrossEntropyLoss`, so we just
forward the kwarg.
"""
from __future__ import annotations
from typing import Any, Dict
import torch.nn as nn


def build_criterion(cfg: Dict[str, Any]) -> nn.Module:
    """Build a loss module.

    Config:
        criterion:
          name: cross_entropy
          label_smoothing: 0.2   # 0.0 = baseline, 0.2 = paper's setting
          ignore_index: -1
    """
    cfg = dict(cfg or {})
    name = cfg.pop("name", "cross_entropy").lower()
    if name in ("cross_entropy", "ce"):
        return nn.CrossEntropyLoss(
            label_smoothing=float(cfg.get("label_smoothing", 0.0)),
            ignore_index=int(cfg.get("ignore_index", -100)),
        )
    raise ValueError(f"Unknown criterion: {name}")
