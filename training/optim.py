"""Optimizer + scheduler factories.

Paper's baseline settings vs improved settings:
    Adam + StepLR  (baseline)  vs  AdamW + CosineAnnealingLR  (improved)

Configs are always dicts; we pop 'name' and pass the rest as kwargs.
"""
from __future__ import annotations
from typing import Any, Dict, Iterable, Optional
import torch
import torch.nn as nn


def build_optimizer(params: Iterable[nn.Parameter], cfg: Dict[str, Any]
                    ) -> torch.optim.Optimizer:
    cfg = dict(cfg)
    name = cfg.pop("name", "adam").lower()
    if name == "adam":
        return torch.optim.Adam(
            params,
            lr=cfg.get("lr", 1e-3),
            weight_decay=cfg.get("weight_decay", 1e-4),
            betas=tuple(cfg.get("betas", (0.9, 0.999))),
        )
    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=cfg.get("lr", 1e-3),
            weight_decay=cfg.get("weight_decay", 1e-4),
            betas=tuple(cfg.get("betas", (0.9, 0.999))),
        )
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=cfg.get("lr", 0.01),
            momentum=cfg.get("momentum", 0.9),
            weight_decay=cfg.get("weight_decay", 1e-4),
            nesterov=cfg.get("nesterov", True),
        )
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Optional[Dict[str, Any]],
                    num_epochs: Optional[int] = None):
    """Build a learning-rate scheduler. Returns None if cfg is missing."""
    if not cfg:
        return None
    cfg = dict(cfg)
    name = cfg.pop("name", "cosine").lower()
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.get("step_size", 20),
            gamma=cfg.get("gamma", 0.7),
        )
    if name == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.get("milestones", [30, 60]),
            gamma=cfg.get("gamma", 0.1),
        )
    if name == "cosine":
        T_max = cfg.get("T_max", num_epochs or 100)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=cfg.get("eta_min", 1e-6),
        )
    if name == "none":
        return None
    raise ValueError(f"Unknown scheduler: {name}")
