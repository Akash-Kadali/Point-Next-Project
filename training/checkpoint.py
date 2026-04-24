"""Checkpoint helpers.

A checkpoint stores:
    - model state dict
    - optimizer state dict
    - scheduler state dict (if any)
    - current epoch
    - best metric value so far
    - config (full dict) for reproducibility
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn


def save_checkpoint(path: Union[str, Path], model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Optional[Any] = None,
                    epoch: int = 0, best_metric: float = float("-inf"),
                    config: Optional[Dict[str, Any]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "epoch": int(epoch),
        "best_metric": float(best_metric),
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if config is not None:
        state["config"] = config
    torch.save(state, path)


def load_checkpoint(path: Union[str, Path], model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Optional[Any] = None,
                    map_location: str = "cpu") -> Dict[str, Any]:
    """Load a checkpoint. Returns the raw state dict so callers can pull metadata."""
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    return state
