"""Model stats: parameter count + inference throughput.

We keep FLOPs optional (thop is a heavy dep and often fails on custom ops),
so throughput + param count is the default.
"""
from __future__ import annotations
import time
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Return total number of (trainable) parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def measure_throughput(
    model: nn.Module,
    input_shapes: Dict[str, Tuple[int, ...]],
    device: str = "cpu",
    num_iters: int = 20,
    num_warmup: int = 5,
) -> Dict[str, float]:
    """Measure inference throughput (instances/sec) for a given model.

    Args:
        model: the model. Must accept a dict of tensors as kwargs, or a single
            tensor if `input_shapes` has exactly one entry named 'x'.
        input_shapes: mapping from kwarg name to shape (including batch dim).
            e.g. {"xyz": (32, 1024, 3), "feats": (32, 1024, 3)}.
        device: 'cpu' or 'cuda'.
        num_iters: timed iterations.
        num_warmup: warmup iterations (not timed).

    Returns:
        dict with keys: batch_size, total_time_s, instances_per_sec, ms_per_batch.
    """
    model = model.to(device).eval()
    inputs = {k: torch.randn(*s, device=device) for k, s in input_shapes.items()}
    batch_size = next(iter(input_shapes.values()))[0]

    # warmup
    for _ in range(num_warmup):
        _ = model(**inputs)
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_iters):
        _ = model(**inputs)
    if device == "cuda":
        torch.cuda.synchronize()
    total = time.time() - start

    per_batch_ms = 1000.0 * total / num_iters
    ips = batch_size * num_iters / total
    return {
        "batch_size": batch_size,
        "total_time_s": total,
        "ms_per_batch": per_batch_ms,
        "instances_per_sec": ips,
    }


def model_stats_summary(model: nn.Module, input_shapes: Dict[str, Tuple[int, ...]],
                        device: str = "cpu") -> Dict[str, Any]:
    """One-shot summary: params + throughput."""
    stats = {"params_total": count_parameters(model, trainable_only=False),
             "params_trainable": count_parameters(model, trainable_only=True)}
    stats.update(measure_throughput(model, input_shapes, device=device))
    return stats
