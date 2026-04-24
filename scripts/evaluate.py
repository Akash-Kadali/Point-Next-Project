#!/usr/bin/env python
"""Evaluate a trained checkpoint on the val split defined in a config.

Usage:
    python scripts/evaluate.py --config configs/cls_pointnext_s.yaml \
                               --checkpoint runs/cls_pointnext_s/best.pt
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import load_config
from utils.seed import set_seed
from utils.logger import get_logger
from datasets.loader import build_dataloader
from models.factory import build_model
from training.checkpoint import load_checkpoint
from evaluation.metrics import (
    classification_metrics,
    segmentation_metrics,
    part_segmentation_metrics,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a checkpoint.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--output", default=None,
                   help="Optional JSON output path for metrics.")
    return p.parse_args()


@torch.no_grad()
def run_eval(model, loader, task: str, device: str, cfg: dict) -> dict:
    """Inference over the whole loader, then compute task-specific metrics."""
    model.eval().to(device)
    preds_all, labels_all, obj_all = [], [], []
    for batch in loader:
        xyz = batch["xyz"].to(device)
        feats = batch["features"].to(device)
        if task == "partseg":
            obj = batch["obj_label"].to(device)
            logits = model(xyz, feats, obj)
        else:
            logits = model(xyz, feats)
        preds = logits.argmax(dim=-1).cpu().numpy()
        preds_all.append(preds)
        labels_all.append(batch["label"].numpy())
        if task == "partseg":
            obj_all.append(batch["obj_label"].numpy())

    if task == "cls":
        nc = cfg["model"]["num_classes"]
        return classification_metrics(
            np.concatenate(preds_all), np.concatenate(labels_all), nc
        )
    if task == "seg":
        nc = cfg["model"]["num_classes"]
        return segmentation_metrics(
            np.concatenate([p.reshape(-1) for p in preds_all]),
            np.concatenate([l.reshape(-1) for l in labels_all]),
            num_classes=nc,
        )
    # partseg
    flat_preds = [p for batch in preds_all for p in batch]
    flat_labels = [l for batch in labels_all for l in batch]
    flat_obj = [int(o) for batch in obj_all for o in batch]
    parts_per_obj = cfg.get("training", {}).get("parts_per_obj")
    if parts_per_obj is None:
        num_parts = cfg["model"]["num_parts"]
        num_obj = cfg["model"]["num_obj_classes"]
        parts_per_obj = {c: list(range(num_parts)) for c in range(num_obj)}
    # YAML loads dict keys as ints by PyYAML only if the key was written as an
    # int; for safety, convert.
    parts_per_obj = {int(k): list(v) for k, v in parts_per_obj.items()}
    return part_segmentation_metrics(flat_preds, flat_labels, flat_obj, parts_per_obj)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config).to_dict()
    set_seed(int(cfg.get("seed", 42)))
    logger = get_logger("evaluate")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    task = cfg.get("task", "cls")

    val_cfg = cfg.get("val_dataset", cfg["dataset"])
    loader, _ = build_dataloader(val_cfg, split="val")

    model = build_model(dict(cfg["model"]))
    state = load_checkpoint(args.checkpoint, model, map_location=device)
    logger.info(f"Loaded checkpoint from epoch {state.get('epoch')} "
                f"(best_metric={state.get('best_metric', float('nan')):.4f})")

    metrics = run_eval(model, loader, task, device, cfg)
    logger.info("Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float) and not np.isnan(v):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({k: (v if not (isinstance(v, float) and np.isnan(v)) else None)
                       for k, v in metrics.items()}, f, indent=2)
        logger.info(f"Wrote metrics to {args.output}")


if __name__ == "__main__":
    main()
