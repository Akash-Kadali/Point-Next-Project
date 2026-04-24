#!/usr/bin/env python
"""Visualize predictions from a trained checkpoint.

Runs one val batch through the model and writes per-sample comparison PNGs
(pred vs ground truth) into the output directory. Also writes an augmentation
before/after comparison if possible.

Usage:
    python scripts/demo_visualize.py --config configs/seg_pointnext_s.yaml \
        --checkpoint runs/seg_pointnext_s/best.pt --out-dir vis/seg
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless safe
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
from datasets.augmentation import build_augmentation
from models.factory import build_model
from training.checkpoint import load_checkpoint
from visualization.plot import (
    plot_points, plot_points_labeled, plot_pred_vs_gt, plot_before_after,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Save prediction visualizations.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None,
                   help="Optional — use untrained model if omitted.")
    p.add_argument("--out-dir", default="vis")
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config).to_dict()
    set_seed(int(cfg.get("seed", 42)))
    logger = get_logger("visualize")

    task = cfg.get("task", "cls")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build and load model ----
    model = build_model(dict(cfg["model"])).to(args.device).eval()
    if args.checkpoint:
        state = load_checkpoint(args.checkpoint, model, map_location=args.device)
        logger.info(f"Loaded checkpoint epoch {state.get('epoch')}")
    else:
        logger.warning("No checkpoint given — predictions will be random.")

    # ---- One val batch for pred/gt comparison ----
    val_cfg = cfg.get("val_dataset", cfg["dataset"])
    loader, ds = build_dataloader(val_cfg, split="val",
                                  batch_size=args.num_samples, shuffle=False)
    batch = next(iter(loader))
    xyz = batch["xyz"].to(args.device)
    feats = batch["features"].to(args.device)
    if task == "partseg":
        obj = batch["obj_label"].to(args.device)
        logits = model(xyz, feats, obj)
    else:
        logits = model(xyz, feats)
    preds = logits.argmax(dim=-1).cpu().numpy()
    labels = batch["label"].numpy()
    xyz_np = batch["xyz"].numpy()

    # ---- Per-sample visualization ----
    for i in range(min(args.num_samples, xyz_np.shape[0])):
        pts = xyz_np[i]
        if task == "cls":
            # Single label per object — color-code by (pred, gt) as a title.
            plot_points(pts, title=f"pred={int(preds[i])} gt={int(labels[i])}",
                        save_path=str(out_dir / f"cls_sample_{i}.png"))
        else:
            # Per-point labels
            plot_pred_vs_gt(
                pts, preds[i], labels[i],
                num_classes=cfg["model"].get("num_classes",
                                              cfg["model"].get("num_parts", 2)),
                save_path=str(out_dir / f"{task}_sample_{i}.png"),
            )
        logger.info(f"Wrote {task}_sample_{i}.png")

    # ---- Augmentation before/after demo (using the train config) ----
    if "augmentation" in cfg.get("dataset", {}):
        raw_sample = ds.load_sample(0)
        pts_before = raw_sample["points"].copy()
        aug = build_augmentation(cfg["dataset"]["augmentation"])
        # don't carry height_appending etc. into the raw comparison
        aug_sample = aug(dict(raw_sample))
        pts_after = aug_sample["points"]
        plot_before_after(
            pts_before, pts_after,
            title_before="raw", title_after="augmented",
            save_path=str(out_dir / "augmentation_before_after.png"),
        )
        logger.info("Wrote augmentation_before_after.png")

    logger.info(f"All visualizations saved under: {out_dir}")


if __name__ == "__main__":
    main()
