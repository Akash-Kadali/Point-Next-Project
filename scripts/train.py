#!/usr/bin/env python
"""Train a model from a YAML config.

Usage:
    python scripts/train.py --config configs/cls_pointnext_s.yaml
    python scripts/train.py --config configs/seg_pointnext_s.yaml --epochs 5
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Make project root importable when running from anywhere.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import load_config
from utils.seed import set_seed
from utils.logger import get_logger
from datasets.loader import build_dataloader
from models.factory import build_model
from training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a PointNet++/PointNeXt model.")
    p.add_argument("--config", required=True, help="Path to YAML config.")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override num_epochs in config.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override batch_size in dataset configs.")
    p.add_argument("--device", default=None, help="cpu | cuda | (auto)")
    p.add_argument("--resume", default=None, help="Path to a checkpoint to resume.")
    p.add_argument("--output-dir", default=None,
                   help="Override output_dir in config.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config).to_dict()

    # Apply CLI overrides before anything else consumes the config.
    if args.epochs is not None:
        cfg.setdefault("training", {})["num_epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["dataset"]["batch_size"] = args.batch_size
        cfg.get("val_dataset", cfg["dataset"])["batch_size"] = args.batch_size
    if args.resume is not None:
        cfg.setdefault("training", {})["resume"] = args.resume
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir

    set_seed(int(cfg.get("seed", 42)))
    logger = get_logger("train")
    logger.info(f"Loaded config: {args.config}")
    logger.info(f"Task: {cfg.get('task')}  |  Model: {cfg['model']['name']}")

    # Build data
    train_loader, train_ds = build_dataloader(cfg["dataset"], split="train")
    val_cfg = cfg.get("val_dataset", cfg["dataset"])
    val_loader, val_ds = build_dataloader(val_cfg, split="val")
    logger.info(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    # Build model
    model = build_model(dict(cfg["model"]))
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params (trainable): {num_params:,}")

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        task=cfg.get("task", "cls"),
        device=args.device,
        output_dir=cfg.get("output_dir", "runs/default"),
    )
    best = trainer.fit()
    logger.info(f"Best metrics: {best}")


if __name__ == "__main__":
    main()
