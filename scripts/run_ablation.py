#!/usr/bin/env python
"""Ablation runner.

Defines a base config and a list of named variants, each of which applies a
patch on top of the base. Trains each variant for a small number of epochs
and writes a CSV + markdown summary.

Variants follow the order of Tabs. 4 and 5 in the paper (additive study):
    PointNet++ baseline
    + improved training (label smoothing, AdamW, cosine)
    + data scaling (resampling / height appending)
    + normalize Δp
    + scale up (PointNeXt-S)
    + PointNeXt-B
    + PointNeXt-L

You can run the full sweep with:
    python scripts/run_ablation.py --task cls --out results/ablation_cls.csv
"""
from __future__ import annotations
import argparse
import copy
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


# --------------------------------------------------------------------------- #
# Variant definitions
# --------------------------------------------------------------------------- #
# Each variant is (name, patch). The patch is a nested dict that gets
# deep-merged onto the base config. We build up the variants incrementally
# (like the paper's additive study), so each later variant inherits earlier
# patches too.
#
# Keys:
#   model.* → changes the architecture (switches model class / preset / flags)
#   dataset.augmentation.* → toggles augmentations
#   criterion.* → label smoothing
#   optimizer.name / scheduler.name → optimization switches

CLS_VARIANTS: List[Tuple[str, Dict[str, Any]]] = [
    ("pointnet2_baseline", {}),

    ("+ point_resampling", {
        "dataset": {"augmentation": {"point_resampling": {"num_points": 1024}}},
    }),

    ("+ height_appending", {
        "dataset": {"augmentation": {"height_appending": True},
                    "val_transform": {"height_appending": True}},
        "val_dataset": {"val_transform": {"height_appending": True}},
        # toy_cls has no base features; height_appending produces 1-channel feats.
        "model": {"in_channels": 1},
    }),

    ("+ random_scaling", {
        "dataset": {"augmentation": {"random_scaling": {"scale_range": [0.9, 1.1]}}},
    }),

    ("+ label_smoothing", {
        "criterion": {"label_smoothing": 0.3},
    }),

    ("+ adamw", {
        "optimizer": {"name": "adamw", "lr": 0.002, "weight_decay": 0.05},
    }),

    ("+ cosine_decay", {
        "scheduler": {"name": "cosine", "eta_min": 1.0e-6},
    }),

    ("+ normalize_dp", {
        "model": {"normalize_dp": True},
    }),

    ("+ pointnext_s", {
        "model": {"name": "pointnext_cls", "preset": "s", "in_channels": 1,
                  "normalize_dp": True},
        # PointNeXt's stem runs *only* on features (not xyz), so we switch to
        # height-only (1 channel).  Both splits must apply height_appending.
        "dataset": {"augmentation": {"height_appending": True},
                    "val_transform": {"height_appending": True}},
        "val_dataset": {"val_transform": {"height_appending": True}},
    }),

    ("+ pointnext_b", {
        "model": {"name": "pointnext_cls", "preset": "b", "in_channels": 1,
                  "normalize_dp": True},
        "dataset": {"augmentation": {"height_appending": True},
                    "val_transform": {"height_appending": True}},
        "val_dataset": {"val_transform": {"height_appending": True}},
    }),
]


SEG_VARIANTS: List[Tuple[str, Dict[str, Any]]] = [
    ("pointnet2_baseline", {}),

    ("+ color_dropping", {
        "dataset": {"augmentation": {"color_dropping": {"p": 0.2}}},
    }),

    ("+ color_auto_contrast", {
        "dataset": {"augmentation": {"color_auto_contrast": {"p": 0.2}}},
    }),

    ("+ height_appending", {
        "dataset": {"augmentation": {"height_appending": True},
                    "val_transform": {"height_appending": True}},
        "val_dataset": {"val_transform": {"height_appending": True}},
        "model": {"in_channels": 4},
    }),

    ("+ label_smoothing", {
        "criterion": {"label_smoothing": 0.2},
    }),

    ("+ adamw", {
        "optimizer": {"name": "adamw", "lr": 0.01, "weight_decay": 1.0e-4},
    }),

    ("+ cosine_decay", {
        "scheduler": {"name": "cosine", "eta_min": 1.0e-6},
    }),

    ("+ normalize_dp", {
        "model": {"normalize_dp": True},
    }),

    ("+ pointnext_s", {
        "model": {"name": "pointnext_seg", "preset": "s",
                  "normalize_dp": True, "in_channels": 4},
    }),

    ("+ pointnext_b", {
        "model": {"name": "pointnext_seg", "preset": "b",
                  "normalize_dp": True, "in_channels": 4},
    }),
]


# --------------------------------------------------------------------------- #
# Config manipulation
# --------------------------------------------------------------------------- #
def deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep-merged copy of base <- patch."""
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_variants_cumulatively(base: Dict[str, Any],
                                variants: List[Tuple[str, Dict[str, Any]]]
                                ) -> List[Tuple[str, Dict[str, Any]]]:
    """Build the cumulative configs for an additive-study sweep."""
    results = []
    cur = copy.deepcopy(base)
    for name, patch in variants:
        cur = deep_update(cur, patch)
        results.append((name, copy.deepcopy(cur)))
    return results


# --------------------------------------------------------------------------- #
# Run one variant
# --------------------------------------------------------------------------- #
def run_variant(name: str, cfg: Dict[str, Any], task: str,
                epochs: int, device: str, out_root: Path) -> Dict[str, Any]:
    set_seed(int(cfg.get("seed", 42)))
    logger = get_logger("ablation")
    logger.info(f"\n=== Variant: {name} ===")
    logger.info(f"  model={cfg['model']['name']} preset={cfg['model'].get('preset','-')} "
                f"normalize_dp={cfg['model'].get('normalize_dp', False)}")
    logger.info(f"  opt={cfg['optimizer']['name']} sched={cfg.get('scheduler',{}).get('name','-')} "
                f"label_smoothing={cfg.get('criterion',{}).get('label_smoothing', 0.0)}")

    # Epoch override for quick sweeps
    cfg.setdefault("training", {})["num_epochs"] = epochs
    run_dir = out_root / name.replace(" ", "_").replace("+", "p")
    cfg["output_dir"] = str(run_dir)

    # Build components
    train_loader, train_ds = build_dataloader(cfg["dataset"], split="train")
    val_cfg = cfg.get("val_dataset", cfg["dataset"])
    val_loader, val_ds = build_dataloader(val_cfg, split="val")
    model = build_model(dict(cfg["model"]))
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        task=task,
        device=device,
        output_dir=cfg["output_dir"],
    )
    t0 = time.time()
    best = trainer.fit()
    dt = time.time() - t0

    return {
        "variant": name,
        "num_params": int(num_params),
        "epochs": int(epochs),
        "train_time_s": round(dt, 1),
        **{k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
           for k, v in best.items()},
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_markdown(rows: List[Dict[str, Any]], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w") as f:
        f.write(f"# {title}\n\n")
        f.write("| " + " | ".join(keys) + " |\n")
        f.write("|" + "|".join(["---"] * len(keys)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |\n")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Additive ablation sweep.")
    p.add_argument("--task", choices=["cls", "seg"], required=True,
                   help="Which task to ablate.")
    p.add_argument("--base-config", default=None,
                   help="Base config path. Defaults to the baseline config.")
    p.add_argument("--epochs", type=int, default=5,
                   help="Epochs per variant (small so the sweep is cheap).")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None,
                   help="CSV output path. Also writes .md next to it.")
    p.add_argument("--out-root", default="runs/ablation",
                   help="Root directory for per-variant run dirs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.base_config is None:
        args.base_config = str(ROOT / f"configs/{args.task}_pointnet2_baseline.yaml")
    base = load_config(args.base_config).to_dict()

    variants = CLS_VARIANTS if args.task == "cls" else SEG_VARIANTS
    cumulative = apply_variants_cumulatively(base, variants)

    out_root = Path(args.out_root) / args.task
    out_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for name, cfg in cumulative:
        row = run_variant(name, cfg, task=args.task, epochs=args.epochs,
                          device=args.device, out_root=out_root)
        rows.append(row)

    out_csv = Path(args.out) if args.out else out_root / "results.csv"
    out_md = out_csv.with_suffix(".md")
    write_csv(rows, out_csv)
    write_markdown(rows, out_md, title=f"Ablation results ({args.task})")
    print(f"\nWrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
