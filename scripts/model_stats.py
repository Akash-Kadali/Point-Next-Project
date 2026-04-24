#!/usr/bin/env python
"""Print model parameters and inference throughput for a given config.

Usage:
    python scripts/model_stats.py --config configs/seg_pointnext_s.yaml
    python scripts/model_stats.py --config configs/seg_pointnext_s.yaml \
        --batch-size 8 --num-points 4096
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import load_config
from utils.model_stats import count_parameters, measure_throughput
from models.factory import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Show model stats.")
    p.add_argument("--config", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-points", type=int, default=1024)
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config).to_dict()

    model_cfg = dict(cfg["model"])
    model = build_model(model_cfg)

    in_channels = model_cfg["in_channels"]
    B = args.batch_size
    N = args.num_points

    # Build input signature — partseg needs an extra obj_label arg.
    # For stats we bypass the partseg head by testing the underlying encoder model.
    is_partseg = "partseg" in model_cfg["name"]

    params_trainable = count_parameters(model, trainable_only=True)
    params_total = count_parameters(model, trainable_only=False)

    print(f"Model:     {model_cfg['name']} "
          f"(preset={model_cfg.get('preset','-')})")
    print(f"Params:    {params_trainable:,} trainable / {params_total:,} total")
    print(f"Input:     B={B}, N={N}, in_channels={in_channels}, "
          f"device={args.device}")

    if is_partseg:
        # Build inputs manually for partseg
        model = model.to(args.device).eval()
        import time
        xyz = torch.randn(B, N, 3, device=args.device)
        feats = torch.randn(B, N, in_channels, device=args.device)
        obj = torch.zeros(B, dtype=torch.long, device=args.device)
        with torch.no_grad():
            for _ in range(args.warmup):
                model(xyz, feats, obj)
            if args.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(args.iters):
                model(xyz, feats, obj)
            if args.device == "cuda":
                torch.cuda.synchronize()
            total = time.time() - t0
        print(f"Time/batch: {1000*total/args.iters:.1f} ms")
        print(f"Throughput: {B*args.iters/total:.1f} instances/sec")
        return

    stats = measure_throughput(
        model,
        input_shapes={"xyz": (B, N, 3), "features": (B, N, in_channels)},
        device=args.device,
        num_iters=args.iters,
        num_warmup=args.warmup,
    )
    print(f"Time/batch: {stats['ms_per_batch']:.1f} ms")
    print(f"Throughput: {stats['instances_per_sec']:.1f} instances/sec")


if __name__ == "__main__":
    main()
