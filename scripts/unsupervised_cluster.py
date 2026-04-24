#!/usr/bin/env python
"""Extract embeddings from an encoder, cluster them, and report metrics.

Usage:
    python scripts/unsupervised_cluster.py --config configs/cls_pointnext_s.yaml
    python scripts/unsupervised_cluster.py --config configs/cls_pointnext_s.yaml \
        --checkpoint runs/cls_pointnext_s/best.pt --method kmeans --n-clusters 4 \
        --project tsne --out-dir vis/unsup
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from unsupervised.embedding import extract_embeddings
from unsupervised.cluster import (
    cluster_embeddings, project_embeddings, evaluate_clustering,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unsupervised clustering of embeddings.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--method", default="kmeans", choices=["kmeans", "dbscan"])
    p.add_argument("--n-clusters", type=int, default=None,
                   help="kmeans only; defaults to model.num_classes if available.")
    p.add_argument("--eps", type=float, default=0.5, help="dbscan eps")
    p.add_argument("--min-samples", type=int, default=5, help="dbscan min_samples")
    p.add_argument("--project", default="pca", choices=["none", "pca", "tsne", "umap"])
    p.add_argument("--out-dir", default="vis/unsup")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config).to_dict()
    set_seed(int(cfg.get("seed", 42)))
    logger = get_logger("unsup")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Model
    model = build_model(dict(cfg["model"])).to(args.device).eval()
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, map_location=args.device)
        logger.info(f"Loaded checkpoint: {args.checkpoint}")
    else:
        logger.warning("No checkpoint — using untrained encoder.")

    # Data
    split_cfg = cfg.get(f"{args.split}_dataset", cfg["dataset"])
    loader, ds = build_dataloader(split_cfg, split=args.split, shuffle=False)

    # Embeddings
    logger.info("Extracting embeddings…")
    emb, labels = extract_embeddings(model, loader, device=args.device)
    logger.info(f"  embeddings: {emb.shape}, labels: {labels.shape}")

    # Cluster
    n_clusters = args.n_clusters
    if n_clusters is None and args.method == "kmeans":
        # Fall back to model.num_classes / num_parts / num_obj_classes
        mcfg = cfg["model"]
        n_clusters = (mcfg.get("num_classes")
                      or mcfg.get("num_obj_classes")
                      or mcfg.get("num_parts"))
        logger.info(f"  auto-selected n_clusters={n_clusters}")

    if args.method == "kmeans":
        preds = cluster_embeddings(emb, method="kmeans", n_clusters=n_clusters)
    else:
        preds = cluster_embeddings(emb, method="dbscan",
                                   eps=args.eps, min_samples=args.min_samples)

    # Metrics
    gt = labels if (labels != -1).any() and labels.ndim == 1 else None
    metrics = evaluate_clustering(preds, ground_truth=gt, embeddings=emb)
    logger.info(f"Clustering metrics: {metrics}")

    # Save metrics to disk
    with open(out_dir / "metrics.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    # Projection + plot
    if args.project != "none":
        logger.info(f"Projecting with {args.project}…")
        proj = project_embeddings(emb, method=args.project, n_components=2)
        # Two panels: colored by ground truth vs colored by cluster
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(proj[:, 0], proj[:, 1], c=labels, cmap="tab20", s=20)
        axes[0].set_title(f"Ground truth ({args.project})")
        axes[1].scatter(proj[:, 0], proj[:, 1], c=preds, cmap="tab20", s=20)
        axes[1].set_title(f"Clusters ({args.method})")
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        out_path = out_dir / f"projection_{args.project}_{args.method}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
