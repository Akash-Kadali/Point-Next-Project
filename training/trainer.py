"""Trainer — one class that handles classification, segmentation, and part
segmentation with task-specific metric handling.

Design:
    - `task` drives which forward signature and which metrics to use.
    - We accumulate (preds, labels) per epoch and compute metrics once at the
      end, which is both simpler and more accurate than running-means.
    - `monitor_metric` tells us which number to use as "best" for saving.

Quick usage:
    trainer = Trainer(model, train_loader, val_loader, cfg, task='cls')
    trainer.fit()
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.metrics import (
    classification_metrics,
    segmentation_metrics,
    part_segmentation_metrics,
    ConfusionMatrix,
)
from utils.logger import get_logger
from training.losses import build_criterion
from training.optim import build_optimizer, build_scheduler
from training.checkpoint import save_checkpoint, load_checkpoint


class Trainer:
    def __init__(self, model: nn.Module,
                 train_loader: DataLoader, val_loader: DataLoader,
                 cfg: Dict[str, Any], task: str = "cls",
                 device: Optional[str] = None,
                 output_dir: str = "runs/default"):
        """
        Args:
            model: the network.
            train_loader / val_loader: loaders yielding dict batches with at
                least 'xyz', 'features', and 'label' (+ 'obj_label' for partseg).
            cfg: the full config dict (used for serialization + hyperparams).
            task: 'cls' | 'seg' | 'partseg'.
            device: 'cpu' | 'cuda' | None (auto-detect).
            output_dir: directory for checkpoints and logs.
        """
        assert task in ("cls", "seg", "partseg")
        self.task = task
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        train_cfg = cfg.get("training", {})
        self.num_epochs = int(train_cfg.get("num_epochs", 10))

        self.criterion = build_criterion(cfg.get("criterion", {"name": "cross_entropy"}))
        self.optimizer = build_optimizer(
            self.model.parameters(),
            cfg.get("optimizer", {"name": "adam", "lr": 1e-3}),
        )
        self.scheduler = build_scheduler(
            self.optimizer, cfg.get("scheduler"),
            num_epochs=self.num_epochs,
        )

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("trainer", log_file=str(self.output_dir / "train.log"))

        # Which metric to monitor when saving the best model
        self.monitor_metric = train_cfg.get("monitor_metric", self._default_monitor())
        self.monitor_mode = train_cfg.get("monitor_mode", "max")  # 'max' or 'min'
        self.best_metric = -float("inf") if self.monitor_mode == "max" else float("inf")
        self.start_epoch = 0

        # Part-seg specific book-keeping
        self.parts_per_obj: Optional[Dict[int, List[int]]] = \
            train_cfg.get("parts_per_obj")

        # Optional resume
        resume_path = train_cfg.get("resume")
        if resume_path:
            self._resume(resume_path)

    # --------------------------------------------------------------------- #
    # Defaults per task
    # --------------------------------------------------------------------- #
    def _default_monitor(self) -> str:
        return {"cls": "overall_accuracy",
                "seg": "mean_iou",
                "partseg": "instance_miou"}[self.task]

    # --------------------------------------------------------------------- #
    # Forward helpers (task-specific)
    # --------------------------------------------------------------------- #
    def _forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        xyz = batch["xyz"].to(self.device)
        feats = batch["features"].to(self.device)
        if self.task == "partseg":
            obj = batch["obj_label"].to(self.device)
            return self.model(xyz, feats, obj)
        return self.model(xyz, feats)

    def _loss(self, logits: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        labels = batch["label"].to(self.device)
        if self.task == "cls":
            # logits [B, C], labels [B]
            return self.criterion(logits, labels)
        # seg / partseg: logits [B, N, C], labels [B, N]
        # Flatten to [B*N, C] and [B*N]
        B, N, C = logits.shape
        return self.criterion(logits.reshape(B * N, C), labels.reshape(B * N))

    # --------------------------------------------------------------------- #
    # Epoch loops
    # --------------------------------------------------------------------- #
    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in self.train_loader:
            logits = self._forward(batch)
            loss = self._loss(logits, batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        if self.scheduler is not None:
            self.scheduler.step()

        avg_loss = total_loss / max(n_batches, 1)
        lr = self.optimizer.param_groups[0]["lr"]
        dt = time.time() - t0
        self.logger.info(
            f"epoch {epoch:3d} | train loss {avg_loss:.4f} | lr {lr:.5f} | {dt:.1f}s"
        )
        return {"loss": avg_loss, "lr": lr, "epoch_time_s": dt}

    @torch.no_grad()
    def validate_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        # accumulate predictions for task-specific metrics
        all_preds: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        all_obj_labels: List[int] = []

        for batch in self.val_loader:
            logits = self._forward(batch)
            loss = self._loss(logits, batch)
            total_loss += float(loss.item())
            n_batches += 1

            preds = logits.argmax(dim=-1).cpu().numpy()  # [B] or [B, N]
            labels = batch["label"].numpy()
            all_preds.append(preds)
            all_labels.append(labels)
            if self.task == "partseg":
                all_obj_labels.append(batch["obj_label"].numpy())

        metrics = self._compute_metrics(all_preds, all_labels, all_obj_labels)
        metrics["val_loss"] = total_loss / max(n_batches, 1)

        monitor = metrics.get(self.monitor_metric, float("nan"))
        pretty = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                            if isinstance(v, (int, float)) and not np.isnan(v))
        self.logger.info(f"epoch {epoch:3d} | val   {pretty}")
        return metrics

    def _compute_metrics(self, preds: List[np.ndarray], labels: List[np.ndarray],
                         obj_labels: List[np.ndarray]) -> Dict[str, float]:
        if self.task == "cls":
            num_classes = int(max(np.concatenate(labels).max() + 1,
                                  np.concatenate(preds).max() + 1,
                                  2))
            # Prefer explicit num_classes from config
            num_classes = self.cfg.get("model", {}).get("num_classes", num_classes)
            return classification_metrics(
                np.concatenate(preds), np.concatenate(labels), num_classes
            )
        if self.task == "seg":
            num_classes = self.cfg.get("model", {}).get("num_classes", 3)
            return segmentation_metrics(
                np.concatenate([p.reshape(-1) for p in preds]),
                np.concatenate([l.reshape(-1) for l in labels]),
                num_classes=num_classes,
            )
        # partseg
        flat_preds = [p for batch in preds for p in batch]
        flat_labels = [l for batch in labels for l in batch]
        flat_obj = [int(o) for batch in obj_labels for o in batch]
        parts_per_obj = self.parts_per_obj
        if parts_per_obj is None:
            # Default: all parts available to all objects (fallback for toy)
            num_parts = self.cfg.get("model", {}).get("num_parts", 2)
            num_obj_cls = self.cfg.get("model", {}).get("num_obj_classes", 4)
            parts_per_obj = {c: list(range(num_parts)) for c in range(num_obj_cls)}
        return part_segmentation_metrics(
            flat_preds, flat_labels, flat_obj, parts_per_obj
        )

    # --------------------------------------------------------------------- #
    # Top-level loop
    # --------------------------------------------------------------------- #
    def fit(self) -> Dict[str, float]:
        """Run the full training loop. Returns the best validation metrics dict."""
        best_val_metrics: Dict[str, float] = {}

        for epoch in range(self.start_epoch + 1, self.num_epochs + 1):
            self.train_one_epoch(epoch)
            val_metrics = self.validate_one_epoch(epoch)

            monitor = val_metrics.get(self.monitor_metric, float("nan"))
            improved = self._is_better(monitor)
            if improved:
                self.best_metric = monitor
                best_val_metrics = dict(val_metrics)
                save_checkpoint(
                    self.output_dir / "best.pt", self.model,
                    self.optimizer, self.scheduler,
                    epoch=epoch, best_metric=self.best_metric,
                    config=self.cfg,
                )
                self.logger.info(f"  ↑ new best {self.monitor_metric}={monitor:.4f}")

            # always save "last" checkpoint for resume
            save_checkpoint(
                self.output_dir / "last.pt", self.model,
                self.optimizer, self.scheduler,
                epoch=epoch, best_metric=self.best_metric,
                config=self.cfg,
            )

        self.logger.info(f"Training complete. Best {self.monitor_metric}={self.best_metric:.4f}")
        return best_val_metrics

    def _is_better(self, value: float) -> bool:
        if np.isnan(value):
            return False
        if self.monitor_mode == "max":
            return value > self.best_metric
        return value < self.best_metric

    def _resume(self, path: str) -> None:
        self.logger.info(f"Resuming from {path}")
        state = load_checkpoint(path, self.model, self.optimizer, self.scheduler,
                                map_location=self.device)
        self.start_epoch = int(state.get("epoch", 0))
        self.best_metric = float(state.get("best_metric", self.best_metric))
