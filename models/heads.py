"""Task-specific heads.

Each head is a thin module that takes encoder (or encoder+decoder) outputs
and produces logits for the target task.
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Global-pool → MLP → logits. Used for object classification."""

    def __init__(self, in_channels: int, num_classes: int,
                 hidden: int = 256, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Args: feats [B, N, C] → global max pool → logits [B, num_classes]."""
        x, _ = feats.max(dim=1)  # [B, C]
        return self.net(x)


class SegmentationHead(nn.Module):
    """1x1 Conv → logits per point. Used for semantic + part segmentation."""

    def __init__(self, in_channels: int, num_classes: int,
                 hidden: int = 128, dropout: float = 0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, num_classes, 1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Args: feats [B, N, C] → logits [B, N, num_classes]."""
        x = feats.transpose(1, 2)      # [B, C, N]
        x = self.conv(x)               # [B, num_classes, N]
        return x.transpose(1, 2)       # [B, N, num_classes]


class PartSegmentationHead(nn.Module):
    """Same as segmentation head but optionally concatenates a one-hot
    object-class label to the per-point features (ShapeNetPart convention).
    """

    def __init__(self, in_channels: int, num_parts: int,
                 num_obj_classes: Optional[int] = None,
                 hidden: int = 128, dropout: float = 0.5):
        super().__init__()
        self.num_obj_classes = num_obj_classes
        extra = num_obj_classes if num_obj_classes else 0
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels + extra, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, num_parts, 1),
        )

    def forward(self, feats: torch.Tensor,
                obj_label: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Args:
            feats: [B, N, C]
            obj_label: [B] long tensor of object classes (optional)
        Returns: [B, N, num_parts]
        """
        x = feats.transpose(1, 2)  # [B, C, N]
        if self.num_obj_classes and obj_label is not None:
            N = x.shape[-1]
            one_hot = torch.zeros(
                x.shape[0], self.num_obj_classes, N, device=x.device
            )
            one_hot.scatter_(1, obj_label.view(-1, 1, 1).expand(-1, 1, N), 1.0)
            x = torch.cat([x, one_hot], dim=1)
        x = self.conv(x)
        return x.transpose(1, 2)
