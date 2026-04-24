"""Extract per-object (or per-scene-patch) embeddings from the encoder.

For classification models, we use the built-in `extract_embedding` method
(global max-pool of the last encoder feature map). For segmentation models
we do the same trick manually: run the encoder, max-pool the deepest feature
map to get one vector per scene.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_embeddings(model: nn.Module, loader: DataLoader,
                       device: str = "cpu",
                       label_key: str = "label") -> Tuple[np.ndarray, np.ndarray]:
    """Run the model over a loader and return (embeddings, labels).

    The embedding is the global max-pooled feature from the deepest encoder
    stage. Works for both classification models and segmentation models — for
    seg, we manually walk the encoder because the forward path ends at a
    per-point head.

    Args:
        model: a built model (cls or seg PointNeXt or PointNet++).
        loader: any loader yielding 'xyz', 'features', and a label key.
        device: 'cpu' or 'cuda'.
        label_key: name of the key in each batch to treat as the ground-truth
            label (used for downstream evaluation only).

    Returns:
        embeddings: [M, C] numpy array (one row per sample).
        labels:     [M] numpy array. For per-point datasets we take the
            majority-vote per sample to get a single label.
    """
    model = model.to(device).eval()
    feats_out: List[np.ndarray] = []
    labels_out: List[np.ndarray] = []

    for batch in loader:
        xyz = batch["xyz"].to(device)
        features = batch["features"].to(device)

        emb = _encoder_forward(model, xyz, features)  # [B, C]
        feats_out.append(emb.cpu().numpy())

        lbl = batch[label_key].cpu().numpy()
        if lbl.ndim > 1:
            # per-point labels — use majority class per sample
            lbl = np.array([np.bincount(row[row >= 0]).argmax()
                            if (row >= 0).any() else -1
                            for row in lbl])
        labels_out.append(lbl)

    return np.concatenate(feats_out, axis=0), np.concatenate(labels_out, axis=0)


def _encoder_forward(model: nn.Module, xyz: torch.Tensor,
                     features: torch.Tensor) -> torch.Tensor:
    """Get a global embedding from whichever model architecture we got.

    We handle 4 cases:
        - PointNeXtClsModel: has `extract_embedding` method.
        - PointNeXtSegModel: walk the stem + encoder, then global-pool.
        - PointNet2ClsModel: run encoder + global_mlp then global-pool.
        - PointNet2SegModel: walk the encoder, then global-pool.
    """
    # PointNeXt classification (fast path)
    if hasattr(model, "extract_embedding") and callable(model.extract_embedding):
        return model.extract_embedding(xyz, features)

    # PointNeXt segmentation — has stem + encoder (ModuleList of PointNeXtStage)
    if hasattr(model, "stem") and hasattr(model, "encoder"):
        x = features.transpose(1, 2)
        x = model.stem(x)
        f = x.transpose(1, 2)
        cur_xyz, cur_feats = xyz, f
        for stage in model.encoder:
            cur_xyz, cur_feats = stage(cur_xyz, cur_feats)
        return cur_feats.max(dim=1)[0]

    # PointNeXt partseg wraps a seg model
    if hasattr(model, "backbone") and hasattr(model.backbone, "stem"):
        return _encoder_forward(model.backbone, xyz, features)

    # PointNet++ classification — run encoder + global_mlp
    if hasattr(model, "sa1") and hasattr(model, "sa2") and hasattr(model, "global_mlp"):
        xyz1, f1 = model.sa1(xyz, features)
        xyz2, f2 = model.sa2(xyz1, f1)
        x = f2.transpose(1, 2)
        x = model.global_mlp(x)
        return x.max(dim=-1)[0]  # global pool over points → [B, C]

    # PointNet++ segmentation — walk encoder, global-pool the deepest feats
    if hasattr(model, "encoder"):
        # Handle the None-features case the same way PointNet2SegModel.forward does
        if features is None:
            features = xyz
        cur_xyz, cur_feats = xyz, features
        for sa in model.encoder:
            cur_xyz, cur_feats = sa(cur_xyz, cur_feats)
        return cur_feats.max(dim=1)[0]

    # PointNet2 partseg
    if hasattr(model, "backbone") and hasattr(model.backbone, "encoder"):
        return _encoder_forward(model.backbone, xyz, features)

    raise RuntimeError(
        f"Don't know how to extract embeddings from {type(model).__name__}. "
        f"Add a case to _encoder_forward."
    )
