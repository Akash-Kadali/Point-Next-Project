"""Models for point cloud learning.

The module is organized so PointNet++ and PointNeXt share almost everything:
- pointnet_ops.py: low-level ops (FPS, ball query, kNN, index_points)
- blocks.py: SetAbstraction, FeaturePropagation, InvResMLP
- heads.py: classification / segmentation heads
- pointnet2.py: the baseline PointNet++ (SA + FP, no InvResMLP)
- pointnext.py: PointNeXt — stem MLP, residual SA, InvResMLP, symmetric decoder

All models use channels-last [B, N, C] tensors externally but internally
transpose to [B, C, N] for Conv1d friendliness.
"""
from .pointnet2 import PointNet2ClsModel, PointNet2SegModel, PointNet2PartSegModel
from .pointnext import (
    PointNeXtClsModel,
    PointNeXtSegModel,
    PointNeXtPartSegModel,
    POINTNEXT_PRESETS,
)
from .factory import build_model

__all__ = [
    "PointNet2ClsModel",
    "PointNet2SegModel",
    "PointNet2PartSegModel",
    "PointNeXtClsModel",
    "PointNeXtSegModel",
    "PointNeXtPartSegModel",
    "POINTNEXT_PRESETS",
    "build_model",
]
