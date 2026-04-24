"""Core building blocks for PointNet++ and PointNeXt.

Three blocks live here:

1. SetAbstractionBlock — the classical PointNet++ block.
   Subsample centers → group neighbors → shared MLPs → max-pool.
   Supports a `normalize_dp` flag (Eq. 2 in the paper) and an optional
   residual connection (used by PointNeXt when stage has only one SA block).

2. FeaturePropagationBlock — the U-Net decoder block.
   Interpolate features from the coarser level to the finer level using
   inverse-distance weighting of the 3 nearest neighbors, concat skip features,
   and run an MLP.

3. InvResMLPBlock — the PointNeXt-specific block.
   Adds: residual connection, separable MLPs (one layer on neighborhood feats
   before reduction + two pointwise layers after), and inverted-bottleneck
   channel expansion (×expansion in the middle). Appended after the main SA
   block in each stage.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnet_ops import (
    farthest_point_sample,
    ball_query,
    index_points,
    SharedMLP,
    SharedMLP2d,
    square_distance,
)


# --------------------------------------------------------------------------- #
# Set Abstraction
# --------------------------------------------------------------------------- #
class SetAbstractionBlock(nn.Module):
    """PointNet++ set abstraction with optional PointNeXt upgrades.

    Args:
        in_channels: input feature dim (excludes xyz).
        mlp_channels: list of channel sizes for the shared MLP stack.
        npoint: number of centers after FPS (None = keep all, i.e. no subsample).
        radius: ball query radius.
        nsample: number of neighbors per ball query.
        use_xyz: whether to concat relative coords with features before MLP.
        normalize_dp: divide relative coords by `radius` (PointNeXt, Eq. 2).
        residual: add a residual from the input subsampled features to the output
            (PointNeXt uses this when stage has only one SA block).
    """

    def __init__(self, in_channels: int, mlp_channels: List[int],
                 npoint: Optional[int], radius: float, nsample: int,
                 use_xyz: bool = True, normalize_dp: bool = False,
                 residual: bool = False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.use_xyz = use_xyz
        self.normalize_dp = normalize_dp
        self.residual = residual

        in_mlp = in_channels + (3 if use_xyz else 0)
        self.mlp = SharedMLP2d([in_mlp] + list(mlp_channels))
        self.out_channels = mlp_channels[-1]

        # optional residual: project in_channels -> out_channels
        if self.residual and in_channels != self.out_channels:
            self.residual_proj = nn.Conv1d(in_channels, self.out_channels, 1)
        else:
            self.residual_proj = None

    def forward(self, xyz: torch.Tensor, feats: Optional[torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
            xyz:   [B, N, 3]
            feats: [B, N, C] or None
        Returns:
            new_xyz:   [B, S, 3] subsampled centers
            new_feats: [B, S, C_out]
        """
        B, N, _ = xyz.shape

        # 1. choose centers
        if self.npoint is None or self.npoint >= N:
            new_xyz = xyz
            center_idx = None
        else:
            center_idx = farthest_point_sample(xyz, self.npoint)   # [B, S]
            new_xyz = index_points(xyz, center_idx)                # [B, S, 3]

        # 2. group neighbors
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)  # [B, S, K]
        grouped_xyz = index_points(xyz, idx)                        # [B, S, K, 3]
        # relative coordinates, optionally normalized by radius
        grouped_dp = grouped_xyz - new_xyz.unsqueeze(2)
        if self.normalize_dp:
            grouped_dp = grouped_dp / self.radius

        if feats is not None:
            grouped_feats = index_points(feats, idx)                # [B, S, K, C]
            if self.use_xyz:
                grouped = torch.cat([grouped_dp, grouped_feats], dim=-1)
            else:
                grouped = grouped_feats
        else:
            grouped = grouped_dp  # use coords only

        # 3. shared MLP + max-pool
        # -> [B, C_in+3, K, S]  (C_in+3 x K x S, with B)
        x = grouped.permute(0, 3, 2, 1).contiguous()
        x = self.mlp(x)                                             # [B, C_out, K, S]
        x = x.max(dim=2)[0]                                         # [B, C_out, S]

        # 4. optional residual connection
        if self.residual:
            if center_idx is not None and feats is not None:
                center_feats = index_points(feats, center_idx)      # [B, S, C_in]
                center_feats = center_feats.transpose(1, 2)         # [B, C_in, S]
            elif feats is not None:
                center_feats = feats.transpose(1, 2)
            else:
                center_feats = None
            if center_feats is not None:
                if self.residual_proj is not None:
                    center_feats = self.residual_proj(center_feats)
                if center_feats.shape == x.shape:
                    x = x + center_feats

        new_feats = x.transpose(1, 2).contiguous()                  # [B, S, C_out]
        return new_xyz, new_feats


# --------------------------------------------------------------------------- #
# Feature Propagation
# --------------------------------------------------------------------------- #
class FeaturePropagationBlock(nn.Module):
    """U-Net decoder block: interpolate coarser features to finer level, then MLP.

    Interpolation: 3-NN inverse-distance weighted average.
    """

    def __init__(self, in_channels: int, mlp_channels: List[int]):
        super().__init__()
        self.mlp = SharedMLP([in_channels] + list(mlp_channels))
        self.out_channels = mlp_channels[-1]

    def forward(self, xyz_fine: torch.Tensor, xyz_coarse: torch.Tensor,
                feats_fine: Optional[torch.Tensor], feats_coarse: torch.Tensor
                ) -> torch.Tensor:
        """Args:
            xyz_fine:   [B, N, 3] the target (higher-res) points
            xyz_coarse: [B, M, 3] the source (lower-res) points (M < N)
            feats_fine: [B, N, C1] skip features from the encoder at this level (or None)
            feats_coarse: [B, M, C2] features at the coarser level
        Returns:
            [B, N, C_out]
        """
        B, N, _ = xyz_fine.shape
        M = xyz_coarse.shape[1]

        if M == 1:
            # edge case: 1 coarse point → broadcast
            interp = feats_coarse.expand(-1, N, -1)
        else:
            dists = square_distance(xyz_fine, xyz_coarse)           # [B, N, M]
            dists, idx = dists.topk(3, dim=-1, largest=False)       # [B, N, 3]
            dists = dists.clamp(min=1e-10)
            weights = 1.0 / dists                                   # [B, N, 3]
            weights = weights / weights.sum(dim=-1, keepdim=True)
            # gather coarse features at the 3 nearest neighbors
            gathered = index_points(feats_coarse, idx)              # [B, N, 3, C2]
            interp = (weights.unsqueeze(-1) * gathered).sum(dim=2)  # [B, N, C2]

        if feats_fine is not None:
            cat = torch.cat([interp, feats_fine], dim=-1)
        else:
            cat = interp

        x = cat.transpose(1, 2).contiguous()                        # [B, C, N]
        x = self.mlp(x)
        return x.transpose(1, 2).contiguous()                       # [B, N, C_out]


# --------------------------------------------------------------------------- #
# InvResMLP (PointNeXt's improved block)
# --------------------------------------------------------------------------- #
class InvResMLPBlock(nn.Module):
    """Inverted-Residual MLP block (PointNeXt, Sec. 3.2.2).

    Structure (input has C channels):
        1. Neighborhood MLP (1 layer, C → C) on grouped features + relative xyz,
           followed by max-pool.  (the 'separable' step on neighborhoods)
        2. Pointwise MLP (2 layers: C → expansion*C → C), with BN + ReLU.
           (the 'inverted bottleneck' — expansion=4 by default.)
        3. Residual add from the input features.

    No subsampling: InvResMLP operates at the same resolution as its input stage.
    """

    def __init__(self, channels: int, radius: float, nsample: int,
                 expansion: int = 4, normalize_dp: bool = True):
        super().__init__()
        self.radius = radius
        self.nsample = nsample
        self.normalize_dp = normalize_dp

        # step 1: neighborhood MLP (grouped feats + relative xyz)
        self.neighbor_mlp = SharedMLP2d([channels + 3, channels])

        # step 2: inverted bottleneck pointwise
        hidden = channels * expansion
        self.pw1 = nn.Conv1d(channels, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.act1 = nn.ReLU(inplace=True)
        self.pw2 = nn.Conv1d(hidden, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act2 = nn.ReLU(inplace=True)

        self.out_channels = channels

    def forward(self, xyz: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """Args: xyz [B, N, 3], feats [B, N, C]. Returns: [B, N, C]."""
        B, N, C = feats.shape
        identity = feats

        # neighborhood step
        idx = ball_query(self.radius, self.nsample, xyz, xyz)       # [B, N, K]
        grouped_xyz = index_points(xyz, idx) - xyz.unsqueeze(2)     # [B, N, K, 3]
        if self.normalize_dp:
            grouped_xyz = grouped_xyz / self.radius
        grouped_feats = index_points(feats, idx)                    # [B, N, K, C]
        grouped = torch.cat([grouped_xyz, grouped_feats], dim=-1)   # [B, N, K, C+3]
        x = grouped.permute(0, 3, 2, 1).contiguous()                # [B, C+3, K, N]
        x = self.neighbor_mlp(x)                                    # [B, C, K, N]
        x = x.max(dim=2)[0]                                         # [B, C, N]

        # pointwise inverted bottleneck (input x is [B, C, N] here)
        x = self.act1(self.bn1(self.pw1(x)))
        x = self.bn2(self.pw2(x))                                   # [B, C, N]

        # residual + final activation (keep [B, C, N] throughout)
        identity_bcn = identity.transpose(1, 2).contiguous()        # [B, C, N]
        x = self.act2(x + identity_bcn)
        out = x.transpose(1, 2).contiguous()                        # [B, N, C]
        return out
