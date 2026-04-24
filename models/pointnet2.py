"""PointNet++ baseline models.

This is the "before PointNeXt" network: 4 Set Abstraction blocks for
segmentation (2 for classification), no residual connections, no normalized
relative coords, no InvResMLP. It reflects the starting point of Section 3
in the paper.

All models expose the same forward signature:
    forward(xyz: [B, N, 3], features: [B, N, C_in]) -> logits
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import torch
import torch.nn as nn

from .blocks import SetAbstractionBlock, FeaturePropagationBlock
from .heads import ClassificationHead, SegmentationHead, PartSegmentationHead


# --------------------------------------------------------------------------- #
# Classification (2-stage encoder)
# --------------------------------------------------------------------------- #
class PointNet2ClsModel(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 40,
                 npoints: Tuple[int, int] = (512, 128),
                 radii: Tuple[float, float] = (0.2, 0.4),
                 nsamples: Tuple[int, int] = (32, 64),
                 mlps: Tuple[List[int], List[int]] = (
                     [64, 64, 128], [128, 128, 256]),
                 normalize_dp: bool = False,
                 head_hidden: int = 256, dropout: float = 0.5):
        super().__init__()
        self.sa1 = SetAbstractionBlock(
            in_channels, mlps[0], npoints[0], radii[0], nsamples[0],
            normalize_dp=normalize_dp)
        self.sa2 = SetAbstractionBlock(
            self.sa1.out_channels, mlps[1], npoints[1], radii[1], nsamples[1],
            normalize_dp=normalize_dp)
        # final "global" SA with npoint=1 is replaced by an MLP + global pool
        self.global_mlp = nn.Sequential(
            nn.Conv1d(self.sa2.out_channels, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )
        self.head = ClassificationHead(1024, num_classes, hidden=head_hidden,
                                       dropout=dropout)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        # Strip xyz-duplicated features for classification (features are usually just xyz here)
        if features is None:
            feats = None
        else:
            feats = features
        xyz1, f1 = self.sa1(xyz, feats)
        xyz2, f2 = self.sa2(xyz1, f1)
        # [B, S, C] -> [B, C, S]
        x = f2.transpose(1, 2)
        x = self.global_mlp(x)
        x = x.transpose(1, 2)  # [B, S, 1024] for pooling
        return self.head(x)


# --------------------------------------------------------------------------- #
# Semantic segmentation (4-stage U-Net)
# --------------------------------------------------------------------------- #
class PointNet2SegModel(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 13,
                 npoints: Tuple[int, ...] = (1024, 256, 64, 16),
                 radii: Tuple[float, ...] = (0.1, 0.2, 0.4, 0.8),
                 nsamples: Tuple[int, ...] = (32, 32, 32, 32),
                 enc_mlps: Tuple[List[int], ...] = (
                     [32, 32, 64], [64, 64, 128],
                     [128, 128, 256], [256, 256, 512]),
                 dec_mlps: Tuple[List[int], ...] = (
                     [256, 256], [256, 256],
                     [256, 128], [128, 128, 128]),
                 normalize_dp: bool = False,
                 head_hidden: int = 128, dropout: float = 0.5):
        super().__init__()
        assert len(npoints) == len(radii) == len(nsamples) == len(enc_mlps) == 4
        assert len(dec_mlps) == 4

        # Encoder
        self.encoder = nn.ModuleList()
        prev = in_channels
        for np_, r, ns, mlp in zip(npoints, radii, nsamples, enc_mlps):
            sa = SetAbstractionBlock(prev, mlp, np_, r, ns,
                                     normalize_dp=normalize_dp)
            self.encoder.append(sa)
            prev = sa.out_channels

        # Decoder — order: deepest first.
        # enc_out is [C_in, C0, C1, C2, C3] (5 values).
        # For stage i (i=3,2,1,0, deepest first):
        #     coarse channels come from enc_out[i+1] for i=3 (deepest encoder),
        #     or from the previous FP's output for i<3.
        #     skip channels come from enc_out[i] (which is the encoder output
        #     *before* the sa block at this level — i.e., the input resolution
        #     features for the finest decoder step).
        # Outputs are given by dec_mlps[k][-1] where k=0..3 (deepest first).
        enc_out = [in_channels] + [sa.out_channels for sa in self.encoder]
        # enc_out[0]=C_in, enc_out[1..4] = stage outputs
        self.decoder = nn.ModuleList()
        # Skip channels, deepest-first: enc_out[3], enc_out[2], enc_out[1], enc_out[0]
        skip_channels = [enc_out[3], enc_out[2], enc_out[1], enc_out[0]]
        # Coarse channels, deepest-first: enc_out[4], then previous FP outputs
        coarse_channels = [enc_out[4]] + [m[-1] for m in dec_mlps[:-1]]
        for coarse_c, skip_c, mlp in zip(coarse_channels, skip_channels, dec_mlps):
            fp = FeaturePropagationBlock(coarse_c + skip_c, mlp)
            self.decoder.append(fp)

        self.head = SegmentationHead(dec_mlps[-1][-1], num_classes,
                                     hidden=head_hidden, dropout=dropout)
        self.in_channels = in_channels

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """xyz [B, N, 3], features [B, N, C_in] → logits [B, N, num_classes]."""
        # If no features provided, use xyz itself as input features.
        # This matches the pretty-common convention in PointNet++ reimplementations
        # and keeps the decoder's final skip connection well-defined.
        if features is None:
            features = xyz

        xyz_list = [xyz]
        feat_list = [features]
        x, f = xyz, features
        for sa in self.encoder:
            x, f = sa(x, f)
            xyz_list.append(x)
            feat_list.append(f)

        # decoder walks from deepest back to input resolution
        f_cur = feat_list[-1]
        for i, fp in enumerate(self.decoder):
            fine_idx = len(self.encoder) - 1 - i     # 3, 2, 1, 0
            coarse_idx = fine_idx + 1                 # 4, 3, 2, 1
            xyz_fine = xyz_list[fine_idx]
            xyz_coarse = xyz_list[coarse_idx]
            skip = feat_list[fine_idx]
            f_cur = fp(xyz_fine, xyz_coarse, skip, f_cur)
        return self.head(f_cur)


# --------------------------------------------------------------------------- #
# Part segmentation (same U-Net + part head)
# --------------------------------------------------------------------------- #
class PointNet2PartSegModel(nn.Module):
    def __init__(self, in_channels: int = 3, num_parts: int = 50,
                 num_obj_classes: Optional[int] = 16,
                 npoints: Tuple[int, ...] = (512, 128, 32, 8),
                 radii: Tuple[float, ...] = (0.1, 0.2, 0.4, 0.8),
                 nsamples: Tuple[int, ...] = (32, 32, 32, 32),
                 normalize_dp: bool = False,
                 head_hidden: int = 128, dropout: float = 0.5):
        super().__init__()
        # Reuse the segmentation backbone; swap the head.
        self.backbone = PointNet2SegModel(
            in_channels=in_channels, num_classes=num_parts,
            npoints=npoints, radii=radii, nsamples=nsamples,
            normalize_dp=normalize_dp, head_hidden=head_hidden, dropout=dropout,
        )
        # Replace the head with a part-seg one
        dec_last_c = self.backbone.head.conv[0].in_channels  # hidden is identical
        # Actually, we want access to the channel just before the head.
        # The backbone's decoder last dec_mlp[-1] channel is fed to the head.
        last_c = self.backbone.decoder[-1].out_channels
        self.backbone.head = PartSegmentationHead(
            last_c, num_parts, num_obj_classes=num_obj_classes,
            hidden=head_hidden, dropout=dropout,
        )
        self.num_obj_classes = num_obj_classes

    def forward(self, xyz: torch.Tensor, features: torch.Tensor,
                obj_label: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Copy of the seg forward, minus the final head call.
        b = self.backbone
        xyz_list = [xyz]
        feat_list = [features]
        x, f = xyz, features
        for sa in b.encoder:
            x, f = sa(x, f)
            xyz_list.append(x)
            feat_list.append(f)
        f_cur = feat_list[-1]
        for i, fp in enumerate(b.decoder):
            fine_idx = len(b.encoder) - 1 - i
            coarse_idx = fine_idx + 1
            f_cur = fp(xyz_list[fine_idx], xyz_list[coarse_idx],
                       feat_list[fine_idx], f_cur)
        return b.head(f_cur, obj_label=obj_label)
