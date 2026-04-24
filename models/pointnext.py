"""PointNeXt models.

Differences vs. PointNet++ (per Section 3.2 of the paper, colored red in Fig. 2):
    1. Stem MLP at the very start (in_channels → C).
    2. Residual connection inside every SA block.
    3. Relative coordinates normalized by the radius (Eq. 2).
    4. Separable MLP idea inside InvResMLP (1 neighborhood layer + 2 pointwise layers).
    5. Inverted-residual bottleneck design in InvResMLP (expansion ×4).
    6. InvResMLP blocks stacked after the main SA block per stage.
    7. Symmetric decoder (channel sizes match encoder).
    8. Presets for PointNeXt-S, -B, -L, -XL.

Model-scaling presets (from the paper, Section 3.2.2):
    PointNeXt-S:  C=32, B=(0, 0, 0, 0)
    PointNeXt-B:  C=32, B=(1, 2, 1, 1)
    PointNeXt-L:  C=32, B=(2, 4, 2, 2)
    PointNeXt-XL: C=64, B=(3, 6, 3, 3)
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import torch
import torch.nn as nn

from .blocks import SetAbstractionBlock, FeaturePropagationBlock, InvResMLPBlock
from .heads import ClassificationHead, SegmentationHead, PartSegmentationHead


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #
POINTNEXT_PRESETS = {
    "s":  {"width": 32, "blocks": (0, 0, 0, 0)},
    "b":  {"width": 32, "blocks": (1, 2, 1, 1)},
    "l":  {"width": 32, "blocks": (2, 4, 2, 2)},
    "xl": {"width": 64, "blocks": (3, 6, 3, 3)},
}


# --------------------------------------------------------------------------- #
# Encoder stage: SA + N×InvResMLP
# --------------------------------------------------------------------------- #
class PointNeXtStage(nn.Module):
    """One encoder stage: a Set Abstraction block followed by `num_blocks`
    InvResMLP blocks at the same resolution."""

    def __init__(self, in_channels: int, out_channels: int,
                 npoint: Optional[int], radius: float, nsample: int,
                 num_blocks: int, expansion: int = 4,
                 sa_mlp_layers: int = 2, normalize_dp: bool = True):
        super().__init__()
        # Per paper: when B=0 (no InvResMLP), use 2 MLP layers in SA.
        # When B!=0, SA uses 1 MLP layer to save compute. We follow that.
        if num_blocks == 0:
            sa_mlp = [out_channels] * 2
        else:
            sa_mlp = [out_channels]
        self.sa = SetAbstractionBlock(
            in_channels=in_channels, mlp_channels=sa_mlp,
            npoint=npoint, radius=radius, nsample=nsample,
            normalize_dp=normalize_dp, residual=True,
        )
        self.blocks = nn.ModuleList([
            InvResMLPBlock(out_channels, radius=radius, nsample=nsample,
                           expansion=expansion, normalize_dp=normalize_dp)
            for _ in range(num_blocks)
        ])
        self.out_channels = out_channels

    def forward(self, xyz: torch.Tensor, feats: Optional[torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        xyz_out, feats_out = self.sa(xyz, feats)
        for blk in self.blocks:
            feats_out = blk(xyz_out, feats_out)
        return xyz_out, feats_out


# --------------------------------------------------------------------------- #
# Segmentation: encoder + symmetric decoder
# --------------------------------------------------------------------------- #
class PointNeXtSegModel(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 13,
                 width: int = 32, blocks: Tuple[int, ...] = (0, 0, 0, 0),
                 npoints: Tuple[int, ...] = (1024, 256, 64, 16),
                 radii: Tuple[float, ...] = (0.1, 0.2, 0.4, 0.8),
                 nsamples: Tuple[int, ...] = (32, 32, 32, 32),
                 expansion: int = 4, normalize_dp: bool = True,
                 head_hidden: int = 64, dropout: float = 0.5):
        super().__init__()
        assert len(blocks) == 4 == len(npoints) == len(radii) == len(nsamples)

        # Stem MLP: in_channels → width
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, width, 1, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
        )

        # Encoder: width → 2w → 4w → 8w → 16w
        enc_channels = [width * (2 ** i) for i in range(5)]  # [w, 2w, 4w, 8w, 16w]
        self.encoder = nn.ModuleList()
        for i in range(4):
            stage = PointNeXtStage(
                in_channels=enc_channels[i],
                out_channels=enc_channels[i + 1],
                npoint=npoints[i], radius=radii[i], nsample=nsamples[i],
                num_blocks=blocks[i], expansion=expansion,
                normalize_dp=normalize_dp,
            )
            self.encoder.append(stage)

        # Decoder (symmetric): channel sizes mirror encoder.
        # At decoder step i (i=3,2,1,0):
        #     coarse features come from enc_channels[i+1] (either directly from
        #     deepest encoder for i=3, or from previous FP's output for i<3,
        #     which also outputs enc_channels[i+1]).
        #     skip features come from enc_channels[i].
        #     output: enc_channels[i].
        # So the FP's in_channels is always enc_channels[i+1] + enc_channels[i].
        self.decoder = nn.ModuleList()
        for i in range(3, -1, -1):
            coarse_in = enc_channels[i + 1]
            skip_c = enc_channels[i]
            out_c = enc_channels[i]
            fp = FeaturePropagationBlock(
                in_channels=coarse_in + skip_c,
                mlp_channels=[out_c, out_c],
            )
            self.decoder.append(fp)

        self.head = SegmentationHead(width, num_classes,
                                     hidden=head_hidden, dropout=dropout)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        # Stem
        x_in = features.transpose(1, 2)  # [B, C, N]
        x_in = self.stem(x_in)
        f = x_in.transpose(1, 2)          # [B, N, width]

        xyz_list = [xyz]
        feat_list = [f]
        cur_xyz = xyz
        cur_feats = f
        for stage in self.encoder:
            cur_xyz, cur_feats = stage(cur_xyz, cur_feats)
            xyz_list.append(cur_xyz)
            feat_list.append(cur_feats)

        # Decoder walks from deepest back to stem
        f_cur = feat_list[-1]
        for i, fp in enumerate(self.decoder):
            fine_idx = 3 - i              # 3, 2, 1, 0
            coarse_idx = fine_idx + 1      # 4, 3, 2, 1
            f_cur = fp(xyz_list[fine_idx], xyz_list[coarse_idx],
                       feat_list[fine_idx], f_cur)

        return self.head(f_cur)


# --------------------------------------------------------------------------- #
# Classification: same encoder, global pool, no decoder
# --------------------------------------------------------------------------- #
class PointNeXtClsModel(nn.Module):
    """Per Fig. I (supplementary): classification encoder has 4 stages too,
    with 2× downsampling each time (instead of 4× for segmentation)."""

    def __init__(self, in_channels: int = 3, num_classes: int = 15,
                 width: int = 32, blocks: Tuple[int, ...] = (0, 0, 0, 0),
                 npoints: Tuple[int, ...] = (512, 256, 128, 64),
                 radii: Tuple[float, ...] = (0.15, 0.3, 0.6, 1.2),
                 nsamples: Tuple[int, ...] = (32, 32, 32, 32),
                 expansion: int = 4, normalize_dp: bool = True,
                 head_hidden: int = 256, dropout: float = 0.5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, width, 1, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
        )
        enc_channels = [width * (2 ** i) for i in range(5)]
        self.encoder = nn.ModuleList()
        for i in range(4):
            self.encoder.append(PointNeXtStage(
                in_channels=enc_channels[i],
                out_channels=enc_channels[i + 1],
                npoint=npoints[i], radius=radii[i], nsample=nsamples[i],
                num_blocks=blocks[i], expansion=expansion,
                normalize_dp=normalize_dp,
            ))
        self.head = ClassificationHead(enc_channels[4], num_classes,
                                       hidden=head_hidden, dropout=dropout)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x = features.transpose(1, 2)
        x = self.stem(x)
        f = x.transpose(1, 2)
        cur_xyz, cur_feats = xyz, f
        for stage in self.encoder:
            cur_xyz, cur_feats = stage(cur_xyz, cur_feats)
        return self.head(cur_feats)

    @torch.no_grad()
    def extract_embedding(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Return the global pooled encoder feature (for unsupervised use)."""
        x = features.transpose(1, 2)
        x = self.stem(x)
        f = x.transpose(1, 2)
        cur_xyz, cur_feats = xyz, f
        for stage in self.encoder:
            cur_xyz, cur_feats = stage(cur_xyz, cur_feats)
        return cur_feats.max(dim=1)[0]  # [B, C]


# --------------------------------------------------------------------------- #
# Part segmentation
# --------------------------------------------------------------------------- #
class PointNeXtPartSegModel(nn.Module):
    def __init__(self, in_channels: int = 3, num_parts: int = 50,
                 num_obj_classes: Optional[int] = 16,
                 width: int = 32, blocks: Tuple[int, ...] = (0, 0, 0, 0),
                 npoints: Tuple[int, ...] = (512, 128, 32, 8),
                 radii: Tuple[float, ...] = (0.1, 0.2, 0.4, 0.8),
                 nsamples: Tuple[int, ...] = (32, 32, 32, 32),
                 expansion: int = 4, normalize_dp: bool = True,
                 head_hidden: int = 128, dropout: float = 0.5):
        super().__init__()
        self.backbone = PointNeXtSegModel(
            in_channels=in_channels, num_classes=num_parts,
            width=width, blocks=blocks,
            npoints=npoints, radii=radii, nsamples=nsamples,
            expansion=expansion, normalize_dp=normalize_dp,
            head_hidden=head_hidden, dropout=dropout,
        )
        last_c = self.backbone.decoder[-1].out_channels
        self.backbone.head = PartSegmentationHead(
            last_c, num_parts, num_obj_classes=num_obj_classes,
            hidden=head_hidden, dropout=dropout,
        )

    def forward(self, xyz: torch.Tensor, features: torch.Tensor,
                obj_label: Optional[torch.Tensor] = None) -> torch.Tensor:
        b = self.backbone
        x = features.transpose(1, 2)
        x = b.stem(x)
        f = x.transpose(1, 2)

        xyz_list = [xyz]
        feat_list = [f]
        cur_xyz, cur_feats = xyz, f
        for stage in b.encoder:
            cur_xyz, cur_feats = stage(cur_xyz, cur_feats)
            xyz_list.append(cur_xyz)
            feat_list.append(cur_feats)

        f_cur = feat_list[-1]
        for i, fp in enumerate(b.decoder):
            fine_idx = 3 - i
            coarse_idx = fine_idx + 1
            f_cur = fp(xyz_list[fine_idx], xyz_list[coarse_idx],
                       feat_list[fine_idx], f_cur)

        return b.head(f_cur, obj_label=obj_label)
