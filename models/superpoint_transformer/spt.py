import torch
import torch.nn as nn
import torch.nn.functional as F


def make_superpoints(xyz, features, num_superpoints=64):
    """
    Simple superpoint creation using random seed points + nearest assignment.
    This is a lightweight replacement for real graph/Cut-Pursuit partitioning.

    xyz:      [B, N, 3]
    features: [B, N, C]
    returns:
        sp_xyz:      [B, S, 3]
        sp_features: [B, S, C + 3]
        assign:      [B, N]
    """
    B, N, _ = xyz.shape
    S = min(num_superpoints, N)

    seed_idx = torch.randperm(N, device=xyz.device)[:S]
    seeds = xyz[:, seed_idx, :]  # [B, S, 3]

    dist = torch.cdist(xyz, seeds)  # [B, N, S]
    assign = dist.argmin(dim=-1)    # [B, N]

    sp_xyz_list = []
    sp_feat_list = []

    for b in range(B):
        cur_xyz = []
        cur_feat = []

        for s in range(S):
            mask = assign[b] == s

            if mask.sum() == 0:
                cur_xyz.append(seeds[b, s])
                pooled_feat = torch.zeros(features.shape[-1], device=xyz.device)
                cur_feat.append(torch.cat([seeds[b, s], pooled_feat], dim=0))
            else:
                pts = xyz[b, mask]
                feats = features[b, mask]

                center = pts.mean(dim=0)
                feat_mean = feats.mean(dim=0)

                cur_xyz.append(center)
                cur_feat.append(torch.cat([center, feat_mean], dim=0))

        sp_xyz_list.append(torch.stack(cur_xyz))
        sp_feat_list.append(torch.stack(cur_feat))

    sp_xyz = torch.stack(sp_xyz_list)          # [B, S, 3]
    sp_features = torch.stack(sp_feat_list)    # [B, S, C + 3]

    return sp_xyz, sp_features, assign


class SuperpointTransformerSeg(nn.Module):
    """
    Minimal Superpoint Transformer for semantic segmentation.

    Pipeline:
        points -> superpoints -> transformer over superpoints
        -> superpoint logits -> map logits back to points
    """

    def __init__(
        self,
        in_channels=4,
        num_classes=3,
        num_superpoints=64,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.num_superpoints = num_superpoints
        self.num_classes = num_classes

        # +3 because superpoint token uses xyz center + pooled features
        self.input_proj = nn.Linear(in_channels + 3, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, xyz, features):
        """
        xyz:      [B, N, 3]
        features: [B, N, C]
        returns:
            point_logits: [B, N, num_classes]
        """
        sp_xyz, sp_features, assign = make_superpoints(
            xyz,
            features,
            num_superpoints=self.num_superpoints,
        )

        tokens = self.input_proj(sp_features)
        tokens = self.transformer(tokens)

        sp_logits = self.head(tokens)  # [B, S, num_classes]

        B, N, _ = xyz.shape
        point_logits = torch.zeros(
            B,
            N,
            self.num_classes,
            device=xyz.device,
        )

        for b in range(B):
            point_logits[b] = sp_logits[b, assign[b]]

        return point_logits