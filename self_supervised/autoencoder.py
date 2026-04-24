import torch
import torch.nn as nn


class PointNeXtAutoEncoder(nn.Module):
    def __init__(self, backbone, embedding_dim=512, num_points=512):
        super().__init__()
        self.backbone = backbone
        self.num_points = num_points

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_points * 3),
        )

    def forward(self, xyz, features):
        emb = self.backbone.extract_embedding(xyz, features)
        recon = self.decoder(emb)
        recon = recon.view(-1, self.num_points, 3)
        return recon, emb


def chamfer_distance(pred, target):
    """
    pred:   [B, N, 3]
    target: [B, N, 3]
    """
    dist = torch.cdist(pred, target)  # [B, N, N]

    pred_to_target = dist.min(dim=2)[0].mean(dim=1)
    target_to_pred = dist.min(dim=1)[0].mean(dim=1)

    return (pred_to_target + target_to_pred).mean()