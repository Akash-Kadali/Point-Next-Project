import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNeXtContrastive(nn.Module):
    def __init__(self, backbone, embedding_dim=512, projection_dim=128):
        super().__init__()
        self.backbone = backbone

        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, projection_dim),
        )

    def forward(self, xyz, features):
        emb = self.backbone.extract_embedding(xyz, features)
        z = self.projector(emb)
        z = F.normalize(z, dim=1)
        return z


def nt_xent_loss(z1, z2, temperature=0.2):
    B = z1.shape[0]

    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature

    mask = torch.eye(2 * B, device=z.device).bool()
    sim = sim.masked_fill(mask, -1e9)

    positives = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B, device=z.device)
    ])

    loss = F.cross_entropy(sim, positives)
    return loss