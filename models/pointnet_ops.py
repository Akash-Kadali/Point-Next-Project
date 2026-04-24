"""Low-level point cloud operations in pure PyTorch.

These are simple, educational implementations. They are O(N^2) memory in
some places (pairwise distances), which is fine for toy/medium inputs but
should be swapped for CUDA kernels in production (e.g., pointnet2_ops).

All ops assume channels-last for point-wise tensors:
    xyz:   [B, N, 3]
    feats: [B, N, C]
"""
from __future__ import annotations
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather rows from `points` using `idx`.

    Args:
        points: [B, N, C]
        idx:    [B, S] or [B, S, K]
    Returns:
        [B, S, C] or [B, S, K, C]
    """
    B = points.shape[0]
    # Build a batch index tensor of shape (B, 1, ..., 1) and broadcast it
    # against `idx`. We rely on advanced indexing broadcasting rather than
    # `expand` (which only expands singleton dims).
    view = (B,) + (1,) * (idx.dim() - 1)
    batch_idx = torch.arange(B, device=points.device).view(*view)
    return points[batch_idx, idx]


# --------------------------------------------------------------------------- #
# Farthest Point Sampling (GPU-friendly loop)
# --------------------------------------------------------------------------- #
def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """FPS implemented in torch. Returns indices [B, npoint].

    This is the standard iterative FPS. It's O(B * npoint * N) compute and
    memory-light. For N up to a few thousand it is fast enough on GPU.
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    # start from a random point per batch
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)  # [B, 1, 3]
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)  # [B, N]
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


# --------------------------------------------------------------------------- #
# Neighbor queries
# --------------------------------------------------------------------------- #
def square_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise squared distances. a: [B, N, 3], b: [B, M, 3] -> [B, N, M]."""
    # || a - b ||^2 = ||a||^2 + ||b||^2 - 2 a.b
    B, N, _ = a.shape
    M = b.shape[1]
    dist = -2 * torch.matmul(a, b.transpose(1, 2))  # [B, N, M]
    dist += torch.sum(a ** 2, dim=-1, keepdim=True)
    dist += torch.sum(b ** 2, dim=-1, keepdim=True).transpose(1, 2)
    return dist.clamp(min=0.0)


def ball_query(radius: float, nsample: int,
               xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """For each point in `new_xyz`, find up to `nsample` neighbors in `xyz`
    within `radius`. Any out-of-radius slot is filled with the first in-range
    neighbor (or the first point globally if the query has no neighbors).

    Handles the edge case where the cloud has fewer than `nsample` points by
    sampling with replacement from whatever is available.

    Args:
        xyz:     [B, N, 3] all points
        new_xyz: [B, S, 3] query centers
    Returns:
        idx: [B, S, nsample] indices into xyz
    """
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    sqrdists = square_distance(new_xyz, xyz)  # [B, S, N]
    group_idx = torch.arange(N, device=xyz.device).view(1, 1, N).expand(B, S, N).clone()
    group_idx[sqrdists > radius ** 2] = N  # sentinel: out of radius
    # Sort so in-range indices come first (sentinels drift to the back).
    group_idx = group_idx.sort(dim=-1)[0]   # [B, S, N] still

    # If we have fewer candidate points than nsample, pad by repeating (wrap).
    if N < nsample:
        reps = (nsample + N - 1) // N
        group_idx = group_idx.repeat(1, 1, reps)[:, :, :nsample]
    else:
        group_idx = group_idx[:, :, :nsample]

    # Replace remaining sentinels (==N) with the first valid index of that row.
    # If a row is all sentinels, use index 0 as a safe fallback.
    first = group_idx[:, :, 0:1].clone()
    first[first == N] = 0
    first = first.expand(-1, -1, nsample)
    mask = group_idx == N
    group_idx[mask] = first[mask]
    return group_idx


def knn(k: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """k-nearest neighbors. Returns indices [B, S, k]."""
    sqrdists = square_distance(new_xyz, xyz)
    return sqrdists.topk(k, dim=-1, largest=False)[1]


# --------------------------------------------------------------------------- #
# Shared MLPs
# --------------------------------------------------------------------------- #
class SharedMLP(nn.Module):
    """A stack of Conv1d(1x1) → BN → ReLU layers applied pointwise.

    Input is [B, C_in, N], output [B, C_out, N].
    """

    def __init__(self, channels, bn: bool = True, activation: bool = True,
                 last_activation: bool = True):
        super().__init__()
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv1d(channels[i], channels[i + 1], kernel_size=1, bias=not bn))
            if bn:
                layers.append(nn.BatchNorm1d(channels[i + 1]))
            is_last = (i == len(channels) - 2)
            if activation and (not is_last or last_activation):
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedMLP2d(nn.Module):
    """SharedMLP over 2D neighborhood features [B, C, N, K]."""

    def __init__(self, channels, bn: bool = True, activation: bool = True,
                 last_activation: bool = True):
        super().__init__()
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=1, bias=not bn))
            if bn:
                layers.append(nn.BatchNorm2d(channels[i + 1]))
            is_last = (i == len(channels) - 2)
            if activation and (not is_last or last_activation):
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
