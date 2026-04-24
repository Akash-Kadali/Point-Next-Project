import torch
import math


def random_rotation_z(xyz):
    B = xyz.shape[0]
    angles = torch.rand(B, device=xyz.device) * 2 * math.pi
    cos, sin = torch.cos(angles), torch.sin(angles)

    R = torch.zeros(B, 3, 3, device=xyz.device)
    R[:, 0, 0] = cos
    R[:, 0, 1] = -sin
    R[:, 1, 0] = sin
    R[:, 1, 1] = cos
    R[:, 2, 2] = 1.0

    return torch.bmm(xyz, R)


def random_scaling(xyz, scale_min=0.8, scale_max=1.2):
    B = xyz.shape[0]
    scale = torch.empty(B, 1, 1, device=xyz.device).uniform_(scale_min, scale_max)
    return xyz * scale


def jitter(xyz, sigma=0.01, clip=0.05):
    noise = torch.clamp(torch.randn_like(xyz) * sigma, -clip, clip)
    return xyz + noise


def point_dropout(xyz, features, drop_ratio=0.2):
    B, N, _ = xyz.shape
    keep = torch.rand(B, N, device=xyz.device) > drop_ratio

    # replace dropped points with first point
    xyz_out = xyz.clone()
    feat_out = features.clone()

    for b in range(B):
        xyz_out[b, ~keep[b]] = xyz[b, 0]
        feat_out[b, ~keep[b]] = features[b, 0]

    return xyz_out, feat_out


def random_crop(xyz, features, crop_ratio=0.8):
    B, N, _ = xyz.shape
    keep_n = int(N * crop_ratio)

    xyz_out, feat_out = [], []

    for b in range(B):
        center = xyz[b, torch.randint(0, N, (1,), device=xyz.device)]
        dist = torch.norm(xyz[b] - center, dim=1)
        idx = torch.argsort(dist)[:keep_n]

        # resample back to original N
        idx = idx[torch.randint(0, keep_n, (N,), device=xyz.device)]

        xyz_out.append(xyz[b, idx])
        feat_out.append(features[b, idx])

    return torch.stack(xyz_out), torch.stack(feat_out)


def augment_point_cloud(xyz, features):
    xyz = random_rotation_z(xyz)
    xyz = random_scaling(xyz)
    xyz = jitter(xyz)
    xyz, features = point_dropout(xyz, features, drop_ratio=0.15)
    xyz, features = random_crop(xyz, features, crop_ratio=0.85)
    return xyz, features


def make_two_views(xyz, features):
    xyz1, feat1 = augment_point_cloud(xyz.clone(), features.clone())
    xyz2, feat2 = augment_point_cloud(xyz.clone(), features.clone())
    return xyz1, feat1, xyz2, feat2