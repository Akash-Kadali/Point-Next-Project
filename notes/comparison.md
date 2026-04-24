# Comparing PointNet++, PointNeXt, and Superpoint Transformer

This note is a plain-English, technically honest comparison of three
representative architectures for 3D point cloud understanding. The goal is
practical: help you pick the right baseline for a new problem.

## 1. What each method actually processes

The single biggest difference is the **unit of processing**:

- **PointNet++** (NeurIPS 2017). Operates on individual **points**. The
  encoder groups local neighborhoods around sampled centers, runs a shared
  MLP, and max-pools. It is a hierarchical version of PointNet.
- **PointNeXt** (NeurIPS 2022). Same processing unit as PointNet++ — still
  per-point. The architecture adds residual connections, an inverted
  bottleneck MLP, a stem MLP, normalized relative coordinates, and stacks
  InvResMLP blocks to scale depth. The paper's headline result is that
  training strategy and scaling matter more than new operators.
- **Superpoint Transformer** (SPT, ICCV 2023; the "Efficient 3D Semantic
  Segmentation with Superpoint Transformer" line of work). Operates on
  **superpoints** — geometrically homogeneous clusters of points produced
  by a partition step (often a graph-based over-segmentation like Cut-Pursuit
  or Felzenszwalb on a point adjacency graph). The network then runs a
  transformer over superpoints, which there are far fewer of than raw points.

Everything downstream of this choice follows from it. Point-level networks
need subsampling (FPS / voxel) to stay tractable; superpoint-level networks
get the subsampling for free from the partition step, at the cost of losing
fine detail inside a superpoint.

## 2. Where each method shines and struggles

### PointNet++
**Good at:** small-to-medium objects (ModelNet40, ShapeNet), clean inputs,
sanity-checks, pedagogy. It's the reference implementation against which
every new method compares.
**Struggles with:** large scenes (S3DIS, Semantic3D, nuScenes) — subsampling
plus small model capacity means it plateaus well below SOTA.

### PointNeXt
**Good at:** the same tasks as PointNet++ but much better on large scenes
thanks to the receptive-field and depth scaling. PointNeXt-XL reaches 74.9%
mIoU on S3DIS 6-fold (Tab. 1), which was competitive with Point Transformer
at the time. **It is the best starting point if you already have a
PointNet++ pipeline** — the gains come from drop-in changes, not a new
paradigm.
**Struggles with:** city-scale scenes (millions of points) where per-point
processing even with FPS becomes a memory bottleneck. You pay O(N) every
forward pass, and the architecture is not invariant to over-sampled
ground / walls.

### Superpoint Transformer
**Good at:** very large scenes. Because the transformer sees O(superpoints)
tokens (~1000 per scene) rather than O(points) (~1M per scene), it can
build global context cheaply. SPT reports competitive mIoU on S3DIS /
Semantic3D / KITTI-360 while being ~10–100× faster at inference than
point-level networks on the same scenes.
**Struggles with:** tasks where the relevant structure is *inside* the
superpoint (fine part boundaries, small objects); partition quality becomes
a hard floor on what the network can learn. Also: the partition is a
separate non-differentiable preprocessing step, which adds complexity and
makes end-to-end training awkward.

## 3. When to pick which

A rough decision tree:

- **"I just want a good per-point baseline on object-level data"** →
  PointNeXt-S or PointNeXt-B. Clean, scalable, well-studied.
- **"I have an S3DIS-scale scene and a V100"** → PointNeXt-L or -XL. SPT is
  also a good choice if inference latency matters.
- **"My point cloud has 5M+ points per scene (aerial LiDAR, driving, urban
  mapping)"** → start with SPT. Point-level networks need either aggressive
  downsampling or chunking, both of which hurt context.
- **"I care about fine part segmentation / small objects"** → PointNeXt.
  Superpoint aggregation will blur the exact targets you care about.
- **"I need differentiable end-to-end training from raw points"** →
  PointNeXt. SPT requires the separate partition step.

## 4. Comparison table

| Aspect | PointNet++ | PointNeXt | Superpoint Transformer |
|---|---|---|---|
| Processing unit | Point | Point | Superpoint (partitioned cluster) |
| Pre-processing | Random / FPS sampling | Same + resampling | Graph over-segmentation |
| Core op | Shared MLP + max-pool | Shared MLP + InvResMLP (residual + inverted bottleneck + separable MLP) | Self-attention over superpoints |
| Receptive field | Local neighborhoods | Local neighborhoods + depth | Global (transformer) |
| Typical size | ~1.5 M params | 0.8 M (S) → 41 M (XL) | ~0.2–5 M params |
| Throughput class | High | High (S) → Medium (XL) | Very high on large scenes |
| Strongest benchmarks | ShapeNet, ModelNet | ScanObjectNN, S3DIS, ShapeNetPart | S3DIS, Semantic3D, KITTI-360 |
| Failure modes | Too shallow to scale | Memory on city-scale scenes | Partition errors, fine detail loss |
| Training | Fully supervised end-to-end | Fully supervised end-to-end | Supervised, non-differentiable partition step |

## 5. What I'd implement first, and why

If I were starting a new 3D segmentation project without strong priors, I'd
**implement PointNeXt first**. Three reasons:

1. **It's the smallest jump from a working PointNet++ baseline**, and the
   paper is explicit about which pieces of the gain come from training vs.
   architecture — which means the debugging story is clean. If your mIoU
   is low, you can isolate whether it's a data issue, training issue, or
   architecture issue.

2. **It scales gracefully.** The S/B/L/XL presets let me sweep capacity
   without rewriting anything, which is exactly what you want for a quick
   "is this task even learnable?" check.

3. **It has no external pre-processing dependency.** Superpoint methods
   require a partition step that is not shipped in most point-cloud
   libraries and is a source of version skew across papers.

Once I had a PointNeXt baseline and had validated the data pipeline, I'd
benchmark SPT on the same data **only if**: (a) the scenes are big enough
that my PointNeXt forward pass is the bottleneck, or (b) mIoU has
plateaued and I suspect the issue is missing global context. Otherwise
PointNeXt gives you 90% of the benefit with 50% of the moving parts.
