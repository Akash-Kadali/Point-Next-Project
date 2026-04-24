# Experiment Report Template

Copy this file to `notes/report_<date>.md` and fill in each section as you go.
Keep numbers in tables so diffs across runs stay readable.

## Objective

_(One paragraph: what question is this run answering? Don't just restate the
task — say what we expect to change vs the last run.)_

## Datasets

| Dataset | Split | Num samples | Num points | Input channels |
|---|---|---|---|---|
| _(e.g. toy_cls)_ | train | | | |
| | val | | | |

Notes on preprocessing (voxel size, subsampling, chunking strategy, etc.):

## Preprocessing

- Normalization: _(unit-ball? zero-mean? voxel-grid?)_
- Sampling strategy: _(FPS, random, voxel)_
- Handling of variable-sized clouds: _(pad, truncate, resample)_

## Baseline setup

- Model: _(PointNet++ with default config)_
- Optimizer / scheduler: _(Adam + Step)_
- Label smoothing ε: 0.0
- Augmentations: _(just random_rotation + jitter)_
- Num epochs, batch size, hardware:

## Improved training setup

- Changes relative to baseline (check each):
  - [ ] Point resampling
  - [ ] Height appending
  - [ ] Color dropping
  - [ ] Color auto-contrast
  - [ ] Random scaling
  - [ ] Label smoothing
  - [ ] AdamW
  - [ ] Cosine decay
- Expected effect: _(e.g. "+X mIoU based on Tab. 5 row Y")_

## PointNeXt architecture changes

- Preset: _(S / B / L / XL)_
- Stem MLP width C:
- InvResMLP blocks per stage B:
- normalize_dp: _(true/false)_
- Residual in SA block: _(true/false)_

## Results

### Semantic segmentation

| Config | OA | mAcc | mIoU | Params | Throughput |
|---|---|---|---|---|---|
| PointNet++ (baseline) | | | | | |
| + improved training | | | | | |
| + PointNeXt-S | | | | | |
| + PointNeXt-B | | | | | |
| + PointNeXt-L | | | | | |

### Classification

| Config | OA | mAcc | Params | Throughput |
|---|---|---|---|---|
| PointNet++ (baseline) | | | | |
| + improved training | | | | |
| + PointNeXt-S | | | | |

### Part segmentation

| Config | Ins. mIoU | Class mIoU | Params | Throughput |
|---|---|---|---|---|
| PointNet++ (baseline) | | | | |
| + improved training | | | | |
| + PointNeXt-S | | | | |

## Unsupervised clustering extension

| Metric | Value |
|---|---|
| Clustering method | _(kmeans / dbscan)_ |
| n_clusters | |
| ARI | |
| NMI | |
| Silhouette | |

## Qualitative visualizations

_Link to PNGs in vis/ — ideally at least one pred-vs-GT per task and one
before/after augmentation._

## Ablation results

_Paste the contents of `runs/ablation/<task>/results.md` here, or summarize
which changes moved the needle and which didn't. Flag anything surprising._

## Observations

1. _(The single most important finding.)_
2. _(Any unexpected regression or plateau.)_
3. _(How much of the gap is training vs architecture? — this is the paper's
   core question; answer it for your data.)_

## Next steps

- [ ] _(one concrete experiment)_
- [ ] _(one bigger question to investigate)_
