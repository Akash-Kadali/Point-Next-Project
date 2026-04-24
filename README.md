# PointNeXt Reproduction — Clean PyTorch Project

A beginner-friendly, modular PyTorch reproduction of the PointNeXt paper
("PointNeXt: Revisiting PointNet++ with Improved Training and Scaling
Strategies", Qian et al., NeurIPS 2022).

The project is organized so each stage of the paper (baseline → improved
training → architecture changes → tasks → ablations) lives in its own module.
Everything runs on toy synthetic data by default so you can smoke-test the
whole pipeline in minutes on a CPU, then swap in real datasets via config.

## What's in the box

| Module | What it does |
|---|---|
| `datasets/` | Toy cls / seg / partseg datasets, preprocessing, 8 augmentations from the paper, config-driven augmentation factory |
| `models/` | PointNet++ baseline (cls/seg/partseg), PointNeXt (cls/seg/partseg), all 4 presets (S / B / L / XL), InvResMLP block, normalized Δp, stem MLP, residual SA, symmetric decoder |
| `training/` | Trainer loop with checkpointing and resume, CE + label smoothing, Adam/AdamW/SGD, Step/MultiStep/Cosine schedulers |
| `evaluation/` | ConfusionMatrix, OA, mAcc, per-class IoU, mIoU, instance mIoU (part seg) |
| `visualization/` | Matplotlib 3D scatter: raw / labeled / pred-vs-GT / before-after |
| `unsupervised/` | Encoder embedding extraction, KMeans + DBSCAN, PCA / t-SNE / UMAP projections, ARI / NMI / silhouette (labeled as an extension — not part of the paper) |
| `scripts/` | `train.py`, `evaluate.py`, `run_ablation.py`, `model_stats.py`, `demo_visualize.py`, `unsupervised_cluster.py` |
| `configs/` | One baseline + one PointNeXt YAML per task |
| `notes/` | PointNet++ / PointNeXt / Superpoint Transformer comparison, experiment report template |
| `tests/` | 15 smoke tests that cover the whole pipeline and pass end-to-end |

## Install

```bash
pip install -r requirements.txt
```

Only core deps (PyTorch, NumPy, PyYAML, matplotlib, scikit-learn, tqdm).
Open3D and UMAP are optional.

## Quick start (toy data, CPU-friendly)

```bash
# Run the smoke test suite — ~30 seconds, tests the whole pipeline
python tests/run_all.py

# Train classification end-to-end
python scripts/train.py --config configs/cls_pointnext_s.yaml

# Train semantic segmentation
python scripts/train.py --config configs/seg_pointnext_s.yaml

# Train part segmentation
python scripts/train.py --config configs/partseg_pointnext_s.yaml

# Evaluate a saved checkpoint
python scripts/evaluate.py --config configs/cls_pointnext_s.yaml \
                           --checkpoint runs/cls_pointnext_s/best.pt

# Run the additive ablation sweep (Tab. 4 / Tab. 5 style)
python scripts/run_ablation.py --task cls --epochs 5 --out results/ablation_cls.csv
python scripts/run_ablation.py --task seg --epochs 5 --out results/ablation_seg.csv

# Print param count and throughput
python scripts/model_stats.py --config configs/seg_pointnext_s.yaml

# Save prediction visualizations
python scripts/demo_visualize.py --config configs/seg_pointnext_s.yaml \
                                 --checkpoint runs/seg_pointnext_s/best.pt \
                                 --out-dir vis/seg

# Unsupervised embedding + KMeans + t-SNE projection
python scripts/unsupervised_cluster.py --config configs/cls_pointnext_s.yaml \
                                       --checkpoint runs/cls_pointnext_s/best.pt \
                                       --method kmeans --project tsne \
                                       --out-dir vis/unsup
```

## Configs

Each YAML has these blocks:

```yaml
model:        # name, preset (s/b/l/xl), in_channels, num_classes, normalize_dp
dataset:      # name, num_samples, num_points, batch_size, augmentation, val_transform
val_dataset:  # optional override for the val split
criterion:    # cross_entropy, label_smoothing
optimizer:    # adam / adamw / sgd
scheduler:    # step / multistep / cosine
training:     # num_epochs, monitor_metric, parts_per_obj (partseg only)
```

Toggle an individual augmentation on / off by editing the `augmentation:`
block — the `build_augmentation` factory picks up any key that's set to
truthy. For ablations, let `scripts/run_ablation.py` do the sweeping.

One gotcha: the `val_transform` key under `dataset` (and `val_dataset`) exists
because some steps labeled "augmentation" in the paper (especially
`height_appending`) are actually deterministic feature-generation steps that
must be applied to both splits or the model input shape won't match. Keep
`height_appending` under `val_transform` anywhere it's under `augmentation`.

## Using real datasets

All toy datasets live in `datasets/toy.py`. To plug in a real dataset:

1. Write a new `Dataset` class that returns the same dict format:
   `{"xyz": [N,3] float, "features": [N,C] float, "label": ... }` plus
   `"obj_label"` for part segmentation.
2. Register it in `datasets/loader.py`:
   ```python
   from datasets.s3dis import S3DISDataset
   register_dataset("s3dis", S3DISDataset)
   ```
3. Point your config at the new name:
   ```yaml
   dataset:
     name: s3dis
     kwargs:
       root: /path/to/S3DIS
       ...
   ```

## Reading order

If you want to understand the project end-to-end, read in this order:

1. `configs/cls_pointnext_s.yaml` — see what an experiment looks like
2. `datasets/augmentation.py` — the paper's training-recipe knobs
3. `models/blocks.py` — `SetAbstractionBlock` and `InvResMLPBlock` side-by-side
4. `models/pointnet2.py` → `models/pointnext.py` — baseline, then the upgrade
5. `training/trainer.py` — the unified training loop
6. `scripts/run_ablation.py` — how ablations are parameterized
7. `notes/comparison.md` — when to reach for PointNeXt vs alternatives

## What's synthetic vs. what's tested

Because the toy datasets are procedurally generated (sphere / cube / cylinder /
cone for cls; ground / wall / furniture planes for seg), don't read meaningful
accuracy numbers out of the smoke runs. The value of the toy stack is:

- Verify every code path works end-to-end
- Confirm ablation patches apply correctly (checkpoint metadata shows this)
- Benchmark inference throughput / memory footprint of each model size

When you swap in real data, you should see the paper's pattern: the baseline
gets a big jump from the improved training alone (Tab. 4 first block), and a
smaller additional jump from PointNeXt-S (Tab. 4 last row).

## Performance notes

This project uses pure-PyTorch implementations of FPS and ball query for
readability. For production use on S3DIS-scale scenes you'll want to swap
these for the compiled CUDA versions from `pointnet2_ops` — drop them into
`models/pointnet_ops.py` replacing the pure-Python ones. The rest of the
project doesn't need to change.

## Citation

```
@inproceedings{qian2022pointnext,
  title={PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies},
  author={Qian, Guocheng and Li, Yuchen and Peng, Houwen and Mai, Jinjie and Hammoud, Hasan Abed Al Kader and Elhoseiny, Mohamed and Ghanem, Bernard},
  booktitle={NeurIPS},
  year={2022}
}
```
