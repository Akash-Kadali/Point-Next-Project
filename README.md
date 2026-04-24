# A Unified Framework for Point Cloud Learning: Extending PointNeXt toward Self-Supervised and Superpoint Transformer Paradigms

A clean, modular PyTorch project for reproducing **PointNeXt** and extending it toward **self-supervised learning**, **unsupervised clustering**, and **superpoint-based transformer ideas** for point cloud understanding.

This project started as a reproduction of the paper **“PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies”** and gradually evolved into a broader experimental framework for 3D point cloud learning. The goal is not only to reproduce the supervised pipeline, but also to make it easy to explore modern directions such as representation learning and scalable transformer-based reasoning.

The codebase is organized to be beginner-friendly, easy to debug, and simple to extend. By default, everything runs on small synthetic toy datasets, so the full pipeline can be tested quickly on a CPU before moving to larger real datasets.

---

## What this project covers

This repository currently supports:

* **Supervised point cloud learning**

  * Classification
  * Semantic segmentation
  * Part segmentation

* **PointNeXt reproduction**

  * PointNet++ baseline
  * PointNeXt variants
  * Improved training strategies
  * Architecture scaling and ablations

* **Self-supervised learning extensions**

  * Contrastive representation learning
  * Autoencoder-based feature learning

* **Unsupervised learning**

  * Embedding extraction
  * KMeans and DBSCAN clustering
  * PCA / t-SNE / UMAP projections
  * ARI / NMI / silhouette evaluation

* **Scalable modeling ideas**

  * Superpoint-based tokenization
  * Initial Superpoint Transformer style experimentation

* **Utilities**

  * Visualization tools
  * Checkpointing and resume
  * Ablation runs
  * Model statistics
  * Smoke tests for the whole pipeline

---

## Project structure

| Folder           | Purpose                                                                              |
| ---------------- | ------------------------------------------------------------------------------------ |
| `datasets/`      | Toy datasets, preprocessing, augmentations, dataset loading                          |
| `models/`        | PointNet++ baseline, PointNeXt models, shared blocks, scalable model variants        |
| `training/`      | Training loop, checkpointing, resume logic, optimizer and scheduler support          |
| `evaluation/`    | Accuracy, mAcc, IoU, mIoU, instance mIoU, clustering metrics                         |
| `visualization/` | 3D scatter plots and prediction visualization tools                                  |
| `unsupervised/`  | Embedding extraction, clustering, dimensionality reduction                           |
| `scripts/`       | Training, evaluation, ablation, visualization, model stats, unsupervised experiments |
| `configs/`       | YAML experiment configs for classification, segmentation, and part segmentation      |
| `notes/`         | Comparison notes, experiment summaries, paper reading notes                          |
| `tests/`         | Smoke tests covering the pipeline end-to-end                                         |

---

## Main models included

### Supervised models

* PointNet++ baseline
* PointNeXt-S
* PointNeXt-B
* Support for larger presets such as L / XL if enabled in config

### Self-supervised extensions

* Contrastive encoder pipeline
* Autoencoder-based reconstruction pipeline

### Experimental transformer direction

* Superpoint construction
* Superpoint-level feature aggregation
* Superpoint Transformer style pipeline for scalable reasoning

---

## Why toy datasets are used first

The repository uses synthetic toy datasets by default because they make debugging much easier.

These datasets help verify that:

* data loading works correctly,
* models run end-to-end,
* losses decrease,
* metrics are computed properly,
* checkpoints save and load correctly,
* ablation scripts behave as expected,
* visualization tools produce the expected outputs.

The current toy datasets are useful for implementation validation, but they are **not** meant to replace real benchmarks.

---

## Toy datasets included

| Task                  | Dataset       | Description                                                  |
| --------------------- | ------------- | ------------------------------------------------------------ |
| Classification        | `toy_cls`     | Simple synthetic shapes such as cube, sphere, cylinder, cone |
| Semantic Segmentation | `toy_seg`     | Small synthetic scenes with basic classes                    |
| Part Segmentation     | `toy_partseg` | Simple objects split into semantic parts                     |

---

## Installation

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

Core dependencies include:

* PyTorch
* NumPy
* PyYAML
* matplotlib
* scikit-learn
* tqdm

Optional:

* UMAP
* Open3D

---

## Quick start

### 1. Run the full smoke test suite

```bash
python tests/run_all.py
```

This is the best first step. It checks that the overall pipeline is working.

### 2. Train classification

```bash
python scripts/train.py --config configs/cls_pointnext_s.yaml
```

### 3. Train semantic segmentation

```bash
python scripts/train.py --config configs/seg_pointnext_s.yaml
```

### 4. Train part segmentation

```bash
python scripts/train.py --config configs/partseg_pointnext_s.yaml
```

### 5. Evaluate a checkpoint

```bash
python scripts/evaluate.py --config configs/cls_pointnext_s.yaml \
                           --checkpoint runs/cls_pointnext_s/best.pt
```

### 6. Run ablation experiments

```bash
python scripts/run_ablation.py --task cls --epochs 5 --out results/ablation_cls.csv
python scripts/run_ablation.py --task seg --epochs 5 --out results/ablation_seg.csv
```

### 7. View model statistics

```bash
python scripts/model_stats.py --config configs/seg_pointnext_s.yaml
```

### 8. Generate prediction visualizations

```bash
python scripts/demo_visualize.py --config configs/seg_pointnext_s.yaml \
                                 --checkpoint runs/seg_pointnext_s/best.pt \
                                 --out-dir vis/seg
```

### 9. Run unsupervised clustering

```bash
python scripts/unsupervised_cluster.py --config configs/cls_pointnext_s.yaml \
                                       --checkpoint runs/cls_pointnext_s/best.pt \
                                       --method kmeans --project tsne \
                                       --out-dir vis/unsup
```

---

## Self-supervised learning support

This project now includes early support for self-supervised representation learning.

### Contrastive learning

The encoder is trained using two augmented views of the same point cloud. The goal is to learn embeddings that remain consistent across transformations while separating different samples.

### Autoencoder learning

The encoder learns a compact latent representation, and the decoder reconstructs the original point cloud from that representation.

These two branches make it possible to explore representation learning before moving into downstream supervised or unsupervised tasks.

---

## Unsupervised learning support

The project also includes an unsupervised analysis pipeline.

Current support includes:

* feature extraction from trained encoders,
* clustering using **KMeans** and **DBSCAN**,
* embedding visualization with **PCA**, **t-SNE**, and **UMAP**,
* clustering quality metrics such as:

  * ARI
  * NMI
  * Silhouette Score

This part of the project is an extension and is **not part of the original PointNeXt paper**.

---

## Superpoint Transformer direction

Beyond PointNeXt, the project also begins exploring **superpoint-based transformer modeling**.

The idea is to first group points into superpoints, then perform learning and reasoning at the superpoint level instead of the individual point level. This reduces the number of tokens and makes transformer-style global reasoning more practical for large scenes.

This section is currently meant as an experimental extension and research direction, not as a final benchmark implementation.

---

## Configuration format

Each YAML config typically contains the following sections:

```yaml
model:
dataset:
val_dataset:
criterion:
optimizer:
scheduler:
training:
```

Typical config options include:

* model name and preset
* input channels
* number of classes
* dataset name and size
* number of points
* batch size
* augmentation settings
* optimizer choice
* scheduler choice
* number of epochs
* task-specific settings

---

## Data augmentation

The augmentation pipeline is config-driven. Common augmentations include:

* random rotation
* random scaling
* jittering
* point resampling
* height appending
* color dropping
* color auto-contrast

For ablation experiments, it is easier to use the provided ablation script rather than manually editing configs for each run.

---

## Reading order for the project

If you are new to the codebase, this is a good order to follow:

1. `configs/cls_pointnext_s.yaml`
   See what a full experiment looks like.

2. `datasets/augmentation.py`
   Understand the training recipe and augmentation setup.

3. `models/blocks.py`
   Review the core building blocks.

4. `models/pointnet2.py`
   Start with the baseline.

5. `models/pointnext.py`
   Then see how PointNeXt extends the baseline.

6. `training/trainer.py`
   Understand the unified training loop.

7. `scripts/run_ablation.py`
   See how ablations are structured.

8. `unsupervised/`
   Review embedding extraction and clustering.

9. `notes/comparison.md`
   Read the high-level comparison with related approaches.

---

## Using real datasets

All default examples use toy data, but the project is designed so real datasets can be added easily.

To plug in a real dataset:

1. Create a new `Dataset` class that returns the expected format:

   ```python
   {
     "xyz": ...,
     "features": ...,
     "label": ...
   }
   ```

   and include `"obj_label"` when needed for part segmentation.

2. Register the dataset in `datasets/loader.py`.

3. Point the config file to the new dataset name.

Possible next datasets:

* ScanObjectNN
* ShapeNetPart
* S3DIS

---

## Notes on performance

For clarity and readability, the project currently uses pure PyTorch implementations for some point operations such as FPS and ball query.

That makes the code easier to read and modify, but it is not the fastest option for large-scale training. For bigger datasets and production-level experiments, CUDA-optimized implementations such as `pointnet2_ops` should be used.

---

## What is tested

The test suite is designed to cover the full workflow:

* dataset loading
* augmentations
* model forward pass
* loss computation
* training loop
* evaluation metrics
* checkpoint saving/loading
* visualization calls
* unsupervised pipeline

All smoke tests are intended to make sure the project is stable before scaling to larger experiments.

---

## Current scope and limitations

This repository is strong as a **research prototype and implementation framework**, but it still has some current limits:

* default runs use toy datasets,
* reported performance on toy data is not meaningful for benchmarking,
* unsupervised and superpoint transformer components are still early-stage,
* larger benchmark validation is still pending.

So the current focus is correctness, flexibility, and extensibility.

---

## Planned next steps

* Add **ScanObjectNN** for real classification
* Add **ShapeNetPart** for real part segmentation
* Add **S3DIS** for real scene segmentation
* Expand self-supervised experiments
* Strengthen Superpoint Transformer implementation
* Compare PointNeXt with transformer-based 3D models on real data
* Add training curves and more qualitative visualization outputs

---

## Citation

If you use this repository, please cite the original PointNeXt paper:

```bibtex
@inproceedings{qian2022pointnext,
  title={PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies},
  author={Qian, Guocheng and Li, Yuchen and Peng, Houwen and Mai, Jinjie and Hammoud, Hasan Abed Al Kader and Elhoseiny, Mohamed and Ghanem, Bernard},
  booktitle={NeurIPS},
  year={2022}
}
```
