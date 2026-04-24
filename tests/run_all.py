"""Self-contained smoke tests. Run directly with:

    python tests/run_all.py

No pytest required. Each test function returns a (name, passed, message)
tuple, and the runner prints a summary at the end. Designed to finish in
under a minute on CPU.
"""
from __future__ import annotations
import sys
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------- #
# Individual tests
# --------------------------------------------------------------------------- #
def test_config_loader() -> None:
    from utils.config import Config
    c = Config({"a": 1, "b": {"c": 2}})
    assert c.a == 1 and c.b.c == 2
    d = c.merge({"b": {"c": 99}, "x": "y"})
    assert d.b.c == 99 and d.x == "y"


def test_augmentations() -> None:
    import numpy as np
    from datasets.augmentation import build_augmentation
    aug = build_augmentation({
        "random_rotation": {"axis": "z"},
        "random_scaling": {"scale_range": [0.9, 1.1]},
        "random_jitter": {"sigma": 0.01},
        "height_appending": True,
    })
    np.random.seed(0)
    sample = {"points": np.random.randn(128, 3).astype(np.float32),
              "features": None, "labels": 0}
    out = aug(sample)
    assert out["points"].shape == (128, 3)
    assert out["features"].shape == (128, 1)  # height appended


def test_metrics_confusion_matrix() -> None:
    import numpy as np
    from evaluation.metrics import ConfusionMatrix
    cm = ConfusionMatrix(num_classes=3)
    cm.update(np.array([0, 1, 2]), np.array([0, 1, 2]))
    assert abs(cm.overall_accuracy() - 1.0) < 1e-6
    assert abs(cm.mean_iou() - 1.0) < 1e-6


def test_metrics_part_segmentation() -> None:
    import numpy as np
    from evaluation.metrics import part_segmentation_metrics
    preds = [np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])]
    labels = [np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])]
    r = part_segmentation_metrics(preds, labels, [0, 1], {0: [0, 1], 1: [0, 1]})
    assert 0.0 <= r["instance_miou"] <= 1.0


def test_dataset_classification() -> None:
    from datasets import build_dataloader
    loader, _ = build_dataloader(
        {"name": "toy_cls", "num_samples": 8, "num_points": 128, "batch_size": 2,
         "augmentation": {"height_appending": True}}, split="train")
    batch = next(iter(loader))
    assert batch["xyz"].shape == (2, 128, 3)
    assert batch["features"].shape[-1] == 1
    assert batch["label"].shape == (2,)


def test_dataset_segmentation() -> None:
    from datasets import build_dataloader
    loader, _ = build_dataloader(
        {"name": "toy_seg", "num_samples": 4, "num_points": 256, "batch_size": 2,
         "augmentation": {"height_appending": True, "color_dropping": {"p": 0.5}}},
        split="train")
    batch = next(iter(loader))
    assert batch["xyz"].shape == (2, 256, 3)
    assert batch["features"].shape[-1] == 4  # 3 color + 1 height
    assert batch["label"].shape == (2, 256)


def test_pointnet2_cls_forward() -> None:
    import torch
    from models import build_model
    m = build_model({"name": "pointnet2_cls", "in_channels": 3, "num_classes": 4})
    m.eval()
    out = m(torch.randn(2, 512, 3), torch.randn(2, 512, 3))
    assert out.shape == (2, 4)


def test_pointnet2_seg_forward() -> None:
    import torch
    from models import build_model
    m = build_model({"name": "pointnet2_seg", "in_channels": 3, "num_classes": 5})
    m.eval()
    out = m(torch.randn(2, 512, 3), torch.randn(2, 512, 3))
    assert out.shape == (2, 512, 5)


def test_pointnet2_partseg_forward() -> None:
    import torch
    from models import build_model
    m = build_model({"name": "pointnet2_partseg", "in_channels": 3,
                     "num_parts": 4, "num_obj_classes": 3})
    m.eval()
    out = m(torch.randn(2, 512, 3), torch.randn(2, 512, 3),
            torch.tensor([0, 1]))
    assert out.shape == (2, 512, 4)


def test_pointnext_cls_all_presets() -> None:
    import torch
    from models import build_model
    for preset in ["s", "b"]:
        m = build_model({"name": "pointnext_cls", "in_channels": 3,
                         "num_classes": 4, "preset": preset})
        m.eval()
        out = m(torch.randn(2, 512, 3), torch.randn(2, 512, 3))
        assert out.shape == (2, 4), f"preset {preset} shape {out.shape}"


def test_pointnext_seg_all_presets() -> None:
    import torch
    from models import build_model
    for preset in ["s", "b", "l"]:
        m = build_model({"name": "pointnext_seg", "in_channels": 3,
                         "num_classes": 5, "preset": preset})
        m.eval()
        out = m(torch.randn(2, 1024, 3), torch.randn(2, 1024, 3))
        assert out.shape == (2, 1024, 5), f"preset {preset} shape {out.shape}"


def test_pointnext_extract_embedding() -> None:
    import torch
    from models import build_model
    m = build_model({"name": "pointnext_cls", "in_channels": 3,
                     "num_classes": 4, "preset": "s"})
    m.eval()
    emb = m.extract_embedding(torch.randn(2, 512, 3), torch.randn(2, 512, 3))
    # For preset s: width=32, encoder ends at 32 * 2^4 = 512 channels
    assert emb.shape[0] == 2 and emb.ndim == 2


def test_training_one_epoch() -> None:
    """Run one epoch of training on toy data to make sure all pieces wire up."""
    import torch
    from datasets import build_dataloader
    from models import build_model
    from training.trainer import Trainer

    cfg = {
        "model": {"name": "pointnext_cls", "in_channels": 1, "num_classes": 4, "preset": "s"},
        "dataset": {"name": "toy_cls", "num_samples": 8, "num_points": 128, "batch_size": 4,
                    "augmentation": {"height_appending": True},
                    "val_transform": {"height_appending": True}},
        "training": {"num_epochs": 1},
        "criterion": {"name": "cross_entropy", "label_smoothing": 0.1},
        "optimizer": {"name": "adamw", "lr": 1e-3},
        "scheduler": {"name": "cosine"},
    }
    model = build_model(dict(cfg["model"]))
    tl, _ = build_dataloader(cfg["dataset"], "train")
    vl, _ = build_dataloader(cfg["dataset"], "val")
    t = Trainer(model, tl, vl, cfg, task="cls", device="cpu",
                output_dir="/tmp/runs/test_training")
    best = t.fit()
    assert "overall_accuracy" in best


def test_unsupervised_pipeline() -> None:
    """Extract embeddings + KMeans + metrics on toy data."""
    import numpy as np
    from datasets import build_dataloader
    from models import build_model
    from unsupervised.embedding import extract_embeddings
    from unsupervised.cluster import cluster_embeddings, evaluate_clustering

    loader, _ = build_dataloader(
        {"name": "toy_cls", "num_samples": 16, "num_points": 128, "batch_size": 4,
         "augmentation": {"height_appending": True},
         "val_transform": {"height_appending": True}}, split="val")
    model = build_model({"name": "pointnext_cls", "in_channels": 1,
                         "num_classes": 4, "preset": "s"})
    emb, lbl = extract_embeddings(model, loader, device="cpu")
    assert emb.shape[0] == 16 and emb.ndim == 2
    preds = cluster_embeddings(emb, method="kmeans", n_clusters=4)
    m = evaluate_clustering(preds, ground_truth=lbl, embeddings=emb)
    assert "ari" in m and "nmi" in m


def test_visualization() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    from visualization.plot import plot_pred_vs_gt
    pts = np.random.randn(256, 3).astype(np.float32)
    preds = np.random.randint(0, 3, size=256)
    gt = np.random.randint(0, 3, size=256)
    out_path = "/tmp/test_viz.png"
    plot_pred_vs_gt(pts, preds, gt, num_classes=3, save_path=out_path)
    assert Path(out_path).exists()


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
TESTS: List[Callable[[], None]] = [
    test_config_loader,
    test_augmentations,
    test_metrics_confusion_matrix,
    test_metrics_part_segmentation,
    test_dataset_classification,
    test_dataset_segmentation,
    test_pointnet2_cls_forward,
    test_pointnet2_seg_forward,
    test_pointnet2_partseg_forward,
    test_pointnext_cls_all_presets,
    test_pointnext_seg_all_presets,
    test_pointnext_extract_embedding,
    test_training_one_epoch,
    test_unsupervised_pipeline,
    test_visualization,
]


def run_all() -> Tuple[int, int]:
    passed, failed = 0, 0
    for t in TESTS:
        name = t.__name__
        try:
            t()
            print(f"  ✓ {name}")
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print()
    print(f"Summary: {passed} passed, {failed} failed, {len(TESTS)} total")
    return passed, failed


if __name__ == "__main__":
    p, f = run_all()
    sys.exit(0 if f == 0 else 1)
