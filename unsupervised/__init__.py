"""Unsupervised extension (NOT part of the PointNeXt paper).

The paper is fully supervised. This module provides a lightweight unsupervised
pipeline on top of a (trained or untrained) PointNeXt encoder:

    1. Extract a global embedding per object from the encoder
       (max-pool of the last encoder stage features).
    2. Cluster those embeddings using KMeans or DBSCAN.
    3. Optionally project to 2D for visualization (PCA / t-SNE / UMAP).
    4. If ground-truth labels are available, report clustering metrics
       (ARI, NMI, silhouette) — used only for evaluation, never for training.

This is exposed as a standalone script too: `scripts/unsupervised_cluster.py`.
"""
from .embedding import extract_embeddings
from .cluster import cluster_embeddings, project_embeddings, evaluate_clustering

__all__ = [
    "extract_embeddings",
    "cluster_embeddings",
    "project_embeddings",
    "evaluate_clustering",
]
