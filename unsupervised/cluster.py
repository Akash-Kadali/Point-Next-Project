"""Cluster embeddings, project to 2D, and evaluate clustering quality.

Uses scikit-learn throughout. UMAP is optional (imported lazily).
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def cluster_embeddings(embeddings: np.ndarray,
                       method: str = "kmeans",
                       n_clusters: Optional[int] = None,
                       random_state: int = 42,
                       **kwargs) -> np.ndarray:
    """Cluster rows of `embeddings`. Returns cluster labels [M].

    Args:
        embeddings: [M, C]
        method: 'kmeans' | 'dbscan'
        n_clusters: required for kmeans
        random_state: seed for kmeans init
        **kwargs: passed through to the sklearn estimator
            (e.g. eps, min_samples for DBSCAN)
    """
    method = method.lower()
    if method == "kmeans":
        if n_clusters is None:
            raise ValueError("kmeans requires n_clusters.")
        model = KMeans(n_clusters=n_clusters, random_state=random_state,
                       n_init=10, **kwargs)
        return model.fit_predict(embeddings)
    if method == "dbscan":
        model = DBSCAN(**kwargs)
        return model.fit_predict(embeddings)
    raise ValueError(f"Unknown clustering method: {method}")


def project_embeddings(embeddings: np.ndarray, method: str = "pca",
                       n_components: int = 2,
                       random_state: int = 42,
                       **kwargs) -> np.ndarray:
    """Project embeddings to low-dim (for plotting). Returns [M, n_components]."""
    method = method.lower()
    if method == "pca":
        return PCA(n_components=n_components, random_state=random_state).fit_transform(embeddings)
    if method == "tsne":
        # t-SNE is slow on high-dim; run PCA to 50 first if needed.
        x = embeddings
        if x.shape[1] > 50:
            x = PCA(n_components=50, random_state=random_state).fit_transform(x)
        return TSNE(n_components=n_components, random_state=random_state,
                    init="pca", learning_rate="auto", **kwargs).fit_transform(x)
    if method == "umap":
        try:
            import umap  # noqa: WPS433
        except ImportError as e:
            raise ImportError(
                "UMAP projection requested but umap-learn is not installed. "
                "Run `pip install umap-learn`."
            ) from e
        reducer = umap.UMAP(n_components=n_components, random_state=random_state, **kwargs)
        return reducer.fit_transform(embeddings)
    raise ValueError(f"Unknown projection method: {method}")


def evaluate_clustering(predicted: np.ndarray,
                        ground_truth: Optional[np.ndarray] = None,
                        embeddings: Optional[np.ndarray] = None
                        ) -> Dict[str, float]:
    """Evaluate clustering quality.

    Args:
        predicted: [M] cluster assignments.
        ground_truth: [M] ground-truth labels if available (for ARI/NMI).
        embeddings: [M, C] original vectors (for silhouette).

    Returns:
        dict with any subset of {ari, nmi, silhouette, num_clusters, num_noise}.
    """
    out: Dict[str, float] = {}
    out["num_clusters"] = int(len(set(predicted) - {-1}))
    out["num_noise"] = int((predicted == -1).sum())

    if ground_truth is not None:
        # ARI and NMI work even when labels/clusters differ in count / names
        out["ari"] = float(adjusted_rand_score(ground_truth, predicted))
        out["nmi"] = float(normalized_mutual_info_score(ground_truth, predicted))

    if embeddings is not None:
        # silhouette needs: ≥2 clusters, ≥1 non-noise point, and
        # strictly fewer clusters than non-noise samples.
        mask = predicted != -1
        n_valid = int(mask.sum())
        n_clusters_valid = len(set(predicted[mask].tolist())) if n_valid > 0 else 0
        if n_valid >= 3 and 2 <= n_clusters_valid < n_valid:
            try:
                out["silhouette"] = float(silhouette_score(embeddings[mask],
                                                            predicted[mask]))
            except Exception:
                out["silhouette"] = float("nan")
        else:
            out["silhouette"] = float("nan")

    return out
