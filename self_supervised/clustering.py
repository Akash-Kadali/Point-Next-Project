import numpy as np
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def run_clustering(embeddings, method="kmeans", n_clusters=4):
    if method == "kmeans":
        pred = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(embeddings)

    elif method == "dbscan":
        pred = DBSCAN(eps=0.5, min_samples=5).fit_predict(embeddings)

    elif method == "spectral":
        pred = SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            random_state=42,
        ).fit_predict(embeddings)

    else:
        raise ValueError(f"Unknown clustering method: {method}")

    return pred


def evaluate_clustering(embeddings, true_labels, cluster_labels):
    results = {}

    if true_labels is not None:
        results["ARI"] = adjusted_rand_score(true_labels, cluster_labels)
        results["NMI"] = normalized_mutual_info_score(true_labels, cluster_labels)

    valid = cluster_labels != -1
    unique_clusters = set(cluster_labels[valid])

    if valid.sum() > 2 and len(unique_clusters) > 1:
        results["Silhouette"] = silhouette_score(
            embeddings[valid],
            cluster_labels[valid],
        )
    else:
        results["Silhouette"] = float("nan")

    results["num_clusters"] = len(unique_clusters)
    results["num_noise"] = int((cluster_labels == -1).sum())

    return results