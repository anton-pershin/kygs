from typing import Optional, Tuple
from pathlib import Path
import time

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.spatial.distance import cosine

from kygs.text_embedding import TextEmbeddingModel
from kygs.utils.report import CsvReport
from kygs.utils.console import console
from kygs.utils.typing import NDArrayInt, NDArrayFloat


class TextClustering:
    def __init__(
        self,
        text_embedding_model: TextEmbeddingModel,
        distance_threshold: float,
    ) -> None:
        self.text_embedding_model = text_embedding_model
        self.distance_threshold = distance_threshold
        self.clustering = AgglomerativeClustering(
            distance_threshold=distance_threshold,
            n_clusters=None,
            metric='cosine',
            linkage='average'
        )
        self.cluster_centroids = None
        self.cluster_sizes = None

    def _compute_centroids(
        self,
        embeddings: NDArrayFloat,
        labels: NDArrayInt
    ) -> Tuple[NDArrayFloat, NDArrayInt]:
        """Compute initial centroids and sizes for each cluster."""
        unique_labels = np.unique(labels)
        centroids = []
        sizes = []
        for label in unique_labels:
            mask = labels == label
            centroid = embeddings[mask].mean(axis=0)
            size = np.sum(mask)
            centroids.append(centroid)
            sizes.append(size)
        return np.array(centroids), np.array(sizes)

    def _update_centroid(
        self,
        centroid: NDArrayFloat,
        size: int,
        new_embedding: NDArrayFloat
    ) -> Tuple[NDArrayFloat, int]:
        """Update centroid using online mean computation."""
        new_size = size + 1
        updated_centroid = centroid + (new_embedding - centroid) / new_size
        return updated_centroid, new_size

    def fit_predict(self, texts: list[str]) -> NDArrayInt:
        """Perform initial clustering on texts.
        Returns indices of clusters where -1 means no cluster"""
        embeddings = self.text_embedding_model.predict(texts)
        labels = self.clustering.fit_predict(embeddings)
        self.cluster_centroids, self.cluster_sizes = self._compute_centroids(embeddings, labels)
        return labels

    def update_predict(self, texts: list[str]) -> NDArrayInt:
        """Assign texts to nearest clusters if within threshold."""
        if self.cluster_centroids is None:
            return np.array([-1] * len(texts))

        embeddings = self.text_embedding_model.predict(texts)
        labels = np.full(len(texts), -1)

        for i, embedding in enumerate(embeddings):
            # Compute distances to all centroids
            distances = np.array([
                cosine(embedding, centroid) for centroid in self.cluster_centroids
            ])
            
            nearest_idx = np.argmin(distances)
            min_distance = distances[nearest_idx]
            
            # Assign to cluster if within threshold
            if min_distance <= self.distance_threshold:
                labels[i] = nearest_idx
                # Update centroid and size for the assigned cluster
                self.cluster_centroids[nearest_idx], self.cluster_sizes[nearest_idx] = \
                    self._update_centroid(
                        self.cluster_centroids[nearest_idx],
                        self.cluster_sizes[nearest_idx],
                        embedding
                    )

        return labels

    @staticmethod
    def print_clustering_report(
        title: str,
        y_true: list[Optional[str | int]],
        y_pred: NDArrayInt,
        time_spent: float,
        metrics_path: str,
        verbose: bool = False
    ) -> None:
        # Filter out messages without ground truth labels (noise)
        valid_indices = [i for i, label in enumerate(y_true) if label is not None]
        
        if not valid_indices:
            if verbose:
                console.print("[yellow]No ground truth labels found for evaluation")
            return
            
        y_true_filtered = np.array([y_true[i] for i in valid_indices])
        y_pred_filtered = y_pred[valid_indices]
        
        # Calculate metrics
        metrics = {
            "adjusted_rand_index": adjusted_rand_score(y_true_filtered, y_pred_filtered),
            "normalized_mutual_information": normalized_mutual_info_score(y_true_filtered, y_pred_filtered),
            "num_ground_truth_clusters": len(set(y_true_filtered)),
            "num_predicted_clusters": len(set(y_pred_filtered)),
            "num_evaluated_messages": len(y_true_filtered),
            "total_messages": len(y_true),
            "time_spent": time_spent
        }

        if verbose:
            # Print metrics
            console.print(f"\n[bold]{title.upper()}:[/bold]")
            console.print(f"Adjusted Rand Index: {metrics['adjusted_rand_index']:.3f}")
            console.print(f"Normalized Mutual Information: {metrics['normalized_mutual_information']:.3f}")
            console.print(f"Number of ground truth clusters: {metrics['num_ground_truth_clusters']}")
            console.print(f"Number of predicted clusters: {metrics['num_predicted_clusters']}")
            console.print(f"Number of evaluated messages: {metrics['num_evaluated_messages']} (out of {metrics['total_messages']} total)")

        # Save metrics
        metrics_report = CsvReport(metrics_path)
        metrics_report.add_record(**metrics)
        metrics_report.dump()

        return y_pred
