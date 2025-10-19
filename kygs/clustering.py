from typing import Protocol, Optional, Tuple, Any
from pathlib import Path
import time
from abc import ABC, abstractmethod

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.spatial.distance import cosine

from kygs.text_embedding import TextEmbeddingModel
from kygs.utils.report import CsvReport
from kygs.utils.console import console
from kygs.utils.typing import NDArrayInt, NDArrayFloat


class HasText(Protocol):
    text: str


class ClusterCollection(ABC):
    @property
    @abstractmethod
    def n_clusters(self):
        raise NotImplementedError()
    
    @abstractmethod
    def get_centroid(self, i: int) -> NDArrayFloat:
        raise NotImplementedError()

    @abstractmethod
    def get_size(self, i: int) -> int:
        raise NotImplementedError()

    @abstractmethod
    def add(self, objs: list[HasText], centroid: NDArrayFloat) -> None:
        raise NotImplementedError()
        
    @abstractmethod
    def update(self, i: int, objs_to_add: list[HasText], new_centroid: NDArrayFloat) -> None:
        raise NotImplementedError()
        

class ClusterListCollection(ClusterCollection):
    def __init__(self):
        self.centroids: list[NDArrayFloat] = []
        self.sizes: list[int] = []
        self.objects: list[list[Any]] = []

    @property
    def n_clusters(self):
        return len(self.centroids)
    
    def get_centroid(self, i: int) -> NDArrayFloat:
        return self.centroids[i]

    def get_size(self, i: int) -> int:
        return self.sizes[i]

    def add(self, objs: list[HasText], centroid: NDArrayFloat) -> None:
        self.centroids.append(centroid)
        self.sizes.append(len(objs))
        self.objects.append(objs)
        
    def update(self, i: int, objs_to_add: list[HasText], new_centroid: NDArrayFloat) -> None:
        self.centroids[i] = new_centroid
        self.objects[i].extend(objs_to_add)
        self.sizes[i] += len(objs_to_add)


class TextClustering:
    def __init__(
        self,
        text_embedding_model: TextEmbeddingModel,
        distance_threshold: float,
        cluster_collection: ClusterCollection,
    ) -> None:
        self.text_embedding_model = text_embedding_model
        self.distance_threshold = distance_threshold
        self.clustering = AgglomerativeClustering(
            distance_threshold=distance_threshold,
            n_clusters=None,
            metric='cosine',
            linkage='average'
        )
        self.cluster_collection = cluster_collection

    def _compute_centroids(
        self,
        embeddings: NDArrayFloat,
        labels: NDArrayInt
    ) -> Tuple[NDArrayFloat, NDArrayInt, NDArrayInt]:
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
        return np.array(centroids), np.array(sizes), unique_labels

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

    def fit_predict(self, objs: list[HasText]) -> NDArrayInt:
        """Perform initial clustering on texts.
        Returns indices of clusters where -1 means no cluster"""
        texts = [t.text for t in objs]
        embeddings = self.text_embedding_model.predict(texts)
        labels = self.clustering.fit_predict(embeddings)
        # TODO: cluster_sizes is no longer used
        cluster_centroids, cluster_sizes, cluster_labels = self._compute_centroids(
            embeddings, labels
        )
        for centroid, cluster_label in zip(cluster_centroids, cluster_labels):
            cluster_objs = [obj for i, obj in enumerate(objs) if labels[i] == cluster_label]
            self.cluster_collection.add(cluster_objs, centroid)

        return labels

    def update_predict(self, objs: list[HasText]) -> NDArrayInt:
        """Assign texts to nearest clusters if within threshold."""
        texts = [t.text for t in objs]
        if self.cluster_collection.n_clusters == 0:
            if len(objs) == 1:
                embeddings = self.text_embedding_model.predict(texts)
                self.cluster_collection.add(objs, embeddings[0])
                labels = np.array([0], dtype=np.int32)
            elif len(objs) > 1:
                labels = self.fit_predict(objs)
            else:
                raise ValueError("No messages passed to update_predict")

            return labels

        embeddings = self.text_embedding_model.predict(texts)
        labels = np.full(len(texts), -1)

        for i, embedding in enumerate(embeddings):
            # Compute distances to all centroids
            distances_to_clusters = np.array([
                cosine(embedding, self.cluster_collection.get_centroid(i)) for i in range(self.cluster_collection.n_clusters)
            ])
            
            nearest_cluster_idx = np.argmin(distances_to_clusters)
            min_distance = distances_to_clusters[nearest_cluster_idx]
            
            # Assign to cluster if within threshold
            if min_distance <= self.distance_threshold:
                labels[i] = nearest_cluster_idx
                # Update centroid and size for the assigned cluster
                updated_centroid, updated_size = self._update_centroid(
                    self.cluster_collection.get_centroid(nearest_cluster_idx),
                    self.cluster_collection.get_size(nearest_cluster_idx),
                    embedding
                )
                self.cluster_collection.update(
                    nearest_cluster_idx,
                    [objs[i]],
                    updated_centroid,
                )
            else:  # Create a new cluster if not
                self.cluster_collection.add([objs[i]], embedding)

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
