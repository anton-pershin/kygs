from abc import abstractmethod
from typing import Any, Optional, Protocol

import numpy as np
from scipy.spatial.distance import cosine

from kygs.clustering.base import (ClusterCollection, EmbeddingProvider, HasText,
                                  TextClusteringViaEmbeddings)
from kygs.text_embedding import TextEmbeddingModel
from kygs.utils.typing import NDArrayFloat, NDArrayInt


class CentroidBasedClusterCollection(ClusterCollection):
    @abstractmethod
    def get_centroid(self, i: int) -> NDArrayFloat:
        raise NotImplementedError()


class CentroidBasedClusterListCollection(CentroidBasedClusterCollection):
    def __init__(self) -> None:
        self.centroids: list[NDArrayFloat] = []
        self.sizes: list[int] = []
        self.objects: list[list[Any]] = []

    @property
    def n_clusters(self) -> int:
        return len(self.centroids)

    def get_centroid(self, i: int) -> NDArrayFloat:
        return self.centroids[i]

    def get_size(self, i: int) -> int:
        return self.sizes[i]

    def add(self, objs: list[HasText], centroid: NDArrayFloat) -> int:
        self.centroids.append(centroid)
        self.sizes.append(len(objs))
        self.objects.append(objs)
        return self.n_clusters - 1

    def update(
        self, i: int, objs_to_add: list[HasText], new_centroid: NDArrayFloat
    ) -> None:
        self.centroids[i] = new_centroid
        self.objects[i].extend(objs_to_add)
        self.sizes[i] += len(objs_to_add)


class CentroidBasedTextClustering(
    TextClusteringViaEmbeddings[CentroidBasedClusterCollection]
):
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        distance_threshold: float,
        cluster_collection: CentroidBasedClusterCollection,
    ) -> None:
        self.distance_threshold = distance_threshold
        super().__init__(cluster_collection, embedding_provider)

    def _compute_centroids(
        self, embeddings: NDArrayFloat, labels: NDArrayInt
    ) -> tuple[NDArrayFloat, NDArrayInt, NDArrayInt]:
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
        self, centroid: NDArrayFloat, size: int, new_embedding: NDArrayFloat
    ) -> tuple[NDArrayFloat, int]:
        """Update centroid using online mean computation."""
        new_size = size + 1
        updated_centroid = centroid + (new_embedding - centroid) / new_size
        return updated_centroid, new_size

    def fit_predict(self, objs: list[HasText]) -> NDArrayInt:
        """Perform initial clustering on texts.
        Returns indices of clusters where -1 means no cluster"""

        if not objs:
            return np.array([], dtype=np.int32)

        # Create the first cluster
        first_obj_label = self.update_predict(objs[:1])

        # Run through others
        other_obj_labels = self.update_predict(objs[1:])
        return np.concatenate((first_obj_label, other_obj_labels))

    def update_predict(self, objs: list[HasText]) -> NDArrayInt:
        """Assign texts to nearest clusters if within threshold."""

        # Address the case where there are no clusters yet
        # (the very beginning in the streaming scenario)
        labels: NDArrayInt
        if self.cluster_collection.n_clusters == 0:
            if len(objs) == 1:
                embeddings = self.embedding_provider(objs)
                self.cluster_collection.add(objs, embeddings[0])
                labels = np.array([0], dtype=np.int32)
            elif len(objs) > 1:
                labels = self.fit_predict(objs)
            else:
                raise ValueError("No messages passed to update_predict")

            return labels

        # Address the case where we already have a set of clusters
        return self._assign_objs_to_existing_clusters_or_create_new_one(objs)

    def _assign_objs_to_existing_clusters_or_create_new_one(
        self, objs: list[HasText]
    ) -> NDArrayInt:
        """Assign texts to nearest clusters assuming that the latter do exist."""
        embeddings = self.embedding_provider(objs)
        labels = np.full(len(objs), -1)

        for i, embedding in enumerate(embeddings):
            # Compute distances to all centroids
            distances_to_clusters = np.array(
                [
                    cosine(embedding, self.cluster_collection.get_centroid(i))
                    for i in range(self.cluster_collection.n_clusters)
                ]
            )

            nearest_cluster_idx = int(np.argmin(distances_to_clusters))
            min_distance = distances_to_clusters[nearest_cluster_idx]

            # Assign to cluster if within threshold
            if min_distance <= self.distance_threshold:
                labels[i] = nearest_cluster_idx
                # Update centroid and size for the assigned cluster
                updated_centroid, updated_size = self._update_centroid(
                    self.cluster_collection.get_centroid(nearest_cluster_idx),
                    self.cluster_collection.get_size(nearest_cluster_idx),
                    embedding,
                )
                self.cluster_collection.update(
                    nearest_cluster_idx,
                    [objs[i]],
                    updated_centroid,
                )
            else:  # Create a new cluster if not
                labels[i] = self.cluster_collection.add([objs[i]], embedding)

        return labels
