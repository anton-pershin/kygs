import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, Optional, Protocol, Tuple

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from kygs.clustering.base import (ClusterCollection, EmbeddingProvider, HasText,
                                  TextClustering, TextClusteringViaEmbeddings)
from kygs.utils.console import console
from kygs.utils.report import CsvReport
from kygs.utils.typing import NDArrayFloat, NDArrayInt

Linkage = Literal["average", "complete", "single", "ward"]


class FullEmbeddingClusterCollection(ClusterCollection):
    @abstractmethod
    def get_embeddings(self, i: int) -> list[NDArrayFloat]:
        raise NotImplementedError()


class FullEmbeddingClusterListCollection(ClusterCollection):
    def __init__(self) -> None:
        self.sizes: list[int] = []
        self.objects: list[list[Any]] = []
        self.embeddings: list[list[NDArrayFloat]] = []

    @property
    def n_clusters(self) -> int:
        return len(self.sizes)

    def get_embeddings(self, i: int) -> list[NDArrayFloat]:
        return self.embeddings[i]

    def get_size(self, i: int) -> int:
        return self.sizes[i]

    def add(self, objs: list[HasText], embeddings: list[NDArrayFloat]) -> int:
        self.sizes.append(len(objs))
        self.objects.append(objs)
        self.embeddings.append(embeddings)
        return self.n_clusters - 1

    def update(
        self, i: int, objs_to_add: list[HasText], embeddings: list[NDArrayFloat]
    ) -> None:
        self.objects[i].extend(objs_to_add)
        self.sizes[i] += len(objs_to_add)
        self.embeddings[i].extend(embeddings)


class HacBasedTextClustering(
    TextClusteringViaEmbeddings[FullEmbeddingClusterCollection]
):
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        distance_threshold: float,
        cluster_collection: FullEmbeddingClusterCollection,
        linkage: Linkage = "average",
    ) -> None:
        self.distance_threshold = distance_threshold
        self.clustering = AgglomerativeClustering(
            distance_threshold=distance_threshold,
            n_clusters=None,
            metric="cosine",
            linkage=linkage,
        )
        self.linkage = linkage
        super().__init__(cluster_collection, embedding_provider)

    def fit_predict(self, objs: list[HasText]) -> NDArrayInt:
        """Perform initial clustering on texts via HAC.
        Returns indices of clusters where -1 means no cluster"""

        if not objs:
            return np.array([], dtype=np.int32)

        embeddings = self.embedding_provider(objs)
        labels = self.clustering.fit_predict(embeddings)
        cluster_ids = np.unique(labels)
        for i in cluster_ids:
            cluster_objs = [obj for i, obj in enumerate(objs) if labels[i] == i]
            cluster_embeddings = [
                embedding for i, embedding in enumerate(embeddings) if labels[i] == i
            ]
            self.cluster_collection.add(cluster_objs, cluster_embeddings)
        return labels

    def update_predict(self, objs: list[HasText]) -> NDArrayInt:
        # TODO: need to adapt to true HAC
        """Assign texts to nearest clusters if within threshold."""

        # Address the case where there are no clusters yet
        # (the very beginning in the streaming scenario)
        labels: NDArrayInt
        if self.cluster_collection.n_clusters == 0:
            if len(objs) == 1:
                embeddings = self.embedding_provider(objs)
                self.cluster_collection.add(objs, [embeddings[0]])
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
        """Assign texts to nearest clusters assuming that the latter do exist.
        We first compute the distances from a given text embedding to all
        the clusters using the average linkage and then choose the nearest one.
        according to this metric.
        """
        embeddings = self.embedding_provider(objs)
        labels = np.full(len(objs), -1)

        for i, embedding in enumerate(embeddings):
            # Compute distances from embedding to each cluster
            distances_to_clusters = np.array(
                [
                    self._compute_distance_to_cluster(embedding, cluster_idx)
                    for cluster_idx in range(self.cluster_collection.n_clusters)
                ]
            )

            nearest_cluster_idx = int(np.argmin(distances_to_clusters))
            min_distance = distances_to_clusters[nearest_cluster_idx]

            # Assign to cluster if within threshold
            if min_distance <= self.distance_threshold:
                labels[i] = nearest_cluster_idx
                self.cluster_collection.update(
                    nearest_cluster_idx,
                    [objs[i]],
                    [embedding],
                )
            else:  # Create a new cluster if not
                labels[i] = self.cluster_collection.add([objs[i]], [embedding])

        return labels

    def _compute_distance_to_cluster(
        self,
        embedding: NDArrayFloat,
        cluster_idx: int,
    ) -> float:
        """Compute distance from embedding to a cluster using a given linkage."""
        cluster_member_embeddings = self.cluster_collection.get_embeddings(cluster_idx)
        distances_to_cluster_members = np.array(
            [
                cosine(embedding, cluster_member_embedding)
                for cluster_member_embedding in cluster_member_embeddings
            ]
        )

        if self.linkage == "average":
            return np.mean(distances_to_cluster_members)
        elif self.linkage == "complete":
            return np.max(distances_to_cluster_members)
        elif self.linkage == "single":
            return np.min(distances_to_cluster_members)
        else:
            raise ValueError(f"Unknown linkage: {self.linkage}")
