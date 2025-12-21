from abc import ABC, abstractmethod
from typing import Generic, Optional, Protocol, TypeVar

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from kygs.utils.console import console
from kygs.utils.report import CsvReport
from kygs.utils.typing import NDArrayFloat, NDArrayInt


class HasText(Protocol):
    text: str


class ClusterCollection(ABC):
    @property
    @abstractmethod
    def n_clusters(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def get_size(self, i: int) -> int:
        raise NotImplementedError()

    @abstractmethod
    def add(self, objs: list[HasText], *args) -> int:
        raise NotImplementedError()

    @abstractmethod
    def update(self, i: int, objs_to_add: list[HasText], *args) -> None:
        raise NotImplementedError()


ClusterCollectionT = TypeVar("ClusterCollectionT", bound=ClusterCollection)


class TextClustering(Generic[ClusterCollectionT], ABC):
    def __init__(
        self,
        cluster_collection: ClusterCollectionT,
    ) -> None:
        self.cluster_collection = cluster_collection

    def fit_predict(self, objs: list[HasText]) -> NDArrayInt:
        """Perform initial clustering on texts.
        Returns indices of clusters where -1 means no cluster"""
        raise NotImplementedError()

    def update_predict(self, objs: list[HasText]) -> NDArrayInt:
        """Assign texts to nearest clusters.
        Returns indices of cluster where -1 means no cluster"""
        raise NotImplementedError()

    @staticmethod
    def print_clustering_report(
        title: str,
        y_true: list[Optional[str | int]],
        y_pred: NDArrayInt,
        time_spent: float,
        metrics_path: str | None = None,
        verbose: bool = False,
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
            "adjusted_rand_index": adjusted_rand_score(
                y_true_filtered, y_pred_filtered
            ),
            "normalized_mutual_information": normalized_mutual_info_score(
                y_true_filtered, y_pred_filtered
            ),
            "num_ground_truth_clusters": len(set(y_true_filtered)),
            "num_predicted_clusters": len(set(y_pred_filtered)),
            "num_evaluated_messages": len(y_true_filtered),
            "total_messages": len(y_true),
            "time_spent": time_spent,
        }

        if verbose:
            # Print metrics
            console.print(f"\n[bold]{title.upper()}:[/bold]")
            console.print(f"Adjusted Rand Index: {metrics['adjusted_rand_index']:.3f}")
            console.print(
                f"Normalized Mutual Information: {metrics['normalized_mutual_information']:.3f}"
            )
            console.print(
                f"Number of ground truth clusters: {metrics['num_ground_truth_clusters']}"
            )
            console.print(
                f"Number of predicted clusters: {metrics['num_predicted_clusters']}"
            )
            console.print(
                f"Number of evaluated messages: {metrics['num_evaluated_messages']} (out of {metrics['total_messages']} total)"
            )

        # Save metrics
        if metrics_path is not None:
            metrics_report = CsvReport(metrics_path)
            metrics_report.add_record(**metrics)
            metrics_report.dump()
