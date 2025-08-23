from typing import Optional
from pathlib import Path
import time

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from kygs.text_embedding import TextEmbeddingModel
from kygs.utils.report import CsvReport
from kygs.utils.console import console


class TextClustering:
    def __init__(
        self,
        text_embedding_model: TextEmbeddingModel,
        distance_threshold: float,
    ) -> None:
        self.text_embedding_model = text_embedding_model
        self.clustering = AgglomerativeClustering(
            distance_threshold=distance_threshold,
            n_clusters=None,
            metric='cosine',
            linkage='average'
        )
        
    def fit(self, texts: list[str]) -> None:
        return None
        
    def predict(self, texts: list[str]) -> np.ndarray:
        embeddings = self.text_embedding_model.predict(texts)
        return self.clustering.fit_predict(embeddings)  # HAC doesn't have separate predict

    def print_clustering_report(
        self,
        title: str,
        X: list[str],
        y_true: list[Optional[str]],
        metrics_path: str,
        verbose: bool = False
    ) -> None:
        # Measure clustering time and get predictions
        start_time = time.time()
        y_pred = self.predict(X)
        time_spent = time.time() - start_time

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
            console.print(f"\n[bold]{title.upper()} CLUSTERING METRICS:[/bold]")
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
