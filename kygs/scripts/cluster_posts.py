from omegaconf import DictConfig
import hydra
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from pathlib import Path
import time

from kygs.message_provider import MessageProvider
from kygs.text_embedding import TextEmbeddingModel
from kygs.clustering import TextClustering
from kygs.utils.report import CsvReport
from kygs.utils.common import get_config_path
from kygs.utils.console import console


CONFIG_NAME = "config_cluster_posts"


def save_dendrogram(embeddings: np.ndarray, output_path: str) -> None:
    # Compute the linkage matrix
    linkage_matrix = linkage(embeddings, metric='cosine', method='average')
    
    # Create figure
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix)
    
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    
    # Save and close
    plt.savefig(output_path)
    plt.close()


def cluster_posts(cfg: DictConfig) -> None:
    # Load messages
    mp = hydra.utils.call(cfg.message_provider)
    
    # Initialize clustering
    text_embedding_model = hydra.utils.instantiate(cfg.embedding)
    clustering = TextClustering(
        text_embedding_model=text_embedding_model,
        distance_threshold=cfg.clustering.distance_threshold,
    )
    
    # Get text sequences and perform clustering
    text_sequences = [msg.text for msg in mp.messages]
    true_labels = [msg.label for msg in mp.messages]
    
    # Get embeddings for dendrogram
    embeddings = text_embedding_model.predict(text_sequences)
    
    clustering.fit(text_sequences)  # Does nothing now
    
    # Run clustering and get evaluation metrics
    cluster_labels = clustering.print_clustering_report(
        title="Clustering Evaluation",
        X=text_sequences,
        y_true=true_labels,
        metrics_path=cfg.output.summary_path,
        verbose=True
    )

    # Save results using CsvReport
    report = CsvReport(cfg.output.clustered_messages_path)
    report.add_columns(
        message=text_sequences,
        label=cluster_labels
    )
    report.dump()
    
    # Save dendrogram
    output_dir = Path(cfg.output.clustered_messages_path).parent
    save_dendrogram(
        embeddings,
        str(output_dir / "clustering_dendrogram.png")
    )

    console.print(f"Saved {len(text_sequences)} clustered message results to {cfg.result_dir}")

if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(cluster_posts)()
