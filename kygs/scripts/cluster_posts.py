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
from kygs.clustering import TextClustering, ClusterListCollection
from kygs.message_handler import MessageJsonSaver, MessageCsvSaver
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
        cluster_collection=ClusterListCollection(),
    )

    # Get text sequences and perform clustering
    text_sequences = [msg.text for msg in mp.messages]
    true_labels = [msg.true_label for msg in mp.messages]
    
    start_time = time.time()
    pred_labels = clustering.fit_predict(mp.messages)
    time_spent = time.time() - start_time

    # Save evaluation metrics
    cluster_labels = clustering.print_clustering_report(
        title="Clustering Evaluation",
        y_true=true_labels,
        y_pred=pred_labels,
        time_spent=time_spent,
        metrics_path=cfg.output.summary_path,
        verbose=True
    )

    # Update message labels
    for msg, label in zip(mp.messages, pred_labels):
        msg.label = label

    # Save results using configured message handlers
    for handler_cfg in cfg.message_handlers:
        handler = hydra.utils.instantiate(handler_cfg)
        handler.handle(mp.messages)
    
    # Save dendrogram
    embeddings = text_embedding_model.predict(text_sequences)
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
