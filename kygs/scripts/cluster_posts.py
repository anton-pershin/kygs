from omegaconf import DictConfig
import hydra
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json

from kygs.message_provider import MessageProvider
from kygs.text_embedding import TextEmbeddingModel
from kygs.clustering import TextClustering
from kygs.utils.report import CsvReport
from kygs.utils.common import get_config_path
from kygs.utils.console import console


CONFIG_NAME = "config_cluster_posts"


def save_clustering_json(
    text_sequences: list[str],
    cluster_labels: list[str],
    true_labels: list[str],
    output_path: str
) -> None:
    # Group messages by cluster
    clusters_dict = {}
    unclustered = []
    
    for text, cluster_label, true_label in zip(text_sequences, cluster_labels, true_labels):
        if cluster_label not in clusters_dict:
            clusters_dict[cluster_label] = []
        clusters_dict[cluster_label].append({
            "text": text,
            "true_label": true_label
        })
    
    # Separate single-message clusters into unclustered
    unclustered_labels = [label for label, msgs in clusters_dict.items() if len(msgs) == 1]
    for label in unclustered_labels:
        unclustered.extend(clusters_dict[label])
        del clusters_dict[label]
    
    # Prepare clusters data
    clusters_data = []
    for label, messages in clusters_dict.items():
        lengths = [len(msg["text"]) for msg in messages]
        clusters_data.append({
            "cluster_label": label,
            "n_messages": len(messages),
            "mean_length": int(round(np.mean(lengths))),
            "q10_length": int(round(np.percentile(lengths, 10))),
            "q90_length": int(round(np.percentile(lengths, 90))),
            "messages": messages
        })
    
    # Sort clusters by size (largest first)
    clusters_data.sort(key=lambda x: x["n_messages"], reverse=True)
    
    # Create final structure, ensuring all numbers are Python native types
    output_data = {
        "n_clusters": int(len(clusters_data)),
        "clusters": [{
            "cluster_label": str(c["cluster_label"]),  # labels are strings
            "n_messages": int(c["n_messages"]),
            "mean_length": int(c["mean_length"]),
            "q10_length": int(c["q10_length"]),
            "q90_length": int(c["q90_length"]),
            "messages": c["messages"]
        } for c in clusters_data],
        "unclustered_messages": unclustered
    }
    
    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


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
    
    # Save results as JSON
    save_clustering_json(
        text_sequences=text_sequences,
        cluster_labels=cluster_labels,
        true_labels=true_labels,
        output_path=cfg.output.clustering_json_path
    )
    
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
