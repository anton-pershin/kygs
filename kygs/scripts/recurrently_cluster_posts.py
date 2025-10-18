from typing import Optional
from pathlib import Path
import time

from omegaconf import DictConfig
import hydra
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from rich.progress import track

from kygs.message_provider import MessageProvider
from kygs.text_embedding import TextEmbeddingModel
from kygs.clustering import TextClustering, ClusterListCollection
from kygs.message_handler import MessageJsonSaver
from kygs.utils.common import get_config_path
from kygs.utils.console import console
from kygs.utils.report import CsvReport
from kygs.utils.typing import NDArrayInt


CONFIG_NAME = "config_recurrent_cluster_posts"


def recurrently_cluster_posts(cfg: DictConfig) -> None:
    # Load messages
    initial_mp = hydra.utils.call(cfg.initial_message_provider)
    streaming_mp = hydra.utils.call(cfg.recurrent_message_provider)
    
    # Get text sequences and labels for training
    initial_texts = [msg.text for msg in initial_mp.messages]
    initial_true_labels = [msg.true_label for msg in initial_mp.messages]
    
    # Initialize clustering
    text_embedding_model = hydra.utils.instantiate(cfg.embedding)
    clustering = TextClustering(
        text_embedding_model=text_embedding_model,
        distance_threshold=cfg.clustering.distance_threshold,
        cluster_collection=ClusterListCollection(),
    )
    
    # Initial clustering
    console.print("\nPerforming initial clustering...")
    start_time = time.time()
    initial_pred_labels = clustering.fit_predict(initial_mp.messages)
    time_spent = time.time() - start_time

    # Update labels in initial message provider
    for i, m in enumerate(initial_mp.messages):
        pred_label = initial_pred_labels[i]
        m.label = pred_label if pred_label != -1 else None

    # Save evaluation metrics
    cluster_labels = clustering.print_clustering_report(
        title="Clustering Evaluation",
        y_true=initial_true_labels,
        y_pred=initial_pred_labels,
        time_spent=time_spent,
        metrics_path=cfg.output.summary_path,
        verbose=True
    )

    console.print(f"\nSaved clustering evaluation to {cfg.output.summary_path}")

    # Process streaming messages one by one
    console.print("\nProcessing streaming messages...")
    streaming_pred_labels = []
    
    message_iterator = track(
        enumerate(streaming_mp.messages),
        description="Assigning messages to clusters",
        total=len(streaming_mp.messages),
    )
    clustering.text_embedding_model.verbose = False

    for i, mp in message_iterator:
        labels = clustering.update_predict([mp])
        label = labels[0]
        mp.label = label if label != -1 else None
        
    # Combined initial and recurrent messages to save results
    combined_mp = MessageProvider.from_message_providers(initial_mp, streaming_mp)

    # Save results using configured message handlers
    for handler_cfg in cfg.message_handlers:
        handler = hydra.utils.instantiate(handler_cfg)
        handler.handle(combined_mp.messages)
    
    console.print(f"\nHandled clustered messages according to config")


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(recurrently_cluster_posts)()
