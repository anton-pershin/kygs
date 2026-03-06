from __future__ import annotations

import json
from pathlib import Path

import hydra
import hydra.utils
from omegaconf import DictConfig

from kygs.message_provider import MessageProvider
from kygs.utils.common import get_config_path
from kygs.utils.console import console

CONFIG_NAME = "config_print_relevant_messages"


def _ensure_scores_file(scores_path: Path) -> dict:
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores file not found: {scores_path}")
    with scores_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def print_relevant_messages(cfg: DictConfig) -> None:
    query_mp: MessageProvider = hydra.utils.call(cfg.query_message_provider)
    document_mp: MessageProvider = hydra.utils.call(cfg.document_message_provider)

    scores_path = Path(cfg.scores_path)
    scores_data = _ensure_scores_file(scores_path)

    doc_lookup = document_mp.messages
    query_lookup = query_mp.messages

    for query_scores in scores_data.get("queries", []):
        query_id = query_scores.get("query_id")
        query = query_lookup[query_id]
        query_text = query.text if query else f"[Unknown query {query_id}]"
        console.print(f"\n[bold]Query:[/bold] {query_text}")

        doc_ids = query_scores.get("document_ids", [])
        scores = query_scores.get("scores", [])
        if not doc_ids:
            console.print("  No relevant documents found")
            continue

        for rank, (doc_id, score) in enumerate(zip(doc_ids, scores), start=1):
            doc = doc_lookup[doc_id]
            doc_text = doc.text if doc else f"[Unknown document {doc_id}]"
            console.print(f"  {rank}. Score: {score:.4f} -> {doc_text}")


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(print_relevant_messages)()
