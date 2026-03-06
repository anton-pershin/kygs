from __future__ import annotations

import json
from pathlib import Path

import hydra
import hydra.utils
from omegaconf import DictConfig

from kygs.message_provider import Message, MessageProvider
from kygs.relevance.base import RelevanceScorer
from kygs.utils.common import get_config_path, set_cuda_visible_devices

CONFIG_NAME = "config_score_relevance"


def _build_score_payload(
    queries: list[Message],
    documents: list[Message],
    scores,
    top_k: int | None,
) -> dict[str, Any]:
    query_ids = list(range(len(queries)))

    result = {"queries": []}
    max_k = len(documents) if top_k is None else min(top_k, len(documents))

    for q_idx in query_ids:
        row = scores[q_idx]
        if row.size == 0 or len(documents) == 0:
            result["queries"].append(
                {
                    "query_id": q_idx,
                    "document_ids": [],
                    "scores": [],
                }
            )
            continue

        sorted_doc_indices = row.argsort()[::-1]
        limited_doc_indices = sorted_doc_indices[:max_k]
        result["queries"].append(
            {
                "query_id": q_idx,
                "document_ids": limited_doc_indices.tolist(),
                "scores": [float(row[i]) for i in limited_doc_indices],
            }
        )

    return result


def save_scores_to_json(
    output_path: Path,
    queries: list[Message],
    documents: list[Message],
    scores,
    top_k: int | None,
) -> None:
    payload = _build_score_payload(queries, documents, scores, top_k)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def score_relevance(cfg: DictConfig) -> None:
    set_cuda_visible_devices(cfg.cuda_visible_devices)

    query_mp: MessageProvider = hydra.utils.call(cfg.query_message_provider)
    document_mp: MessageProvider = hydra.utils.call(cfg.document_message_provider)

    scoring_strategy = hydra.utils.instantiate(cfg.scoring_strategy)
    scorer = RelevanceScorer(strategy=scoring_strategy)

    queries = list(query_mp.messages)
    documents = list(document_mp.messages)

    score_matrix = scorer.score(queries, documents)

    output_path = Path(cfg.result_dir) / cfg.output.scores_json
    save_scores_to_json(output_path, queries, documents, score_matrix.scores, cfg.top_k)


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(score_relevance)()
