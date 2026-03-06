from __future__ import annotations

import numpy as np
import pytest

from kygs.relevance.base import (
    RelevanceScoreMatrix,
    RelevanceScorer,
    RelevanceScoringStrategy,
)
from kygs.relevance.embedding import EmbeddingSimilarityScoringStrategy

from tests.test_cluster_posts import (
    MockEmbedding,
    create_message_dict_provider,
)


@pytest.fixture()
def message_provider():
    """Create a MessageProvider populated with the shared synthetic messages."""

    return create_message_dict_provider()


class TextEqualityScoringStrategy(RelevanceScoringStrategy):
    """Scores 1.0 when texts match exactly, otherwise 0.0."""

    def score(self, queries, documents) -> RelevanceScoreMatrix:  # type: ignore[override]
        queries_list = list(queries)
        documents_list = list(documents)
        scores = np.zeros((len(queries_list), len(documents_list)), dtype=np.float32)

        doc_index_by_text = {doc.text: idx for idx, doc in enumerate(documents_list)}
        for q_idx, query in enumerate(queries_list):
            doc_idx = doc_index_by_text.get(query.text)
            if doc_idx is not None:
                scores[q_idx, doc_idx] = 1.0

        return RelevanceScoreMatrix(queries_list, documents_list, scores)


@pytest.fixture()
def embedding_provider():
    mock_embedding = MockEmbedding()

    def provider(objs):
        texts = [obj.text for obj in objs]
        return mock_embedding.predict(texts)

    return provider


@pytest.mark.parametrize(
    "strategy_factory",
    [
        lambda provider: EmbeddingSimilarityScoringStrategy(provider),
        lambda _: TextEqualityScoringStrategy(),
    ],
)
def test_queries_rank_themselves_highest(message_provider, strategy_factory, embedding_provider):
    queries = message_provider.messages
    documents = message_provider.messages
    strategy = strategy_factory(embedding_provider)
    scores = RelevanceScorer(strategy).score(queries, documents)

    for idx, query in enumerate(scores.queries):
        row = scores.scores[idx]
        assert row.size > 0
        max_score = row.max()
        assert np.isclose(row[idx], max_score)
        assert max_score >= 0
