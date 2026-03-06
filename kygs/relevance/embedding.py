from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.linalg import norm

from kygs.clustering.base import EmbeddingProvider
from kygs.message_provider import Message
from kygs.utils.typing import NDArrayFloat

from .base import RelevanceScoreMatrix, RelevanceScoringStrategy


class EmbeddingSimilarityScoringStrategy(RelevanceScoringStrategy):
    """Scores relevance using cosine similarity on text embeddings."""

    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self.embedding_provider = embedding_provider

    def _normalize(self, embeddings: NDArrayFloat) -> NDArrayFloat:
        if embeddings.size == 0:
            return embeddings

        norms = norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def score(
        self,
        queries: Sequence[Message],
        documents: Sequence[Message],
    ) -> RelevanceScoreMatrix:
        if not queries or not documents:
            score_matrix = np.zeros((len(queries), len(documents)), dtype=np.float32)
            return RelevanceScoreMatrix(list(queries), list(documents), score_matrix)

        query_embeddings = self.embedding_provider(list(queries))
        doc_embeddings = self.embedding_provider(list(documents))

        query_embeddings = self._normalize(query_embeddings)
        doc_embeddings = self._normalize(doc_embeddings)

        similarity_matrix = np.matmul(query_embeddings, doc_embeddings.T)
        return RelevanceScoreMatrix(
            queries=list(queries),
            documents=list(documents),
            scores=similarity_matrix.astype(np.float32),
        )
