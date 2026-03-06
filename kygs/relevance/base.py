from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from kygs.message_provider import Message
from kygs.utils.typing import NDArrayFloat


@dataclass
class RelevanceScoreMatrix:
    """Container for pairwise relevance scores."""

    queries: list[Message]
    documents: list[Message]
    scores: NDArrayFloat

    def top_k(self, k: int) -> list[list[tuple[int, float]]]:
        """Return top-k document indices and scores per query."""

        if k <= 0:
            return [[] for _ in self.queries]

        k = min(k, len(self.documents))
        top_matches: list[list[tuple[int, float]]] = []
        for row in self.scores:
            if row.size == 0:
                top_matches.append([])
                continue

            top_indices = np.argsort(row)[-k:][::-1]
            top_matches.append([(int(idx), float(row[idx])) for idx in top_indices])
        return top_matches


class RelevanceScoringStrategy(ABC):
    @abstractmethod
    def score(
        self,
        queries: Sequence[Message],
        documents: Sequence[Message],
    ) -> RelevanceScoreMatrix:
        """Compute relevance scores for all query/document pairs."""
        raise NotImplementedError


class RelevanceScorer:
    def __init__(self, strategy: RelevanceScoringStrategy) -> None:
        self.strategy = strategy

    def score(
        self,
        queries: Sequence[Message],
        documents: Sequence[Message],
    ) -> RelevanceScoreMatrix:
        return self.strategy.score(queries, documents)
