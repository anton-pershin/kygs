from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from kygs.message_provider import Message, MessageCollection
from kygs.metadata import Metadata, merge_metadatas


@dataclass
class Summary:
    text: str
    metadata: Metadata


class BaseSummarization(ABC):
    @abstractmethod
    def __call__(self, message_collections: list[MessageCollection]) -> list[Summary]:
        """Summarize a list of collections of messages. Each collection is going to
        have its own summary

        Args:
            message_collections: A list of collections of messages to summarize

        Returns:
            Collection of summaries, one summary per collection
        """


def to_message_collection(summaries: list[Summary]) -> MessageCollection:
    messages = [
        Message(
            text=s.text,
            time=datetime.now(),
            author="summarizer",
            label=None,
            true_label=None,
        )
        for s in summaries
    ]
    metadata = merge_metadatas([s.metadata for s in summaries])
    return MessageCollection(
        messages=messages,
        metadata=metadata,
    )
