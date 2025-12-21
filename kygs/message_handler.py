import json
from abc import ABC, abstractmethod
from typing import TypedDict

import numpy as np

from kygs.message_provider import Message
from kygs.utils.report import CsvReport


class MessageHandler(ABC):
    @abstractmethod
    def handle(self, messages: list[Message], **kwargs) -> None:
        """Handle a list of messages.

        Args:
            messages: List of messages to handle
            **kwargs: Additional arguments specific to concrete handlers
        """


class MessageJsonSaver(MessageHandler):
    def __init__(self, output_path: str):
        self.output_path = output_path

    def handle(self, messages: list[Message], **kwargs) -> None:
        MessageData = TypedDict(  # pylint: disable=invalid-name
            "MessageData", {"text": str, "true_label": str | None}
        )

        # Group messages by label
        clusters_dict: dict[str | None, list[MessageData]] = {}
        unclustered = []

        for msg in messages:
            label = msg.label
            if label not in clusters_dict:
                clusters_dict[label] = []
            clusters_dict[label].append(
                {"text": msg.text, "true_label": msg.true_label}
            )

        # Separate single-message clusters into unclustered
        unclustered_labels = [
            label
            for label, msgs in clusters_dict.items()
            if len(msgs) == 1 or label is None
        ]
        for label in unclustered_labels:
            unclustered.extend(clusters_dict[label])
            del clusters_dict[label]

        # Prepare clusters data
        clusters_data: list[dict[str, str | int | list[MessageData]]] = []
        for label, cluster_messages in clusters_dict.items():
            lengths = [len(msg["text"]) for msg in cluster_messages]
            clusters_data.append(
                {
                    "cluster_label": str(label),
                    "n_messages": len(cluster_messages),
                    "mean_length": int(round(np.mean(lengths))),
                    "q10_length": int(round(np.percentile(lengths, 10))),
                    "q90_length": int(round(np.percentile(lengths, 90))),
                    "messages": cluster_messages,
                }
            )

        # Sort clusters by size
        clusters_data.sort(key=lambda x: x["n_messages"], reverse=True)

        # Create final structure
        output_data = {
            "n_clusters": len(clusters_data),
            "clusters": clusters_data,
            "unclustered_messages": unclustered,
        }

        # Save to JSON
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)


class MessageCsvSaver(MessageHandler):
    def __init__(self, output_path: str):
        self.output_path = output_path

    def handle(self, messages: list[Message], **kwargs) -> None:
        report = CsvReport(self.output_path)
        report.add_columns(
            message=[msg.text for msg in messages],
            label=[msg.label for msg in messages],
        )
        report.dump()
