from abc import ABC, abstractmethod
from dataclasses import dataclass

from kygs.message_provider import Message


class MessageAnnotation(ABC):
    @abstractmethod
    def __call__(self, messages: list[Message], labels: dict[str, str]) -> list[str]:
        """Annotate a list of messages with provided labels.

        Args:
            messages: List of messages to annotate
            labels: Dict valid labels to choose from (key is label, value is description)

        Returns:
            List of assigned labels
        """
        pass
