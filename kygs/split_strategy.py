from __future__ import annotations

import datetime
from abc import ABC, abstractmethod
from typing import Any
from collections import defaultdict

from kygs.message_provider import Message, MessageCollection
from kygs.metadata import Metadata, TimeMetadata, LabelMetadata
from kygs.utils.time import datetime_ceil, datetime_floor, increment_datetime
from kygs.utils.typing import TimeUnit


class SplitStrategy(ABC):
    @abstractmethod
    def split(self, messages: list[Message]) -> list[MessageCollection]:
        ...


class TimeSplitStrategy(SplitStrategy):
    def __init__(self, time_unit: TimeUnit) -> None:
        self.time_unit = time_unit

    def split(self, messages: list[Message]) -> list[MessageCollection]:
        if not messages:
            return []

        
        times = [m.time for m in messages]
        times.sort()
        start_dt = datetime_floor(times[0], self.time_unit)
        end_dt = datetime_ceil(times[-1], self.time_unit)

        split_start_dt = start_dt
        split_end_dt = increment_datetime(start_dt, self.time_unit)

        end_dt = max(end_dt, split_end_dt)

        splits: list[MessageCollection] = []
        i = 0
        while split_start_dt < end_dt:
            split_messages = [
                m for m in messages if split_start_dt <= m.time <= split_end_dt
            ]
            splits.append(
                MessageCollection(
                    messages=split_messages,
                    metadata=TimeMetadata(start_dt=split_start_dt, end_dt=split_end_dt),
                )
            )

            i += 1
            split_start_dt = increment_datetime(start_dt, self.time_unit, amount=i)
            split_end_dt = increment_datetime(start_dt, self.time_unit, amount=i + 1)

        return splits


class LabelSplitStrategy(SplitStrategy):
    def split(self, messages: list[Message]) -> list[MessageCollection]:
        if not messages:
            return []
        
        cluster_label_to_messages = defaultdict(list)
        for m in messages:
            cluster_label_to_messages[m.label].append(m)
       
        splits = []
        for cluster_label, messages in cluster_label_to_messages.items():
            splits.append(
                MessageCollection(
                    messages=messages,
                    metadata=LabelMetadata(labels=[cluster_label]),
                )
            )

        return splits


class NoSplitStrategy(SplitStrategy):
    def split(self, messages: list[Message]) -> list[MessageCollection]:
        metadata: Metadata = Metadata()
        if messages:
            times = [m.time for m in messages]
            metadata = TimeMetadata(start_dt=min(times), end_dt=max(times))
        return [MessageCollection(messages=messages, metadata=metadata)]


class PrecomputedSplitStrategy(SplitStrategy):
    def __init__(self, splits: list[MessageCollection]) -> None:
        self._splits = splits

    def split(self, messages: list[Message]) -> list[MessageCollection]:
        return self._splits
