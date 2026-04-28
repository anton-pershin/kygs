from __future__ import annotations

import datetime
from abc import ABC, abstractmethod
from typing import Any

from kygs.message_provider import Message, MessageCollection
from kygs.metadata import Metadata
from kygs.utils.time import datetime_ceil, datetime_floor, increment_datetime
from kygs.utils.typing import TimeUnit


class TimeMetadata(Metadata):
    def __init__(self, start_dt: datetime.datetime, end_dt: datetime.datetime) -> None:
        super().__init__(start_dt=start_dt, end_dt=end_dt)
        self.start_dt = start_dt
        self.end_dt = end_dt

    def merge(self, other: Metadata) -> Metadata:
        if isinstance(other, TimeMetadata):
            return TimeMetadata(
                start_dt=min(self.start_dt, other.start_dt),
                end_dt=max(self.end_dt, other.end_dt),
            )
        return super().merge(other)


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
