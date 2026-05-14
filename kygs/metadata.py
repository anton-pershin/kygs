from __future__ import annotations

import datetime
from typing import Callable


class MetadataFieldCollision(Exception):
    """Raised when a metadata field already exists and would be overwritten."""


def merge_concat(old_value: list, new_value: list) -> list:
    """Concatenate two lists."""
    return old_value + new_value


def merge_union(old_value: list, new_value: list) -> list:
    """Compute set union of two lists, return as list."""
    return list(set(old_value) | set(new_value))


def merge_replace(_old_value, new_value):
    """Replace old value with new value."""
    return new_value


def merge_min(old_value, new_value):
    """Take minimum of two values."""
    return min(old_value, new_value)


def merge_max(old_value, new_value):
    """Take maximum of two values."""
    return max(old_value, new_value)


MERGE_STRATEGIES: dict[str, Callable] = {
    "concat": merge_concat,
    "union": merge_union,
    "replace": merge_replace,
    "min": merge_min,
    "max": merge_max,
}


class Metadata(dict):
    _merge_strategies: dict[str, str] = {}

    def merge(self, other: Metadata) -> Metadata:
        merged = dict(self)
        for key, value in other.items():
            if key not in merged:
                merged[key] = value
            else:
                strategy_name = self._merge_strategies.get(key, "replace")
                strategy_fn = MERGE_STRATEGIES.get(strategy_name, merge_replace)
                merged[key] = strategy_fn(merged[key], value)
        result = type(self).__new__(type(self))
        dict.__init__(result, merged)
        for key, value in merged.items():
            setattr(result, key, value)
        return result

    def to_json_dict(self) -> dict[str, str]:
        """Convert metadata to a JSON-serializable dict with string values."""
        return {key: str(value) for key, value in self.items()}


class TimeMetadata(Metadata):
    _merge_strategies = {
        "start_dt": "min",
        "end_dt": "max",
    }

    def __init__(self, start_dt: datetime.datetime, end_dt: datetime.datetime) -> None:
        super().__init__(start_dt=start_dt, end_dt=end_dt)
        self.start_dt = start_dt
        self.end_dt = end_dt

    def to_json_dict(self) -> dict[str, str]:
        """Convert TimeMetadata to JSON-serializable dict with formatted datetimes."""
        return {
            "start_dt": self.start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "end_dt": self.end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        }


class LabelMetadata(Metadata):
    _merge_strategies = {
        "labels": "concat",
    }

    def __init__(self, labels: list[str]) -> None:
        super().__init__(labels=labels)
        self.labels = labels


def merge_metadatas(metadatas: list[Metadata]) -> Metadata:
    merged = metadatas[0]
    for m in metadatas[1:]:
        merged = merged.merge(m)
    return merged
