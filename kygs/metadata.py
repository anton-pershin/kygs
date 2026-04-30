from __future__ import annotations

import datetime


class Metadata(dict):
    def merge(self, other: Metadata) -> Metadata:
        merged = dict(self)
        for key, value in other.items():
            if key not in merged:
                merged[key] = value
        return Metadata(merged)

    def to_json_dict(self) -> dict[str, str]:
        """Convert metadata to a JSON-serializable dict with string values."""
        return {key: str(value) for key, value in self.items()}


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

    def to_json_dict(self) -> dict[str, str]:
        """Convert TimeMetadata to JSON-serializable dict with formatted datetimes."""
        return {
            "start_dt": self.start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "end_dt": self.end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        }


class LabelMetadata(Metadata):
    def __init__(self, labels: list[str]) -> None:
        super().__init__(labels=labels)
        self.labels = labels 

    def merge(self, other: Metadata) -> Metadata:
        if isinstance(other, LabelMetadata):
            return LabelMetadata(
                labels=self.labels + other.labels,
            )
        return super().merge(other)
