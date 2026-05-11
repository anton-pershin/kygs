import json
from abc import ABC, abstractmethod
from typing import Any

from kygs.split_strategy import TimeMetadata
from kygs.summarization.base import Summary
from kygs.utils.report import CsvReport


class SummaryHandler(ABC):
    @abstractmethod
    def handle(self, summaries: list[Summary], **kwargs: Any) -> None:
        """Handle a list of summaries.

        Args:
            summaries: List of summaries to handle
            **kwargs: Additional arguments specific to concrete handlers
        """


class SummaryJsonSaver(SummaryHandler):
    def __init__(self, output_path: str) -> None:
        self.output_path = output_path

    def handle(self, summaries: list[Summary], **kwargs: Any) -> None:
        records: list[dict[str, Any]] = []
        for s in summaries:
            record: dict[str, Any] = {"summary": s.text}
            if isinstance(s.metadata, TimeMetadata):
                record["start_time"] = s.metadata.start_dt.strftime("%Y-%m-%d %H:%M:%S")
                record["end_time"] = s.metadata.end_dt.strftime("%Y-%m-%d %H:%M:%S")
            for k, v in s.metadata.items():
                if k not in ("start_dt", "end_dt"):
                    record[k] = v
            records.append(record)

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)


class SummaryCsvSaver(SummaryHandler):
    def __init__(self, output_path: str) -> None:
        self.output_path = output_path

    def handle(self, summaries: list[Summary], **kwargs: Any) -> None:
        start_time: list[str] = []
        end_time: list[str] = []
        for s in summaries:
            if isinstance(s.metadata, TimeMetadata):
                start_time.append(s.metadata.start_dt.strftime("%Y-%m-%d %H:%M:%S"))
                end_time.append(s.metadata.end_dt.strftime("%Y-%m-%d %H:%M:%S"))
            else:
                start_time.append("")
                end_time.append("")

        report = CsvReport(self.output_path)
        report.add_columns(
            summary=[s.text for s in summaries],
            start_time=start_time,
            end_time=end_time,
        )
        report.dump()
