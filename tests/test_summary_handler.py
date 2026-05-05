import datetime
import json
import os
import tempfile

from kygs.split_strategy import Metadata, TimeMetadata
from kygs.summarization.base import Summary
from kygs.summary_handler import SummaryCsvSaver, SummaryJsonSaver


START_DT = datetime.datetime(2025, 1, 15, 10, 0, 0)
END_DT = datetime.datetime(2025, 1, 15, 11, 0, 0)


class TestSummaryHandler:
    def test_csv_saver_with_time_metadata(self) -> None:
        summaries = [
            Summary(
                text="Summary about cats.",
                metadata=TimeMetadata(start_dt=START_DT, end_dt=END_DT),
            ),
            Summary(
                text="Summary about dogs.",
                metadata=TimeMetadata(
                    start_dt=datetime.datetime(2025, 1, 15, 11, 0, 0),
                    end_dt=datetime.datetime(2025, 1, 15, 12, 0, 0),
                ),
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "summaries.csv")
            handler = SummaryCsvSaver(output_path=csv_path)
            handler.handle(summaries)

            assert os.path.exists(csv_path)
            with open(csv_path, encoding="utf-8") as f:
                content = f.read()
            assert "Summary about cats" in content
            assert "Summary about dogs" in content
            assert "2025-01-15 10:00:00" in content
            assert "2025-01-15 12:00:00" in content

    def test_csv_saver_with_generic_metadata(self) -> None:
        summaries = [
            Summary(text="Summary about cats.", metadata=Metadata(topic="cats")),
            Summary(text="Summary about dogs.", metadata=Metadata(topic="dogs")),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "summaries.csv")
            handler = SummaryCsvSaver(output_path=csv_path)
            handler.handle(summaries)

            assert os.path.exists(csv_path)
            with open(csv_path, encoding="utf-8") as f:
                content = f.read()
            assert "Summary about cats" in content
            assert "Summary about dogs" in content

    def test_json_saver_with_time_metadata(self) -> None:
        summaries = [
            Summary(
                text="Summary about cats.",
                metadata=TimeMetadata(start_dt=START_DT, end_dt=END_DT),
            ),
            Summary(
                text="Summary about dogs.",
                metadata=TimeMetadata(
                    start_dt=datetime.datetime(2025, 1, 15, 11, 0, 0),
                    end_dt=datetime.datetime(2025, 1, 15, 12, 0, 0),
                ),
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = os.path.join(tmp_dir, "summaries.json")
            handler = SummaryJsonSaver(output_path=json_path)
            handler.handle(summaries)

            assert os.path.exists(json_path)
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)

            assert len(data) == 2
            assert data[0]["summary"] == "Summary about cats."
            assert data[0]["start_time"] == "2025-01-15 10:00:00"
            assert data[0]["end_time"] == "2025-01-15 11:00:00"
            assert data[1]["summary"] == "Summary about dogs."
            assert data[1]["start_time"] == "2025-01-15 11:00:00"
            assert data[1]["end_time"] == "2025-01-15 12:00:00"

    def test_json_saver_with_generic_metadata(self) -> None:
        summaries = [
            Summary(text="Summary about cats.", metadata=Metadata(topic="cats")),
            Summary(text="Summary about dogs.", metadata=Metadata(topic="dogs")),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = os.path.join(tmp_dir, "summaries.json")
            handler = SummaryJsonSaver(output_path=json_path)
            handler.handle(summaries)

            assert os.path.exists(json_path)
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)

            assert len(data) == 2
            assert data[0]["summary"] == "Summary about cats."
            assert data[0]["topic"] == "cats"
            assert data[1]["summary"] == "Summary about dogs."
            assert data[1]["topic"] == "dogs"

    def test_json_saver_output_is_serializable(self) -> None:
        summaries = [
            Summary(
                text="Summary about cats.",
                metadata=TimeMetadata(start_dt=START_DT, end_dt=END_DT),
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = os.path.join(tmp_dir, "summaries.json")
            handler = SummaryJsonSaver(output_path=json_path)
            handler.handle(summaries)

            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)

            json_str = json.dumps(data)
            assert isinstance(json_str, str)
