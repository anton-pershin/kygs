import ast
import datetime
import json
from dataclasses import dataclass
from typing import Optional
from unittest.mock import patch

import pytest

from kygs.message_provider import Message, MessageCollection
from kygs.split_strategy import Metadata, TimeMetadata
from kygs.summarization import (SummarizationException, Summary,
                                _compose_user_prompt_for_summary_collection,
                                _merge_summary_metadatas, summarize_posts,
                                summarize_summaries)


def _make_message(text: str, time: datetime.datetime) -> Message:
    return Message(
        text=text, time=time, author="test_author", label=None, true_label=None
    )


START_DT = datetime.datetime(2025, 1, 15, 10, 0, 0)
END_DT = datetime.datetime(2025, 1, 15, 11, 0, 0)


def _make_message_splits_with_time_metadata() -> list[MessageCollection]:
    return [
        MessageCollection(
            messages=[
                _make_message(
                    "post about cats", datetime.datetime(2025, 1, 15, 10, 10, 0)
                ),
                _make_message(
                    "post about dogs", datetime.datetime(2025, 1, 15, 10, 30, 0)
                ),
            ],
            metadata=TimeMetadata(start_dt=START_DT, end_dt=END_DT),
        ),
    ]


def _make_message_splits_with_custom_metadata() -> list[MessageCollection]:
    return [
        MessageCollection(
            messages=[
                _make_message(
                    "post about cats", datetime.datetime(2025, 1, 15, 10, 10, 0)
                ),
            ],
            metadata=Metadata(topic="animals"),
        ),
    ]


def _make_empty_message_split() -> MessageCollection:
    return MessageCollection(
        messages=[], metadata=TimeMetadata(start_dt=START_DT, end_dt=END_DT)
    )


@dataclass
class MockLlm:
    url: str = "http://localhost:8000"
    max_concurrent_requests: int = 1
    authorization: Optional[str] = None
    model: str = "test-model"
    model_family: str = "test_family"


def _identity(x: str) -> str:
    return x


MOCK_THINKING_REMOVERS = {"test_family": _identity}


class TestSummarizePosts:
    @patch("kygs.summarization.THINKING_REMOVERS", MOCK_THINKING_REMOVERS)
    @patch("kygs.summarization.request_based_on_prompts")
    def test_time_based_splits_produce_summary_with_time_metadata(self, mock_request):
        mock_request.return_value = ["Cats and dogs were discussed."]
        llm = MockLlm()
        splits = _make_message_splits_with_time_metadata()

        result = summarize_posts(
            message_splits=splits,
            llm=llm,
            system_prompt="You are a summarizer.",
            user_prompt_template="Summarize: {messages_as_json}",
        )

        assert len(result) == 1
        assert result[0].text == "Cats and dogs were discussed."
        assert isinstance(result[0].metadata, TimeMetadata)
        assert result[0].metadata.start_dt == START_DT
        assert result[0].metadata.end_dt == END_DT

    @patch("kygs.summarization.THINKING_REMOVERS", MOCK_THINKING_REMOVERS)
    @patch("kygs.summarization.request_based_on_prompts")
    def test_custom_metadata_splits_produce_summary_with_same_metadata(
        self, mock_request
    ):
        mock_request.return_value = ["Animals were discussed."]
        llm = MockLlm()
        splits = _make_message_splits_with_custom_metadata()

        result = summarize_posts(
            message_splits=splits,
            llm=llm,
            system_prompt="You are a summarizer.",
            user_prompt_template="Summarize: {messages_as_json}",
        )

        assert len(result) == 1
        assert result[0].text == "Animals were discussed."
        assert isinstance(result[0].metadata, Metadata)
        assert result[0].metadata["topic"] == "animals"
        assert "start_dt" not in result[0].metadata

    @patch("kygs.summarization.THINKING_REMOVERS", MOCK_THINKING_REMOVERS)
    @patch("kygs.summarization.request_based_on_prompts")
    def test_empty_splits_are_skipped(self, mock_request):
        mock_request.return_value = ["Summary text."]
        llm = MockLlm()
        splits = [
            _make_empty_message_split(),
            _make_message_splits_with_time_metadata()[0],
        ]

        result = summarize_posts(
            message_splits=splits,
            llm=llm,
            system_prompt="You are a summarizer.",
            user_prompt_template="Summarize: {messages_as_json}",
        )

        assert len(result) == 1
        assert mock_request.call_count == 1

    @patch("kygs.summarization.THINKING_REMOVERS", MOCK_THINKING_REMOVERS)
    @patch("kygs.summarization.request_based_on_prompts")
    def test_user_prompt_contains_formatted_messages(self, mock_request):
        mock_request.return_value = ["Summary."]
        llm = MockLlm()
        splits = _make_message_splits_with_time_metadata()

        summarize_posts(
            message_splits=splits,
            llm=llm,
            system_prompt="sys",
            user_prompt_template="Posts: {messages_as_json}",
        )

        user_prompts = mock_request.call_args.kwargs["user_prompts"]
        assert "Posts: " in user_prompts[0]
        json_part = user_prompts[0].replace("Posts: ", "")
        parsed = ast.literal_eval(json_part)
        assert len(parsed) == 2
        assert parsed[0]["author"] == "test_author"
        assert parsed[0]["message"] == "post about cats"

    @patch("kygs.summarization.THINKING_REMOVERS", MOCK_THINKING_REMOVERS)
    @patch("kygs.summarization.request_based_on_prompts")
    def test_multiple_splits(self, mock_request):
        mock_request.return_value = ["Summary 1.", "Summary 2."]
        llm = MockLlm()
        splits = [
            MessageCollection(
                messages=[_make_message("a", datetime.datetime(2025, 1, 15, 10, 0, 0))],
                metadata=TimeMetadata(start_dt=START_DT, end_dt=END_DT),
            ),
            MessageCollection(
                messages=[_make_message("b", datetime.datetime(2025, 1, 15, 11, 0, 0))],
                metadata=TimeMetadata(
                    start_dt=datetime.datetime(2025, 1, 15, 11, 0, 0),
                    end_dt=datetime.datetime(2025, 1, 15, 12, 0, 0),
                ),
            ),
        ]

        result = summarize_posts(
            message_splits=splits,
            llm=llm,
            system_prompt="sys",
            user_prompt_template="Summarize: {messages_as_json}",
        )

        assert len(result) == 2
        assert result[0].text == "Summary 1."
        assert result[1].text == "Summary 2."


class TestSummarizeSummaries:
    @patch("kygs.summarization.THINKING_REMOVERS", MOCK_THINKING_REMOVERS)
    @patch("kygs.summarization.request_based_on_prompts")
    def test_with_time_based_summaries(self, mock_request):
        mock_request.return_value = ["Overall summary."]
        llm = MockLlm()
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

        result = summarize_summaries(
            summary_collection=summaries,
            llm=llm,
            system_prompt="sys",
            user_prompt_template="Summarize: {summaries_as_json}",
            max_characters_in_prompt=10000,
        )

        assert len(result) == 1
        assert result[0].text == "Overall summary."
        assert isinstance(result[0].metadata, TimeMetadata)
        assert result[0].metadata.start_dt == START_DT
        assert result[0].metadata.end_dt == datetime.datetime(2025, 1, 15, 12, 0, 0)

    @patch("kygs.summarization.THINKING_REMOVERS", MOCK_THINKING_REMOVERS)
    @patch("kygs.summarization.request_based_on_prompts")
    def test_with_custom_metadata_summaries(self, mock_request):
        mock_request.return_value = ["Overall summary."]
        llm = MockLlm()
        summaries = [
            Summary(text="Summary about cats.", metadata=Metadata(topic="cats")),
            Summary(text="Summary about dogs.", metadata=Metadata(topic="dogs")),
        ]

        result = summarize_summaries(
            summary_collection=summaries,
            llm=llm,
            system_prompt="sys",
            user_prompt_template="Summarize: {summaries_as_json}",
            max_characters_in_prompt=10000,
        )

        assert len(result) == 1
        assert result[0].metadata["topic"] == "cats"

    @patch("kygs.summarization.THINKING_REMOVERS", MOCK_THINKING_REMOVERS)
    @patch("kygs.summarization.request_based_on_prompts")
    def test_splits_by_max_characters(self, mock_request):
        mock_request.return_value = ["Batch 1 summary.", "Batch 2 summary."]
        llm = MockLlm()
        summaries = [
            Summary(
                text="A" * 80, metadata=TimeMetadata(start_dt=START_DT, end_dt=END_DT)
            ),
            Summary(
                text="B" * 80,
                metadata=TimeMetadata(
                    start_dt=datetime.datetime(2025, 1, 15, 11, 0, 0),
                    end_dt=datetime.datetime(2025, 1, 15, 12, 0, 0),
                ),
            ),
        ]

        result = summarize_summaries(
            summary_collection=summaries,
            llm=llm,
            system_prompt="sys",
            user_prompt_template="Summarize: {summaries_as_json}",
            max_characters_in_prompt=100,
        )

        assert len(result) == 2
        user_prompts_arg = mock_request.call_args.kwargs["user_prompts"]
        assert len(user_prompts_arg) == 2


class TestComposeUserPromptForSummaryCollection:
    def test_includes_time_metadata_when_present(self):
        summaries = [
            Summary(
                text="Cats are great.",
                metadata=TimeMetadata(start_dt=START_DT, end_dt=END_DT),
            ),
        ]
        result = _compose_user_prompt_for_summary_collection(
            summary_collection=summaries,
            user_prompt_template="Summarize: {summaries_as_json}",
        )
        json_part = result.replace("Summarize: ", "")
        parsed = ast.literal_eval(json_part)
        assert "start_dt" in parsed[0]
        assert "end_dt" in parsed[0]
        assert parsed[0]["summary"] == "Cats are great."
        # Verify that datetimes are converted to strings
        assert isinstance(parsed[0]["start_dt"], str)
        assert isinstance(parsed[0]["end_dt"], str)
        assert parsed[0]["start_dt"] == "2025-01-15 10:00:00"
        assert parsed[0]["end_dt"] == "2025-01-15 11:00:00"

    def test_omits_time_metadata_when_absent(self):
        summaries = [
            Summary(text="Cats are great.", metadata=Metadata()),
        ]
        result = _compose_user_prompt_for_summary_collection(
            summary_collection=summaries,
            user_prompt_template="Summarize: {summaries_as_json}",
        )
        json_part = result.replace("Summarize: ", "")
        parsed = ast.literal_eval(json_part)
        assert "start_dt" not in parsed[0]
        assert "end_dt" not in parsed[0]
        assert parsed[0]["summary"] == "Cats are great."

    def test_custom_metadata_included(self):
        summaries = [
            Summary(text="Cats are great.", metadata=Metadata(topic="animals")),
        ]
        result = _compose_user_prompt_for_summary_collection(
            summary_collection=summaries,
            user_prompt_template="Summarize: {summaries_as_json}",
        )
        json_part = result.replace("Summarize: ", "")
        parsed = ast.literal_eval(json_part)
        assert parsed[0]["topic"] == "animals"
        assert parsed[0]["summary"] == "Cats are great."


class TestMergeSummaryMetadatas:
    def test_merge_time_metadatas(self):
        summaries = [
            Summary(text="a", metadata=TimeMetadata(start_dt=START_DT, end_dt=END_DT)),
            Summary(
                text="b",
                metadata=TimeMetadata(
                    start_dt=datetime.datetime(2025, 1, 15, 11, 0, 0),
                    end_dt=datetime.datetime(2025, 1, 15, 12, 0, 0),
                ),
            ),
        ]
        merged = _merge_summary_metadatas(summaries)
        assert isinstance(merged, TimeMetadata)
        assert merged.start_dt == START_DT
        assert merged.end_dt == datetime.datetime(2025, 1, 15, 12, 0, 0)

    def test_merge_generic_metadatas(self):
        summaries = [
            Summary(text="a", metadata=Metadata(topic="cats")),
            Summary(text="b", metadata=Metadata(topic="dogs", count=5)),
        ]
        merged = _merge_summary_metadatas(summaries)
        assert merged["topic"] == "cats"
        assert merged["count"] == 5

    def test_merge_single_summary(self):
        summaries = [
            Summary(text="a", metadata=TimeMetadata(start_dt=START_DT, end_dt=END_DT)),
        ]
        merged = _merge_summary_metadatas(summaries)
        assert isinstance(merged, TimeMetadata)
        assert merged.start_dt == START_DT


class TestSummary:
    def test_metadata_stored(self):
        metadata = TimeMetadata(start_dt=START_DT, end_dt=END_DT)
        summary = Summary(text="test", metadata=metadata)
        assert summary.text == "test"
        assert isinstance(summary.metadata, TimeMetadata)
        assert summary.metadata.start_dt == START_DT
        assert summary.metadata.end_dt == END_DT

    def test_generic_metadata(self):
        metadata = Metadata(topic="cats")
        summary = Summary(text="test", metadata=metadata)
        assert summary.metadata["topic"] == "cats"


class TestMetadataToJsonDict:
    def test_generic_metadata_to_json_dict(self):
        metadata = Metadata(topic="animals", count=5)
        json_dict = metadata.to_json_dict()
        assert json_dict == {"topic": "animals", "count": "5"}
        assert all(isinstance(v, str) for v in json_dict.values())

    def test_empty_metadata_to_json_dict(self):
        metadata = Metadata()
        json_dict = metadata.to_json_dict()
        assert json_dict == {}

    def test_time_metadata_to_json_dict(self):
        metadata = TimeMetadata(start_dt=START_DT, end_dt=END_DT)
        json_dict = metadata.to_json_dict()
        assert json_dict == {
            "start_dt": "2025-01-15 10:00:00",
            "end_dt": "2025-01-15 11:00:00",
        }
        assert all(isinstance(v, str) for v in json_dict.values())

    def test_to_json_dict_is_json_serializable(self):
        metadata = TimeMetadata(start_dt=START_DT, end_dt=END_DT)
        json_dict = metadata.to_json_dict()
        # This should not raise an exception
        json_str = json.dumps(json_dict)
        assert isinstance(json_str, str)
