import ast
import datetime
import json
from dataclasses import dataclass
from typing import Optional, cast
from unittest.mock import patch

import pytest
from rally.llm import Llm

from kygs.message_provider import Message, MessageCollection
from kygs.metadata import Metadata, TimeMetadata, merge_metadatas
from kygs.summarization.base import Summary, to_message_collection
from kygs.summarization.direct import (
    DirectSummarization,
    OnlyMessageSummarizationPrompt,
    PlainSummaryBuilder,
    TimeBasedSummarizationPrompt,
)
from kygs.summarization.recursive import (
    OutOfContextLengthException,
    RecursiveSummarization,
    _partition_collection,
)


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
    model: Optional[str] = "test-model"
    model_family: Optional[str] = "test_family"
    max_output_tokens: Optional[int] = None


def _identity(x: str) -> str:
    return x


MOCK_THINKING_REMOVERS = {"test_family": _identity}


class TestSummarizePosts:
    @patch("kygs.summarization.direct.THINKING_REMOVERS", MOCK_THINKING_REMOVERS)
    @patch("kygs.summarization.direct.request_based_on_prompts")
    def test_time_based_splits_produce_summary_with_time_metadata(self, mock_request):
        mock_request.return_value = ["Cats and dogs were discussed."]
        llm = cast(Llm, MockLlm())
        splits = _make_message_splits_with_time_metadata()
        summarization_prompt = TimeBasedSummarizationPrompt(
            system_prompt="You are a summarizer.",
            user_prompt_template="Summarize: {messages_as_json}",
        )
        summarization = DirectSummarization(
            llm=llm,
            summarization_prompt=summarization_prompt,
            summary_builder=PlainSummaryBuilder(),
        )

        result = summarization(message_collections=splits)

        assert len(result) == 1
        assert result[0].text == "Cats and dogs were discussed."
        assert isinstance(result[0].metadata, TimeMetadata)
        assert result[0].metadata.start_dt == START_DT
        assert result[0].metadata.end_dt == END_DT

    @patch("kygs.summarization.direct.THINKING_REMOVERS", MOCK_THINKING_REMOVERS)
    @patch("kygs.summarization.direct.request_based_on_prompts")
    def test_custom_metadata_splits_produce_summary_with_same_metadata(
        self, mock_request
    ):
        mock_request.return_value = ["Animals were discussed."]
        llm = cast(Llm, MockLlm())
        splits = _make_message_splits_with_custom_metadata()
        summarization_prompt = TimeBasedSummarizationPrompt(
            system_prompt="You are a summarizer.",
            user_prompt_template="Summarize: {messages_as_json}",
        )
        summarization = DirectSummarization(
            llm=llm,
            summarization_prompt=summarization_prompt,
            summary_builder=PlainSummaryBuilder(),
        )

        result = summarization(message_collections=splits)

        assert len(result) == 1
        assert result[0].text == "Animals were discussed."
        assert isinstance(result[0].metadata, Metadata)
        assert result[0].metadata["topic"] == "animals"
        assert "start_dt" not in result[0].metadata

    @patch("kygs.summarization.direct.THINKING_REMOVERS", MOCK_THINKING_REMOVERS)
    @patch("kygs.summarization.direct.request_based_on_prompts")
    def test_empty_splits_are_skipped(self, mock_request):
        mock_request.return_value = ["Summary text."]
        llm = cast(Llm, MockLlm())
        splits = [
            _make_empty_message_split(),
            _make_message_splits_with_time_metadata()[0],
        ]
        summarization_prompt = TimeBasedSummarizationPrompt(
            system_prompt="You are a summarizer.",
            user_prompt_template="Summarize: {messages_as_json}",
        )
        summarization = DirectSummarization(
            llm=llm,
            summarization_prompt=summarization_prompt,
            summary_builder=PlainSummaryBuilder(),
        )

        result = summarization(message_collections=splits)

        assert len(result) == 1
        assert mock_request.call_count == 1

    @patch("kygs.summarization.direct.THINKING_REMOVERS", MOCK_THINKING_REMOVERS)
    @patch("kygs.summarization.direct.request_based_on_prompts")
    def test_user_prompt_contains_formatted_messages(self, mock_request):
        mock_request.return_value = ["Summary."]
        llm = cast(Llm, MockLlm())
        splits = _make_message_splits_with_time_metadata()
        summarization_prompt = TimeBasedSummarizationPrompt(
            system_prompt="sys",
            user_prompt_template="Posts: {messages_as_json}",
        )
        summarization = DirectSummarization(
            llm=llm,
            summarization_prompt=summarization_prompt,
            summary_builder=PlainSummaryBuilder(),
        )

        summarization(message_collections=splits)

        user_prompts = mock_request.call_args.kwargs["user_prompts"]
        assert "Posts: " in user_prompts[0]
        json_part = user_prompts[0].replace("Posts: ", "")
        parsed = ast.literal_eval(json_part)
        assert len(parsed) == 2
        assert parsed[0]["message"] == "post about cats"

    @patch("kygs.summarization.direct.THINKING_REMOVERS", MOCK_THINKING_REMOVERS)
    @patch("kygs.summarization.direct.request_based_on_prompts")
    def test_multiple_splits(self, mock_request):
        mock_request.return_value = ["Summary 1.", "Summary 2."]
        llm = cast(Llm, MockLlm())
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
        summarization_prompt = TimeBasedSummarizationPrompt(
            system_prompt="sys",
            user_prompt_template="Summarize: {messages_as_json}",
        )
        summarization = DirectSummarization(
            llm=llm,
            summarization_prompt=summarization_prompt,
            summary_builder=PlainSummaryBuilder(),
        )

        result = summarization(message_collections=splits)

        assert len(result) == 2
        assert result[0].text == "Summary 1."
        assert result[1].text == "Summary 2."


class TestSummarizeSummaries:
    @patch("kygs.summarization.direct.THINKING_REMOVERS", MOCK_THINKING_REMOVERS)
    @patch("kygs.summarization.direct.request_based_on_prompts")
    def test_with_time_based_summaries(self, mock_request):
        mock_request.return_value = ["Overall summary."]
        llm = cast(Llm, MockLlm())
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
        message_collection = to_message_collection(summaries)
        original_prompt = OnlyMessageSummarizationPrompt(
            system_prompt="sys",
            user_prompt_template="Summarize: {messages_as_json}",
        )
        partial_prompt = OnlyMessageSummarizationPrompt(
            system_prompt="sys",
            user_prompt_template="Summarize: {messages_as_json}",
        )
        summarization = RecursiveSummarization(
            llm=llm,
            original_message_summarization_prompt=original_prompt,
            partial_summary_summarization_prompt=partial_prompt,
            original_message_summary_builder=PlainSummaryBuilder(),
            partial_summary_builder=PlainSummaryBuilder(),
            max_characters_in_prompt=10000,
        )

        result = summarization(message_collections=[message_collection])

        assert len(result) == 1
        assert result[0].text == "Overall summary."
        assert isinstance(result[0].metadata, TimeMetadata)
        assert result[0].metadata.start_dt == START_DT
        assert result[0].metadata.end_dt == datetime.datetime(2025, 1, 15, 12, 0, 0)

    @patch("kygs.summarization.direct.THINKING_REMOVERS", MOCK_THINKING_REMOVERS)
    @patch("kygs.summarization.direct.request_based_on_prompts")
    def test_with_custom_metadata_summaries(self, mock_request):
        mock_request.return_value = ["Overall summary."]
        llm = cast(Llm, MockLlm())
        summaries = [
            Summary(text="Summary about cats.", metadata=Metadata(topic="cats")),
            Summary(text="Summary about dogs.", metadata=Metadata(topic="dogs")),
        ]
        message_collection = to_message_collection(summaries)
        original_prompt = OnlyMessageSummarizationPrompt(
            system_prompt="sys",
            user_prompt_template="Summarize: {messages_as_json}",
        )
        partial_prompt = OnlyMessageSummarizationPrompt(
            system_prompt="sys",
            user_prompt_template="Summarize: {messages_as_json}",
        )
        summarization = RecursiveSummarization(
            llm=llm,
            original_message_summarization_prompt=original_prompt,
            partial_summary_summarization_prompt=partial_prompt,
            original_message_summary_builder=PlainSummaryBuilder(),
            partial_summary_builder=PlainSummaryBuilder(),
            max_characters_in_prompt=10000,
        )

        result = summarization(message_collections=[message_collection])

        assert len(result) == 1
        assert result[0].metadata["topic"] == "cats"

    @patch("kygs.summarization.direct.THINKING_REMOVERS", MOCK_THINKING_REMOVERS)
    @patch("kygs.summarization.direct.request_based_on_prompts")
    def test_splits_by_max_characters(self, mock_request):
        mock_request.return_value = ["Summary."]
        llm = cast(Llm, MockLlm())
        summaries = [
            Summary(
                text="A" * 40, metadata=TimeMetadata(start_dt=START_DT, end_dt=END_DT)
            ),
            Summary(
                text="B" * 40,
                metadata=TimeMetadata(
                    start_dt=datetime.datetime(2025, 1, 15, 11, 0, 0),
                    end_dt=datetime.datetime(2025, 1, 15, 12, 0, 0),
                ),
            ),
        ]
        message_collection = to_message_collection(summaries)
        original_prompt = OnlyMessageSummarizationPrompt(
            system_prompt="sys",
            user_prompt_template="Summarize: {messages_as_json}",
        )
        partial_prompt = OnlyMessageSummarizationPrompt(
            system_prompt="sys",
            user_prompt_template="Summarize: {messages_as_json}",
        )
        summarization = RecursiveSummarization(
            llm=llm,
            original_message_summarization_prompt=original_prompt,
            partial_summary_summarization_prompt=partial_prompt,
            original_message_summary_builder=PlainSummaryBuilder(),
            partial_summary_builder=PlainSummaryBuilder(),
            max_characters_in_prompt=100,
        )

        result = summarization(message_collections=[message_collection])

        assert len(result) == 1
        assert mock_request.call_count == 1


class TestBaseSummarizationPrompt:
    def test_time_based_prompt_includes_time_and_message(self):
        message = _make_message(
            "Cats are great.", datetime.datetime(2025, 1, 15, 10, 0, 0)
        )
        mc = MessageCollection(
            messages=[message],
            metadata=TimeMetadata(start_dt=START_DT, end_dt=END_DT),
        )
        prompt = TimeBasedSummarizationPrompt(
            system_prompt="sys",
            user_prompt_template="Summarize: {messages_as_json}",
        )
        result = prompt(mc)
        json_part = result.replace("Summarize: ", "")
        parsed = ast.literal_eval(json_part)
        assert len(parsed) == 1
        assert "time" in parsed[0]
        assert "message" in parsed[0]
        assert parsed[0]["message"] == "Cats are great."
        assert parsed[0]["time"] == "2025-01-15 10:00:00"

    def test_only_message_prompt_omits_time(self):
        message = _make_message(
            "Cats are great.", datetime.datetime(2025, 1, 15, 10, 0, 0)
        )
        mc = MessageCollection(
            messages=[message],
            metadata=TimeMetadata(start_dt=START_DT, end_dt=END_DT),
        )
        prompt = OnlyMessageSummarizationPrompt(
            system_prompt="sys",
            user_prompt_template="Summarize: {messages_as_json}",
        )
        result = prompt(mc)
        json_part = result.replace("Summarize: ", "")
        parsed = ast.literal_eval(json_part)
        assert len(parsed) == 1
        assert "time" not in parsed[0]
        assert "message" in parsed[0]
        assert parsed[0]["message"] == "Cats are great."

    def test_custom_metadata_not_included_in_prompt(self):
        message = _make_message(
            "Cats are great.", datetime.datetime(2025, 1, 15, 10, 0, 0)
        )
        mc = MessageCollection(
            messages=[message],
            metadata=Metadata(topic="animals"),
        )
        prompt = TimeBasedSummarizationPrompt(
            system_prompt="sys",
            user_prompt_template="Summarize: {messages_as_json}",
        )
        result = prompt(mc)
        json_part = result.replace("Summarize: ", "")
        parsed = ast.literal_eval(json_part)
        assert "topic" not in parsed[0]
        assert parsed[0]["message"] == "Cats are great."


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
        merged = merge_metadatas([s.metadata for s in summaries])
        assert isinstance(merged, TimeMetadata)
        assert merged.start_dt == START_DT
        assert merged.end_dt == datetime.datetime(2025, 1, 15, 12, 0, 0)

    def test_merge_generic_metadatas(self):
        summaries = [
            Summary(text="a", metadata=Metadata(topic="cats")),
            Summary(text="b", metadata=Metadata(topic="dogs", count=5)),
        ]
        merged = merge_metadatas([s.metadata for s in summaries])
        assert merged["topic"] == "cats"
        assert merged["count"] == 5

    def test_merge_single_summary(self):
        summaries = [
            Summary(text="a", metadata=TimeMetadata(start_dt=START_DT, end_dt=END_DT)),
        ]
        merged = merge_metadatas([s.metadata for s in summaries])
        assert isinstance(merged, TimeMetadata)
        assert merged.start_dt == START_DT


class TestSummary:
    def test_metadata_stored(self):
        metadata = TimeMetadata(start_dt=START_DT, end_dt=END_DT)
        summary = Summary(text="test", metadata=metadata)
        assert summary.text == "test"
        assert isinstance(summary.metadata, TimeMetadata)
        # pylint: disable=no-member
        assert summary.metadata.start_dt == START_DT
        assert summary.metadata.end_dt == END_DT
        # pylint: enable=no-member

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
        json_str = json.dumps(json_dict)
        assert isinstance(json_str, str)


def _make_mc(texts: list[str], metadata: Metadata | None = None) -> MessageCollection:
    messages = [
        _make_message(t, datetime.datetime(2025, 1, 15, 10, i, 0))
        for i, t in enumerate(texts)
    ]
    if metadata is None:
        metadata = TimeMetadata(start_dt=START_DT, end_dt=END_DT)
    return MessageCollection(messages=messages, metadata=metadata)


class TestPartitionCollection:
    def test_single_message_fits(self):
        mc = _make_mc(["hello"])
        result = _partition_collection(mc, max_chars_in_prompt=10)
        assert len(result) == 1
        assert [m.text for m in result[0].messages] == ["hello"]

    def test_single_message_exactly_fills_limit(self):
        mc = _make_mc(["a" * 10])
        result = _partition_collection(mc, max_chars_in_prompt=10)
        assert len(result) == 1
        assert [m.text for m in result[0].messages] == ["a" * 10]

    def test_single_message_exceeds_limit(self):
        mc = _make_mc(["a" * 11])
        with pytest.raises(OutOfContextLengthException):
            _partition_collection(mc, max_chars_in_prompt=10)

    def test_all_messages_fit_in_one_partition(self):
        mc = _make_mc(["aaa", "bbb", "ccc"])
        result = _partition_collection(mc, max_chars_in_prompt=10)
        assert len(result) == 1
        assert [m.text for m in result[0].messages] == ["aaa", "bbb", "ccc"]

    def test_messages_split_into_two_partitions(self):
        mc = _make_mc(["aaaa", "bbbb", "cc"])
        result = _partition_collection(mc, max_chars_in_prompt=8)
        assert len(result) == 2
        assert [m.text for m in result[0].messages] == ["aaaa", "bbbb"]
        assert [m.text for m in result[1].messages] == ["cc"]

    def test_exact_boundary_messages_fit_together(self):
        mc = _make_mc(["aaaaa", "bbbbb"])
        result = _partition_collection(mc, max_chars_in_prompt=10)
        assert len(result) == 1
        assert [m.text for m in result[0].messages] == ["aaaaa", "bbbbb"]

    def test_exact_boundary_exceeds_by_one(self):
        mc = _make_mc(["aaaaa", "bbbbbb"])
        result = _partition_collection(mc, max_chars_in_prompt=10)
        assert len(result) == 2
        assert [m.text for m in result[0].messages] == ["aaaaa"]
        assert [m.text for m in result[1].messages] == ["bbbbbb"]

    def test_each_message_exactly_fills_one_partition(self):
        mc = _make_mc(["aaaaa", "bbbbb", "ccccc"])
        result = _partition_collection(mc, max_chars_in_prompt=5)
        assert len(result) == 3
        for i, text in enumerate(["aaaaa", "bbbbb", "ccccc"]):
            assert [m.text for m in result[i].messages] == [text]

    def test_first_message_after_split_exactly_fills_limit(self):
        mc = _make_mc(["aaa", "aaaaa", "bb"])
        result = _partition_collection(mc, max_chars_in_prompt=5)
        assert len(result) == 3
        assert [m.text for m in result[0].messages] == ["aaa"]
        assert [m.text for m in result[1].messages] == ["aaaaa"]
        assert [m.text for m in result[2].messages] == ["bb"]

    def test_empty_messages_returns_empty_list(self):
        mc = _make_mc([], metadata=TimeMetadata(start_dt=START_DT, end_dt=END_DT))
        result = _partition_collection(mc, max_chars_in_prompt=10)
        assert not result

    def test_metadata_propagated_to_all_partitions(self):
        metadata = Metadata(topic="test")
        mc = _make_mc(["aaa", "bbbb", "cc"], metadata=metadata)
        result = _partition_collection(mc, max_chars_in_prompt=5)
        assert len(result) == 3
        for partition in result:
            assert partition.metadata == metadata

    def test_multiple_splits(self):
        mc = _make_mc(["aaaa", "aaaa", "aaaa", "aaaa"])
        result = _partition_collection(mc, max_chars_in_prompt=8)
        assert len(result) == 2
        assert [m.text for m in result[0].messages] == ["aaaa", "aaaa"]
        assert [m.text for m in result[1].messages] == ["aaaa", "aaaa"]

    def test_zero_length_message(self):
        mc = _make_mc(["", "aaa", ""])
        result = _partition_collection(mc, max_chars_in_prompt=5)
        assert len(result) == 1
        assert [m.text for m in result[0].messages] == ["", "aaa", ""]
