import datetime

import pytest

from kygs.message_provider import Message, MessageCollection
from kygs.split_strategy import (
    Metadata,
    NoSplitStrategy,
    PrecomputedSplitStrategy,
    TimeMetadata,
    TimeSplitStrategy,
)


def _make_message(text: str, time: datetime.datetime) -> Message:
    return Message(text=text, time=time, author="test", label=None, true_label=None)


class TestTimeSplitStrategy:
    def test_single_message_produces_single_split(self):
        msg = _make_message("hello", datetime.datetime(2025, 1, 15, 10, 30, 0))
        strategy = TimeSplitStrategy(time_unit="hour")
        splits = strategy.split([msg])

        assert len(splits) == 1
        assert len(splits[0].messages) == 1
        assert splits[0].messages[0].text == "hello"
        assert isinstance(splits[0].metadata, TimeMetadata)
        assert splits[0].metadata.start_dt == datetime.datetime(2025, 1, 15, 10, 0, 0)
        assert splits[0].metadata.end_dt == datetime.datetime(2025, 1, 15, 11, 0, 0)

    def test_messages_in_same_interval_produce_single_split(self):
        messages = [
            _make_message("a", datetime.datetime(2025, 1, 15, 10, 10, 0)),
            _make_message("b", datetime.datetime(2025, 1, 15, 10, 30, 0)),
            _make_message("c", datetime.datetime(2025, 1, 15, 10, 50, 0)),
        ]
        strategy = TimeSplitStrategy(time_unit="hour")
        splits = strategy.split(messages)

        assert len(splits) == 1
        assert len(splits[0].messages) == 3

    def test_messages_spanning_multiple_intervals(self):
        messages = [
            _make_message("a", datetime.datetime(2025, 1, 15, 10, 10, 0)),
            _make_message("b", datetime.datetime(2025, 1, 15, 11, 10, 0)),
            _make_message("c", datetime.datetime(2025, 1, 15, 12, 30, 0)),
        ]
        strategy = TimeSplitStrategy(time_unit="hour")
        splits = strategy.split(messages)

        assert len(splits) == 3
        assert len(splits[0].messages) == 1
        assert len(splits[1].messages) == 1
        assert len(splits[2].messages) == 1

    def test_metadata_is_time_metadata(self):
        messages = [
            _make_message("a", datetime.datetime(2025, 1, 15, 10, 30, 0)),
            _make_message("b", datetime.datetime(2025, 1, 15, 12, 30, 0)),
        ]
        strategy = TimeSplitStrategy(time_unit="hour")
        splits = strategy.split(messages)

        for split in splits:
            assert isinstance(split.metadata, TimeMetadata)
            assert split.metadata.start_dt < split.metadata.end_dt
            assert "start_dt" in split.metadata
            assert "end_dt" in split.metadata

    def test_day_time_unit(self):
        messages = [
            _make_message("a", datetime.datetime(2025, 1, 15, 10, 0, 0)),
            _make_message("b", datetime.datetime(2025, 1, 16, 10, 0, 0)),
        ]
        strategy = TimeSplitStrategy(time_unit="day")
        splits = strategy.split(messages)

        assert len(splits) == 2
        assert splits[0].metadata.start_dt == datetime.datetime(2025, 1, 15, 0, 0, 0)
        assert splits[0].metadata.end_dt == datetime.datetime(2025, 1, 16, 0, 0, 0)
        assert splits[1].metadata.start_dt == datetime.datetime(2025, 1, 16, 0, 0, 0)
        assert splits[1].metadata.end_dt == datetime.datetime(2025, 1, 17, 0, 0, 0)

    def test_empty_messages_returns_empty_list(self):
        strategy = TimeSplitStrategy(time_unit="hour")
        splits = strategy.split([])

        assert splits == []

    def test_messages_assigned_to_correct_split(self):
        messages = [
            _make_message("first", datetime.datetime(2025, 1, 15, 10, 59, 59)),
            _make_message("second", datetime.datetime(2025, 1, 15, 11, 0, 1)),
        ]
        strategy = TimeSplitStrategy(time_unit="hour")
        splits = strategy.split(messages)

        assert len(splits) == 2
        assert splits[0].messages[0].text == "first"
        assert splits[1].messages[0].text == "second"


class TestNoSplitStrategy:
    def test_returns_single_collection_with_all_messages(self):
        messages = [
            _make_message("a", datetime.datetime(2025, 1, 15, 10, 0, 0)),
            _make_message("b", datetime.datetime(2025, 1, 15, 12, 0, 0)),
        ]
        strategy = NoSplitStrategy()
        splits = strategy.split(messages)

        assert len(splits) == 1
        assert len(splits[0].messages) == 2

    def test_metadata_is_time_metadata_when_messages_present(self):
        messages = [
            _make_message("a", datetime.datetime(2025, 1, 15, 10, 0, 0)),
            _make_message("b", datetime.datetime(2025, 1, 15, 12, 0, 0)),
        ]
        strategy = NoSplitStrategy()
        splits = strategy.split(messages)

        assert isinstance(splits[0].metadata, TimeMetadata)
        assert splits[0].metadata.start_dt == datetime.datetime(2025, 1, 15, 10, 0, 0)
        assert splits[0].metadata.end_dt == datetime.datetime(2025, 1, 15, 12, 0, 0)

    def test_empty_messages_returns_empty_metadata(self):
        strategy = NoSplitStrategy()
        splits = strategy.split([])

        assert len(splits) == 1
        assert len(splits[0].messages) == 0
        assert isinstance(splits[0].metadata, Metadata)
        assert len(splits[0].metadata) == 0

    def test_single_message(self):
        msg = _make_message("hello", datetime.datetime(2025, 1, 15, 10, 0, 0))
        strategy = NoSplitStrategy()
        splits = strategy.split([msg])

        assert len(splits) == 1
        assert len(splits[0].messages) == 1
        assert splits[0].metadata.start_dt == datetime.datetime(2025, 1, 15, 10, 0, 0)
        assert splits[0].metadata.end_dt == datetime.datetime(2025, 1, 15, 10, 0, 0)


class TestPrecomputedSplitStrategy:
    def test_returns_precomputed_splits(self):
        precomputed = [
            MessageCollection(
                messages=[_make_message("a", datetime.datetime(2025, 1, 1))],
                metadata=Metadata(topic="cats"),
            ),
            MessageCollection(
                messages=[_make_message("b", datetime.datetime(2025, 1, 2))],
                metadata=Metadata(topic="dogs"),
            ),
        ]
        strategy = PrecomputedSplitStrategy(splits=precomputed)
        result = strategy.split([])

        assert result is precomputed
        assert len(result) == 2
        assert result[0].metadata["topic"] == "cats"
        assert result[1].metadata["topic"] == "dogs"


class TestMetadata:
    def test_dict_access_works(self):
        metadata = Metadata(topic="cats", count=5)
        assert metadata["topic"] == "cats"
        assert metadata["count"] == 5
        assert metadata.get("missing") is None

    def test_merge_preserves_existing_keys(self):
        m1 = Metadata(topic="cats")
        m2 = Metadata(topic="dogs", count=5)
        merged = m1.merge(m2)
        assert merged["topic"] == "dogs"
        assert merged["count"] == 5

    def test_merge_adds_new_keys(self):
        m1 = Metadata(topic="cats")
        m2 = Metadata(count=5)
        merged = m1.merge(m2)
        assert merged["topic"] == "cats"
        assert merged["count"] == 5


class TestTimeMetadata:
    def test_dict_access_works(self):
        metadata = TimeMetadata(
            start_dt=datetime.datetime(2025, 1, 15, 10, 0, 0),
            end_dt=datetime.datetime(2025, 1, 15, 11, 0, 0),
        )
        assert metadata["start_dt"] == datetime.datetime(2025, 1, 15, 10, 0, 0)
        assert metadata["end_dt"] == datetime.datetime(2025, 1, 15, 11, 0, 0)

    def test_attribute_access_works(self):
        metadata = TimeMetadata(
            start_dt=datetime.datetime(2025, 1, 15, 10, 0, 0),
            end_dt=datetime.datetime(2025, 1, 15, 11, 0, 0),
        )
        assert metadata.start_dt == datetime.datetime(2025, 1, 15, 10, 0, 0)
        assert metadata.end_dt == datetime.datetime(2025, 1, 15, 11, 0, 0)

    def test_merge_two_time_metadatas(self):
        m1 = TimeMetadata(
            start_dt=datetime.datetime(2025, 1, 15, 10, 0, 0),
            end_dt=datetime.datetime(2025, 1, 15, 11, 0, 0),
        )
        m2 = TimeMetadata(
            start_dt=datetime.datetime(2025, 1, 15, 11, 0, 0),
            end_dt=datetime.datetime(2025, 1, 15, 12, 0, 0),
        )
        merged = m1.merge(m2)
        assert isinstance(merged, TimeMetadata)
        assert merged.start_dt == datetime.datetime(2025, 1, 15, 10, 0, 0)
        assert merged.end_dt == datetime.datetime(2025, 1, 15, 12, 0, 0)

    def test_merge_time_with_generic_metadata(self):
        m1 = TimeMetadata(
            start_dt=datetime.datetime(2025, 1, 15, 10, 0, 0),
            end_dt=datetime.datetime(2025, 1, 15, 11, 0, 0),
        )
        m2 = Metadata(topic="cats")
        merged = m1.merge(m2)
        assert isinstance(merged, Metadata)
        assert merged["start_dt"] == datetime.datetime(2025, 1, 15, 10, 0, 0)
        assert merged["topic"] == "cats"

    def test_isinstance_of_dict_and_metadata(self):
        metadata = TimeMetadata(
            start_dt=datetime.datetime(2025, 1, 15, 10, 0, 0),
            end_dt=datetime.datetime(2025, 1, 15, 11, 0, 0),
        )
        assert isinstance(metadata, dict)
        assert isinstance(metadata, Metadata)


class TestMessageProviderSplitByConsistency:
    def test_split_by_matches_time_split_strategy(self):
        from kygs.message_provider import MessageProvider

        messages = [
            _make_message("a", datetime.datetime(2025, 1, 15, 10, 10, 0)),
            _make_message("b", datetime.datetime(2025, 1, 15, 10, 30, 0)),
            _make_message("c", datetime.datetime(2025, 1, 15, 11, 5, 0)),
        ]
        mp = MessageProvider(messages=messages)

        result_mp = mp.split_by("hour")
        result_strategy = TimeSplitStrategy(time_unit="hour").split(messages)

        assert len(result_mp) == len(result_strategy)
        for mp_split, strategy_split in zip(result_mp, result_strategy):
            assert len(mp_split.messages) == len(strategy_split.messages)
            assert mp_split.metadata.start_dt == strategy_split.metadata.start_dt
            assert mp_split.metadata.end_dt == strategy_split.metadata.end_dt
