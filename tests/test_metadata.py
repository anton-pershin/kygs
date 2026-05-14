import datetime

import pytest

from kygs.metadata import Metadata, LabelMetadata, TimeMetadata


class TestFieldLevelMergeStrategies:
    """Test field-level merge strategy configuration."""

    def test_field_level_merge_strategies(self):
        """Test that different fields can have different merge strategies."""

        class CustomMetadata(Metadata):
            _merge_strategies = {
                "labels": "concat",
                "annotation_labels": "union",
                "count": "replace",
            }

            def __init__(
                self,
                labels: list[str] | None = None,
                annotation_labels: list[str] | None = None,
                count: int = 0,
            ):
                super().__init__(
                    labels=labels or [],
                    annotation_labels=annotation_labels or [],
                    count=count,
                )

        m1 = CustomMetadata(
            labels=["a", "b"],
            annotation_labels=["x", "y"],
            count=1,
        )
        m2 = CustomMetadata(
            labels=["c"],
            annotation_labels=["y", "z"],
            count=2,
        )
        merged = m1.merge(m2)

        assert merged["labels"] == ["a", "b", "c"]
        assert set(merged["annotation_labels"]) == {"x", "y", "z"}
        assert merged["count"] == 2

    def test_default_merge_strategy(self):
        """Test that undefined fields use default 'replace' strategy."""

        class PartialConfigMetadata(Metadata):
            _merge_strategies = {
                "labels": "concat",
            }

            def __init__(self, labels: list[str], name: str):
                super().__init__(labels=labels, name=name)

        m1 = PartialConfigMetadata(labels=["a"], name="first")
        m2 = PartialConfigMetadata(labels=["b"], name="second")
        merged = m1.merge(m2)

        assert merged["labels"] == ["a", "b"]
        assert merged["name"] == "second"

    def test_union_merge_strategy_with_duplicates(self):
        """Test that union merge strategy removes duplicates."""

        class UnionMetadata(Metadata):
            _merge_strategies = {
                "tags": "union",
            }

            def __init__(self, tags: list[str]):
                super().__init__(tags=tags)

        m1 = UnionMetadata(tags=["a", "b", "c"])
        m2 = UnionMetadata(tags=["b", "c", "d"])
        merged = m1.merge(m2)

        assert set(merged["tags"]) == {"a", "b", "c", "d"}

    def test_merge_different_metadata_types(self):
        """Test merging metadata of different types."""
        lm = LabelMetadata(["a", "b"])
        base_md = Metadata({"extra": "value"})

        merged = lm.merge(base_md)
        assert merged["labels"] == ["a", "b"]
        assert merged["extra"] == "value"


class TestAnnotatedSummaryBuilderPreservesClass:
    """Test that AnnotatedSummaryBuilder preserves metadata class type."""

    def test_annotated_summary_builder_preserves_class(self):
        """Test that AnnotatedSummaryBuilder preserves the metadata class."""
        import json

        from kygs.summarization.direct import AnnotatedSummaryBuilder

        class CustomMetadataWithAnnotations(Metadata):
            _merge_strategies = {
                "custom_field": "concat",
            }

            def __init__(
                self,
                custom_field: list[str] | None = None,
            ):
                super().__init__(
                    custom_field=custom_field or [],
                )

        input_metadata = CustomMetadataWithAnnotations(
            custom_field=["field1"],
        )

        builder = AnnotatedSummaryBuilder(metadata_key="annotation_labels")
        llm_response = json.dumps(
            {
                "summary": "Test summary",
                "labels": ["ann1", "ann2", "ann3"],
            }
        )

        summary = builder(text=llm_response, metadata=input_metadata)

        assert isinstance(summary.metadata, CustomMetadataWithAnnotations)
        assert set(summary.metadata["annotation_labels"]) == {"ann1", "ann2", "ann3"}
        assert summary.metadata["custom_field"] == ["field1"]


class TestBackwardCompatibility:
    """Test that existing metadata classes work as before."""

    def test_base_metadata_merge(self):
        """Test base Metadata merge behavior."""
        m1 = Metadata({"a": 1, "b": 2})
        m2 = Metadata({"b": 3, "c": 4})
        merged = m1.merge(m2)
        assert merged == {"a": 1, "b": 3, "c": 4}

    def test_label_metadata_merge(self):
        """Test LabelMetadata merge behavior."""
        lm1 = LabelMetadata(["a", "b"])
        lm2 = LabelMetadata(["c"])
        merged_lm = lm1.merge(lm2)
        assert merged_lm.labels == ["a", "b", "c"]

    def test_time_metadata_merge(self):
        """Test TimeMetadata merge behavior."""
        tm1 = TimeMetadata(
            start_dt=datetime.datetime(2024, 1, 1),
            end_dt=datetime.datetime(2024, 1, 10),
        )
        tm2 = TimeMetadata(
            start_dt=datetime.datetime(2024, 1, 5),
            end_dt=datetime.datetime(2024, 1, 15),
        )
        merged_tm = tm1.merge(tm2)
        assert merged_tm.start_dt == datetime.datetime(2024, 1, 1)
        assert merged_tm.end_dt == datetime.datetime(2024, 1, 15)

    def test_label_metadata_preserves_class_on_merge(self):
        """Test that LabelMetadata.merge returns LabelMetadata instance."""
        lm1 = LabelMetadata(["a", "b"])
        lm2 = LabelMetadata(["c"])
        merged = lm1.merge(lm2)
        assert isinstance(merged, LabelMetadata)

    def test_time_metadata_preserves_class_on_merge(self):
        """Test that TimeMetadata.merge returns TimeMetadata instance."""
        tm1 = TimeMetadata(
            start_dt=datetime.datetime(2024, 1, 1),
            end_dt=datetime.datetime(2024, 1, 10),
        )
        tm2 = TimeMetadata(
            start_dt=datetime.datetime(2024, 1, 5),
            end_dt=datetime.datetime(2024, 1, 15),
        )
        merged = tm1.merge(tm2)
        assert isinstance(merged, TimeMetadata)


class TestMetadataClassPreservation:
    """Test that metadata class is preserved through various operations."""

    def test_metadata_constructor_with_dict(self):
        """Test Metadata constructor with dict argument."""
        md = Metadata({"a": 1, "b": 2})
        assert md["a"] == 1
        assert md["b"] == 2

    def test_custom_metadata_class_preserved_on_merge(self):
        """Test that custom metadata class is preserved on merge."""

        class CustomMetadata(Metadata):
            _merge_strategies = {"value": "replace"}

            def __init__(self, value: int = 0):
                super().__init__(value=value)

        m1 = CustomMetadata(value=10)
        m2 = CustomMetadata(value=20)
        merged = m1.merge(m2)

        assert isinstance(merged, CustomMetadata)
        assert merged["value"] == 20
