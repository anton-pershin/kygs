# Spec: Metadata Merge Strategy with Field-Level Configuration

## 1. Requirement analysis

The current architecture has the following problems:

1. **Class-specific merge strategy**: `AnnotatedSummaryBuilder` creates new metadata using the base `Metadata` class, which loses the merge strategy of the input metadata class (e.g., `LabelMetadata`, `TimeMetadata`).

2. **No field-level merge control**: All fields within a metadata class use the same merge strategy, but different fields may need different strategies:
   - `LabelMetadata.labels`: Should use concatenation (`+`)
   - Annotation labels (e.g., `annotation_labels`): Should use set union

**Requirements:**

1. **Preserve metadata class type**: When enriching metadata (e.g., in `AnnotatedSummaryBuilder`), the output metadata should be the same class as the input metadata.

2. **Field-level merge strategies**: Each field should have its own configurable merge strategy defined at class definition time.

3. **Per-field configuration**: Merge strategies are configured per field name, not per data type.

4. **Class definition time configuration**: Merge strategies are fixed when the Metadata subclass is defined (no runtime configuration needed).

5. **Backward compatibility**: Existing `Metadata`, `TimeMetadata`, and `LabelMetadata` classes should continue to work with their current behavior.

**Expected merge strategies:**
- `concat`: Concatenate lists using `+` (current behavior for `LabelMetadata.labels`)
- `union`: Compute set union for lists (desired for annotation labels)
- `replace`: Replace with the new value (default for non-list fields)
- `min`: Take minimum (used in `TimeMetadata.start_dt`)
- `max`: Take maximum (used in `TimeMetadata.end_dt`)

## 2. Tests

### Test 1: Field-level merge strategy configuration
Test that a `Metadata` subclass can define different merge strategies for different fields.

```python
def test_field_level_merge_strategies():
    """Test that different fields can have different merge strategies."""
    class CustomMetadata(Metadata):
        _merge_strategies = {
            "labels": "concat",      # concatenation
            "annotation_labels": "union",  # set union
            "count": "replace",      # replacement
        }
        
        def __init__(self, labels: list[str], annotation_labels: list[str], count: int):
            super().__init__(
                labels=labels,
                annotation_labels=annotation_labels,
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
    
    assert merged["labels"] == ["a", "b", "c"]  # concat
    assert set(merged["annotation_labels"]) == {"x", "y", "z"}  # union
    assert merged["count"] == 2  # replace (from m2)
```

### Test 2: AnnotatedSummaryBuilder preserves metadata class
Test that `AnnotatedSummaryBuilder` preserves the input metadata class type.

```python
def test_annotated_summary_builder_preserves_class():
    """Test that AnnotatedSummaryBuilder preserves the metadata class."""
    class LabelMetadataWithAnnotations(LabelMetadata):
        _merge_strategies = {
            "labels": "concat",
            "annotation_labels": "union",
        }
    
    input_metadata = LabelMetadataWithAnnotations(
        labels=["label1"],
        annotation_labels=["ann1"],
    )
    
    builder = AnnotatedSummaryBuilder(metadata_key="annotation_labels")
    llm_response = json.dumps({
        "summary": "Test summary",
        "labels": ["ann2", "ann3"],
    })
    
    summary = builder(text=llm_response, metadata=input_metadata)
    
    # Check that the class is preserved
    assert isinstance(summary.metadata, LabelMetadataWithAnnotations)
    assert set(summary.metadata["annotation_labels"]) == {"ann1", "ann2", "ann3"}
```

### Test 3: Default merge strategy for undefined fields
Test that fields without explicit merge strategy use the default (replace).

```python
def test_default_merge_strategy():
    """Test that undefined fields use default 'replace' strategy."""
    class PartialConfigMetadata(Metadata):
        _merge_strategies = {
            "labels": "concat",
            # "name" is not configured, should use default "replace"
        }
        
        def __init__(self, labels: list[str], name: str):
            super().__init__(labels=labels, name=name)
    
    m1 = PartialConfigMetadata(labels=["a"], name="first")
    m2 = PartialConfigMetadata(labels=["b"], name="second")
    merged = m1.merge(m2)
    
    assert merged["labels"] == ["a", "b"]
    assert merged["name"] == "second"  # default replace
```

### Test 4: Backward compatibility with existing classes
Test that existing `Metadata`, `TimeMetadata`, and `LabelMetadata` work as before.

```python
def test_backward_compatibility():
    """Test that existing metadata classes work as before."""
    # Test base Metadata
    m1 = Metadata({"a": 1, "b": 2})
    m2 = Metadata({"b": 3, "c": 4})
    merged = m1.merge(m2)
    assert merged == {"a": 1, "b": 2, "c": 4}  # m1's "b" is kept
    
    # Test LabelMetadata
    lm1 = LabelMetadata(["a", "b"])
    lm2 = LabelMetadata(["c"])
    merged_lm = lm1.merge(lm2)
    assert merged_lm.labels == ["a", "b", "c"]
    
    # Test TimeMetadata
    from datetime import datetime
    tm1 = TimeMetadata(
        start_dt=datetime(2024, 1, 1),
        end_dt=datetime(2024, 1, 10),
    )
    tm2 = TimeMetadata(
        start_dt=datetime(2024, 1, 5),
        end_dt=datetime(2024, 1, 15),
    )
    merged_tm = tm1.merge(tm2)
    assert merged_tm.start_dt == datetime(2024, 1, 1)  # min
    assert merged_tm.end_dt == datetime(2024, 1, 15)  # max
```

### Test 5: Union merge strategy with duplicates
Test that union merge strategy properly handles duplicates.

```python
def test_union_merge_strategy_with_duplicates():
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
    # Order may vary, but all unique elements should be present
```

### Test 6: Merge with different metadata types
Test merging metadata of different types (fallback to base behavior).

```python
def test_merge_different_metadata_types():
    """Test merging metadata of different types."""
    lm = LabelMetadata(["a", "b"])
    base_md = Metadata({"extra": "value"})
    
    # LabelMetadata.merge should handle non-LabelMetadata gracefully
    merged = lm.merge(base_md)
    assert merged["labels"] == ["a", "b"]
    assert merged["extra"] == "value"
```

## 3. Implementation plan

### 3.1 Todo list

1. [ ] Write the tests in `tests/test_metadata.py`
2. [ ] Run all the tests and ensure that they fail
3. [ ] Implement field-level merge strategy mechanism in `Metadata` class
4. [ ] Add built-in merge strategy functions (concat, union, replace, min, max)
5. [ ] Update `AnnotatedSummaryBuilder` to preserve metadata class type
6. [ ] Ensure backward compatibility with existing metadata classes
7. [ ] Run all tests and ensure they pass
8. [ ] Run linters (black, isort, pylint, mypy)

### 3.2 Modification summary

| File | Action |
|------|--------|
| `kygs/metadata.py` | Modified: Add `_merge_strategies` class attribute support, implement field-level merge logic, add built-in merge strategy functions |
| `kygs/summarization/direct.py` | Modified: Update `AnnotatedSummaryBuilder` to preserve input metadata class type |
| `tests/test_metadata.py` | New: Add comprehensive tests for field-level merge strategies |

## Design Details

### Merge Strategy Functions

Define module-level merge strategy functions:

```python
def merge_concat(old_value, new_value):
    """Concatenate two lists."""
    return old_value + new_value

def merge_union(old_value, new_value):
    """Compute set union of two lists, return as list."""
    return list(set(old_value) | set(new_value))

def merge_replace(old_value, new_value):
    """Replace old value with new value."""
    return new_value

def merge_min(old_value, new_value):
    """Take minimum of two values."""
    return min(old_value, new_value)

def merge_max(old_value, new_value):
    """Take maximum of two values."""
    return max(old_value, new_value)

MERGE_STRATEGIES = {
    "concat": merge_concat,
    "union": merge_union,
    "replace": merge_replace,
    "min": merge_min,
    "max": merge_max,
}
```

### Updated Metadata Class

```python
class Metadata(dict):
    _merge_strategies: dict[str, str] = {}  # field_name -> strategy_name
    
    def merge(self, other: Metadata) -> Metadata:
        merged = dict(self)
        for key, value in other.items():
            if key not in merged:
                merged[key] = value
            else:
                # Use field-specific strategy if defined
                strategy_name = self._merge_strategies.get(key, "replace")
                strategy_fn = MERGE_STRATEGIES.get(strategy_name, merge_replace)
                merged[key] = strategy_fn(merged[key], value)
        
        # Return instance of the same class
        return self.__class__(merged)
```

### Updated AnnotatedSummaryBuilder

```python
class AnnotatedSummaryBuilder(BaseSummaryBuilder):
    def __init__(self, metadata_key: str = "annotation_labels") -> None:
        self.metadata_key = metadata_key

    def __call__(self, text: str, metadata: Metadata) -> Summary:
        parsed = json.loads(text)
        
        # Check for field collision
        if self.metadata_key in metadata:
            raise MetadataFieldCollision(
                f"Metadata already contains key '{self.metadata_key}'"
            )
        
        # Preserve input metadata class and merge strategies
        original_class = type(metadata)
        class_name = f"Annotated{original_class.__name__}"
        parent_strategies = getattr(original_class, "_merge_strategies", {})
        new_strategies = {**parent_strategies, self.metadata_key: "union"}
        
        # Create annotated subclass dynamically
        AnnotatedMetadataClass = type(
            class_name, (original_class,), {"_merge_strategies": new_strategies}
        )
        
        # Enrich metadata with annotation labels
        enriched_dict = {**metadata, self.metadata_key: parsed["labels"]}
        enriched_metadata = AnnotatedMetadataClass.__new__(AnnotatedMetadataClass)
        dict.__init__(enriched_metadata, enriched_dict)
        
        return Summary(text=parsed["summary"], metadata=enriched_metadata)
```
