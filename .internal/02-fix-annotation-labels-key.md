# Fix AnnotatedSummaryBuilder labels key configuration

## 1. Requirement analysis

**Problem**: In `AnnotatedSummaryBuilder.__call__`, the key `"labels"` is hardcoded when accessing the parsed JSON response (line 134 in `kygs/summarization/direct.py`). This key name depends on the prompt structure and may vary across different prompts.

**Solution**: Move the labels key name to the `AnnotatedSummaryBuilder` configuration by adding a new `labels_key` parameter.

**Requirements**:
1. Add a `labels_key` parameter to `AnnotatedSummaryBuilder.__init__` with a default value of `"labels"` for backward compatibility
2. Use the `labels_key` parameter when accessing the labels from the parsed JSON response
3. Update the existing config file `config/summary_builder/annotated.yaml` to explicitly specify the `labels_key` parameter
4. Ensure existing tests continue to pass with the default value
5. Add tests to verify the `labels_key` parameter works correctly with custom values

**Expected variants**:
- Default behavior: `labels_key="labels"` (backward compatible)
- Custom behavior: User can specify any key name via config (e.g., `labels_key: "categories"`, `labels_key: "tags"`, etc.)

## 2. Tests

**Existing tests to verify** (should pass without modification due to backward compatibility):
- `TestAnnotatedSummaryBuilder.test_parses_json_with_single_label` - uses default `labels_key="labels"`
- `TestAnnotatedSummaryBuilder.test_parses_json_with_multiple_labels` - uses default `labels_key="labels"`
- `TestAnnotatedSummaryBuilder.test_parses_json_with_empty_labels` - uses default `labels_key="labels"`
- `TestAnnotatedSummaryBuilder.test_preserves_existing_metadata` - uses default `labels_key="labels"`
- `TestAnnotatedSummaryBuilder.test_preserves_existing_labels_key_in_metadata` - uses default `labels_key="labels"`
- `TestAnnotatedSummaryBuilder.test_custom_metadata_key` - uses default `labels_key="labels"`
- `TestAnnotatedSummaryBuilder.test_custom_metadata_key_with_existing_metadata` - uses default `labels_key="labels"`
- `TestAnnotatedSummaryBuilder.test_raises_on_metadata_field_collision` - uses default `labels_key="labels"`
- `TestAnnotatedSummaryBuilder.test_annotation_labels_use_union_merge_strategy` - uses default `labels_key="labels"`

**New tests to add**:
1. `test_custom_labels_key` - Test that a custom `labels_key` parameter correctly extracts labels from a different JSON key
2. `test_custom_labels_key_with_config` - Integration test verifying the config file can specify `labels_key`

## 3. Implementation plan

### 3.1 Todo list

1. [ ] Write the new tests for custom `labels_key` functionality
2. [ ] Run all tests and ensure new tests fail (TDD approach)
3. [ ] Modify `AnnotatedSummaryBuilder.__init__` to accept `labels_key` parameter with default `"labels"`
4. [ ] Modify `AnnotatedSummaryBuilder.__call__` to use `self.labels_key` instead of hardcoded `"labels"`
5. [ ] Update `config/summary_builder/annotated.yaml` to explicitly specify `labels_key: "labels"`
6. [ ] Run all tests and ensure they pass
7. [ ] Run linters: `black`, `isort`, `pylint`, `mypy`
8. [ ] Update `docs/summarization.md` to document the new `labels_key` parameter

### 3.2 Modification summary

| File | Action |
|------|--------|
| `kygs/summarization/direct.py` | Modified: Add `labels_key` parameter to `AnnotatedSummaryBuilder.__init__` and use it in `__call__` |
| `config/summary_builder/annotated.yaml` | Modified: Add explicit `labels_key: "labels"` configuration |
| `tests/test_summarization.py` | Modified: Add new test methods for custom `labels_key` functionality |
| `docs/summarization.md` | Modified: Document the new `labels_key` parameter in `AnnotatedSummaryBuilder` |
