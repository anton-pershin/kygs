# Spec 04: Summary build failure handling in RecursiveSummarization

## 1. Requirement analysis

### Problem
When `AnnotatedSummaryBuilder.__call__` fails to parse the model response (malformed
JSON), it returns `None`. `DirectSummarization.__call__` filters out these `None`
values (`[s for s in summaries if s is not None]`). If **all** summaries in a
partition fail to parse, the resulting `summaries` list is empty.

In `RecursiveSummarization._summarize_recursively`, an empty `summaries` list is
passed to `to_message_collection([])`, which calls `merge_metadatas([])` →
`metadatas[0]` → `IndexError`. This crash was observed in production.

### Requirements
1. `AnnotatedSummaryBuilder` continues to return `None` on parse failure (no change
   — it is a low-level component and must not decide control flow).
2. `DirectSummarization` continues to filter out `None` summaries (no change).
3. `RecursiveSummarization._summarize_recursively` must detect when `summaries` is
   empty after a summarization step and raise a new controllable exception
   `SummaryBuildFailureException` (defined in `recursive.py`).
4. `RecursiveSummarization.__call__` must catch `SummaryBuildFailureException` for
   each message collection, log a warning, and skip that collection, continuing with
   the remaining ones (graceful degradation — returns fewer summaries instead of
   crashing the whole batch).
5. **Boundary case**: if all message collections are skipped and the resulting
   `summaries` list in `__call__` is empty, `__call__` must raise
   `SummaryBuildFailureException` (it is expected to return at least one summary).

### Out of scope
- Retry logic for failed LLM requests.
- Fallback to raw text as a plain summary.
- Changes to `AnnotatedSummaryBuilder` or `DirectSummarization` behavior.

## 2. Tests

All tests live in `tests/test_summarization.py`. The mock LLM setup
(`MockLlm`, `MOCK_THINKING_REMOVERS`, `_identity`) already present in the file is
reused.

### Test 2.1 (primary — the "one clear and simple test" that currently fails)
**`test_recursive_raises_when_all_summaries_fail_to_parse`** (in `TestSummarizeSummaries`)

- Setup: a single `MessageCollection` built from two `Summary` objects (so the
  recursion enters the `while` loop with >1 message).
- Both `original_message_summary_builder` and `partial_summary_builder` are
  `AnnotatedSummaryBuilder` instances.
- Mock `request_based_on_prompts` to return `["this is not valid json"]` (malformed
  response → `AnnotatedSummaryBuilder` returns `None` → filtered → empty list).
- Assert: `pytest.raises(SummaryBuildFailureException)`.

This test currently fails because the code raises `IndexError` (from
`merge_metadatas([])`) instead of `SummaryBuildFailureException`.

### Test 2.2 (skip behavior)
**`test_recursive_skips_failed_collection_and_returns_successful`** (in
`TestSummarizeSummaries`)

- Setup: two `MessageCollection`s, each built from two `Summary` objects.
- Mock `request_based_on_prompts` with `side_effect`:
  - 1st call (collection 1) → `["this is not valid json"]` (fails).
  - 2nd call (collection 2) → `[json.dumps({"summary": "Valid summary.", "labels":
    ["positive"]})]` (succeeds).
- Assert: result has length 1, `result[0].text == "Valid summary."`.

### Test 2.3 (boundary case — all collections fail)
**`test_recursive_raises_when_all_collections_skipped`** (in `TestSummarizeSummaries`)

- Setup: two `MessageCollection`s, each built from two `Summary` objects.
- Mock `request_based_on_prompts` to always return `["this is not valid json"]`.
- Assert: `pytest.raises(SummaryBuildFailureException)`.

## 3. Implementation plan

### 3.1 Solution design

All changes are localized to `kygs/summarization/recursive.py`:

1. **New exception**: `SummaryBuildFailureException(Exception)` (alongside the
   existing `OutOfContextLengthException` and `LackOfConvergenceException`).

2. **`_summarize_recursively`**: after obtaining `summaries` from either
   `original_message_summarization` or `partial_summary_summarization`, and before
   calling `to_message_collection(summaries)`, check `if not summaries:` and raise
   `SummaryBuildFailureException`.

3. **`__call__`**: wrap the `self._summarize_recursively(mc)` call in a
   `try/except SummaryBuildFailureException`. On catch, log a warning and `continue`
   to the next collection. After the loop, `if not summaries:` raise
   `SummaryBuildFailureException`.

### 3.2 Todo list

1. [ ] Write tests 2.1, 2.2, 2.3 in `tests/test_summarization.py`.
2. [ ] Run the tests and ensure they fail (2.1 and 2.3 with `IndexError` instead of
   `SummaryBuildFailureException`; 2.2 with `IndexError`).
3. [ ] Add `SummaryBuildFailureException` to `recursive.py`.
4. [ ] Add the empty-summaries check + raise in `_summarize_recursively`.
5. [ ] Add the try/except skip logic + boundary raise in `__call__`.
6. [ ] Run the tests and ensure they pass.
7. [ ] Run the full test suite (`pytest`) to ensure no regressions.
8. [ ] Run linters: `black kygs/`, `isort kygs/`, `pylint kygs/`, `mypy kygs/`.

### 3.3 Modification summary

| File | Action |
|------|--------|
| `kygs/summarization/recursive.py` | Modified: add `SummaryBuildFailureException`; add empty-summaries check in `_summarize_recursively`; add try/except skip + boundary raise in `__call__` |
| `tests/test_summarization.py` | Modified: add three tests in `TestSummarizeSummaries` |
