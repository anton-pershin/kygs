# 03 — Refactor logging to standard Python logging

## 1. Requirement analysis

The project currently mixes bare `print()` and `console.print()` (rich) statements for output. We want to move to standard Python `logging` for non-interactive, non-rich-formatted output while keeping `console.print` for interactive UIs and rich-formatted final reports.

### Files whose `print()`/`console.print()` MUST be replaced with `logging`

| File | Current | Replacement | Reason |
|------|---------|-------------|--------|
| `kygs/summarization/direct.py` | `print(f"Failed to parse the model response: '{text}'")` (line 125) | `logging.error(...)` | Error condition |
| `kygs/summarization/recursive.py` | `print(f"Summarizing the {mc_i + 1}/{n_mcs} cluster", end="")` and `print(f". Time elapsed: {time_elapsed} sec")` (lines 45, 52) | `logging.info(...)` (single message per cluster) | Progress/status |
| `kygs/scripts/recurrently_cluster_posts.py` | 4× `console.print` status messages (lines 26, 46, 49, 70) | `logging.info(...)` | Script status/progress |
| `kygs/scripts/tools/split_into_train_and_test.py` | 3× `print` path info (lines 25-27) | `logging.info(...)` | Config/paths info |
| `kygs/scripts/tools/convert_jsonl_to_message_dataset.py` | 3× `print` path info (lines 35-37) | `logging.info(...)` | Config/paths info |
| `kygs/scripts/tools/merge_message_datasets.py` | 4× `print` merge info (lines 23-28) | `logging.info(...)` | Config/paths info |

### Files whose `console.print`/`print` MUST be kept (no change)

- `kygs/clustering/base.py` — rich-formatted evaluation report (keep)
- `kygs/classifier.py`, `kygs/lightweight_classifier.py` — rich classification report (keep)
- `kygs/message_provider.py` — interactive browse UI (keep)
- `kygs/annotation/manual.py` — interactive annotation UI (keep)
- `kygs/utils/console.py` — `prompt_user()` interactive (keep)
- `kygs/scripts/train_text_classifier.py`, `kygs/scripts/train_lightweight_text_classifier.py` — rich Table summary (keep)
- `kygs/scripts/collect_recent_posts.py` — keep all
- `kygs/scripts/annotate.py`, `kygs/scripts/cluster_posts.py` — keep
- `kygs/scripts/tools/print_relevant_messages.py` — final user-facing search output (keep)

### Logging infrastructure

There is currently no logging configuration in the project. We use the stdlib `logging` module directly — no custom wrapper module:

- Library modules (`kygs/summarization/direct.py`, `kygs/summarization/recursive.py`) call `logging.error(...)` / `logging.info(...)` directly (these emit on the root logger). They do NOT configure logging.

- Each executable script whose main function emits log messages calls `logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")` **under the `if __name__ == "__main__":` guard**, before `hydra.main(...)` is invoked (NOT inside the main function body). This repo is used as a library whose script functions are imported by applications with their own logging setup; placing `basicConfig` under the guard ensures importing a script never reconfigures the host's logging. `basicConfig` is otherwise a no-op if the root logger is already configured. Specifically:
  - `kygs/scripts/recurrently_cluster_posts.py`
  - `kygs/scripts/tools/split_into_train_and_test.py`
  - `kygs/scripts/tools/convert_jsonl_to_message_dataset.py`
  - `kygs/scripts/tools/merge_message_datasets.py`

### Expected variants

- V1: `direct.py` parse-failure logs at `ERROR` level and returns `None` (existing behavior preserved).
- V2: `recursive.py` logs one `INFO` line per cluster summarised (combining the previous two prints into a single log call per cluster) OR two `INFO` lines (start + end) — implementation may choose; test only asserts that an `INFO` record mentioning the cluster index is emitted per cluster.
- V3: tools scripts log paths via `logging.info` and produce no `print()` output.
- V4: `recurrently_cluster_posts` logs the four status messages at `INFO` level.

## 2. Tests

No tests are added for this spec. Per the user's decision (deviating from KDD's TDD core), correctness is verified manually since the changes are trivial and observable from script output. The existing test suite must still pass after the refactor (no regressions).

## 3. Implementation plan

### 3.1 Solution design

1. **Logging via stdlib** — no new module. Library modules call `logging.error()`/`logging.info()` directly (root logger). Scripts call `logging.basicConfig(...)` at the top of their main function.

2. **Library modules** replace `print` with `logging.error` / `logging.info` and never configure logging, so host applications with their own handlers capture their records.

3. **Executable scripts** that emit log messages call `logging.basicConfig(...)` **under the `if __name__ == "__main__":` guard** (before `hydra.main(...)`), NOT inside the main function body. Because this repo doubles as a library whose script functions are imported by applications with their own logging setup, `basicConfig` must only run on direct execution — never on import. `print`/`console.print` status messages are replaced with `logging.info`; unused `console` imports are removed.

4. **No changes** to the files in the "keep" list.

5. Documentation: update `README.md` / `docs/` only if they describe the print/console output behavior (check during implementation; add a short note about `logging.basicConfig` if a developer-guide section exists).

### 3.2 Todo list

1. [ ] Refactor `kygs/summarization/direct.py` (use `logging.error`)
2. [ ] Refactor `kygs/summarization/recursive.py` (use `logging.info`)
3. [ ] Refactor `kygs/scripts/recurrently_cluster_posts.py` (call `logging.basicConfig` under `if __name__ == "__main__"`, use `logging.info`)
4. [ ] Refactor `kygs/scripts/tools/split_into_train_and_test.py`
5. [ ] Refactor `kygs/scripts/tools/convert_jsonl_to_message_dataset.py`
6. [ ] Refactor `kygs/scripts/tools/merge_message_datasets.py`
7. [ ] Run the existing test suite to ensure no regressions
8. [ ] Run linters: `black kygs/`, `isort kygs/`, `pylint kygs/`, `mypy kygs/`
9. [ ] Update `README.md` / `docs/` if needed

### 3.3 Modification summary

| File | Action |
|------|--------|
| `kygs/summarization/direct.py` | Modified: replace `print` with `logging.error`; add `import logging` |
| `kygs/summarization/recursive.py` | Modified: replace `print` with `logging.info`; add `import logging` |
| `kygs/scripts/recurrently_cluster_posts.py` | Modified: call `logging.basicConfig` under `if __name__ == "__main__"`; replace 4× `console.print` with `logging.info`; drop unused `console` import |
| `kygs/scripts/tools/split_into_train_and_test.py` | Modified: call `logging.basicConfig` under `if __name__ == "__main__"`; replace 3× `print` with `logging.info` |
| `kygs/scripts/tools/convert_jsonl_to_message_dataset.py` | Modified: call `logging.basicConfig` under `if __name__ == "__main__"`; replace 3× `print` with `logging.info` |
| `kygs/scripts/tools/merge_message_datasets.py` | Modified: call `logging.basicConfig` under `if __name__ == "__main__"`; replace 4× `print` with `logging.info` |
| `README.md` / `docs/` | Modified (if relevant): note logging usage |
