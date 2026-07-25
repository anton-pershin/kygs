# TODO refactor logging

## User request (as understood)

Refactor the logging system to use standard Python `logging` instead of a mix of `print()` and `console.print()`. Replace `print`/`console.print` in non-interactive, non-rich-formatted contexts with `logging.info()`/`logging.error()`. Keep `console.print` for interactive UIs and rich-formatted final reports. No custom logging utility module; scripts call `logging.basicConfig(...)` under `if __name__ == "__main__":`. No new tests (per user decision); existing test suite must still pass.

## Solution description

See spec `.opencode/specs/03-refactor-logging.md`.

## Tasks

- [x] Create branch `20260725_refactor_logging`
- [x] Refactor `kygs/summarization/direct.py` (use `logging.error`)
- [x] Refactor `kygs/summarization/recursive.py` (use `logging.info`)
- [x] Refactor `kygs/scripts/recurrently_cluster_posts.py` (call `logging.basicConfig` under `if __name__ == "__main__"`, use `logging.info`)
- [x] Refactor `kygs/scripts/tools/split_into_train_and_test.py`
- [x] Refactor `kygs/scripts/tools/convert_jsonl_to_message_dataset.py`
- [x] Refactor `kygs/scripts/tools/merge_message_datasets.py`
- [x] Run the existing test suite to ensure no regressions (94 passed)
- [x] Run linters: `black kygs/`, `isort kygs/`, `pylint kygs/`, `mypy kygs/` (no new issues; pre-existing unused-var in direct.py and pre-existing mypy None-return unchanged)
- [x] Update `README.md` / `docs/` — no changes needed (no docs describe print/logging output)
- [ ] Commit changes
