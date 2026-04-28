# 20260429_add_json_summary_handler

## User request

Add JSON output option for summary dump in summarization (mirroring message handlers in cluster_posts).

## Solution description

Introduce a `SummaryHandler` abstraction and two concrete implementations (`SummaryCsvSaver`, `SummaryJsonSaver`) similar to `MessageHandler` in `kygs/message_handler.py`. Refactor `summarize_posts.py` to use the handler loop pattern, and update `config_summarize_posts.yaml` to expose handler lists.

## Tasks

- [ ] Create `kygs/summary_handler.py` with `SummaryHandler` base class and `SummaryCsvSaver`, `SummaryJsonSaver` concrete implementations
- [ ] Create `config/summary_handler/` directory with YAML defaults for the handlers
- [ ] Refactor `kygs/scripts/summarize_posts.py` to use handler loop instead of hardcoded CSV dumping
- [ ] Update `config/config_summarize_posts.yaml` to use `summary_handlers` lists in both `original_post_summarization` and `recursive_summarization` sections
- [ ] Add tests for `SummaryHandler` implementations in `tests/test_summary_handler.py`
- [ ] Run linters and typecheckers: `black`, `isort`, `pylint`, `mypy`
- [ ] Run tests: `pytest`
