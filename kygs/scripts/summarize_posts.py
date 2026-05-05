import hydra
from omegaconf import DictConfig, ListConfig

from kygs.message_provider import MessageCollection
from kygs.split_strategy import SplitStrategy
from kygs.summarization.base import Summary, to_message_collection
from kygs.summarization.recursive import RecursiveSummarization
from kygs.utils.common import get_config_path

CONFIG_NAME = "config_summarize_posts"


def _dump_summaries(
    summary_collection: list[Summary],
    summary_handlers: ListConfig,
) -> None:
    for handler_cfg in summary_handlers:
        handler = hydra.utils.instantiate(handler_cfg)
        handler.handle(summary_collection)


def run_summarize_posts(cfg: DictConfig) -> None:
    mp = hydra.utils.call(cfg.message_provider)
    llm = hydra.utils.instantiate(cfg.llm)
    split_strategy: SplitStrategy = hydra.utils.instantiate(cfg.split_strategy)
    message_splits: list[MessageCollection] = split_strategy.split(mp.messages)
    original_message_summarization_prompt = hydra.utils.instantiate(
        cfg.original_message_summarization_prompt
    )
    partial_summary_summarization_prompt = hydra.utils.instantiate(
        cfg.partial_summary_summarization_prompt
    )
    original_message_summary_builder = hydra.utils.instantiate(
        cfg.original_message_summary_builder
    )
    partial_summary_builder = hydra.utils.instantiate(cfg.partial_summary_builder)

    original_split_summarization = RecursiveSummarization(
        llm=llm,
        original_message_summarization_prompt=original_message_summarization_prompt,
        partial_summary_summarization_prompt=partial_summary_summarization_prompt,
        original_message_summary_builder=original_message_summary_builder,
        partial_summary_builder=partial_summary_builder,
        max_characters_in_prompt=cfg.max_characters_in_prompt,
        verbose=cfg.verbose,
    )
    overall_summarization = RecursiveSummarization(
        llm=llm,
        original_message_summarization_prompt=original_message_summarization_prompt,
        partial_summary_summarization_prompt=partial_summary_summarization_prompt,
        original_message_summary_builder=original_message_summary_builder,
        partial_summary_builder=partial_summary_builder,
        max_characters_in_prompt=cfg.max_characters_in_prompt,
        verbose=cfg.verbose,
    )

    # We expect to have as many summaries as splits
    message_split_summaries: list[Summary] = original_split_summarization(
        message_splits
    )
    _dump_summaries(
        summary_collection=message_split_summaries,
        summary_handlers=cfg.summary_per_split_handlers,
    )

    # We expect to have only summary in the list summarizing split summaries
    overall_summary: list[Summary] = overall_summarization(
        [to_message_collection(message_split_summaries)]
    )

    _dump_summaries(
        summary_collection=overall_summary,
        summary_handlers=cfg.overall_summary_handlers,
    )


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(run_summarize_posts)()
