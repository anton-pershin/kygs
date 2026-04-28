import hydra
from omegaconf import DictConfig

from kygs.split_strategy import SplitStrategy
from kygs.summarization import Summary, summarize_posts, summarize_summaries
from kygs.utils.common import get_config_path

CONFIG_NAME = "config_summarize_posts"


def _dump_summaries(summary_collection: list[Summary], cfg: DictConfig) -> None:
    for handler_cfg in cfg.summary_handlers:
        handler = hydra.utils.instantiate(handler_cfg)
        handler.handle(summary_collection)


def run_summarize_posts(cfg: DictConfig) -> None:
    mp = hydra.utils.call(cfg.message_provider)
    llm = hydra.utils.instantiate(cfg.llm)
    split_strategy: SplitStrategy = hydra.utils.instantiate(cfg.split_strategy)

    message_splits = split_strategy.split(mp.messages)

    summary_collection: list[Summary] = summarize_posts(
        message_splits=message_splits,
        llm=llm,
        system_prompt=cfg.system_prompt,
        user_prompt_template=cfg.original_post_summarization.user_prompt_template,
        verbose=True,
    )

    _dump_summaries(
        summary_collection=summary_collection,
        cfg=cfg.original_post_summarization,
    )

    max_chars_in_prompt = cfg.recursive_summarization.max_characters_in_prompt
    while len(summary_collection) > 1:
        summary_collection = summarize_summaries(
            summary_collection=summary_collection,
            llm=llm,
            system_prompt=cfg.system_prompt,
            user_prompt_template=cfg.recursive_summarization.user_prompt_template,
            max_characters_in_prompt=max_chars_in_prompt,
            verbose=True,
        )

    _dump_summaries(
        summary_collection=summary_collection,
        cfg=cfg.recursive_summarization,
    )


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(run_summarize_posts)()
