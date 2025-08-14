import datetime
import json

import hydra
from omegaconf import DictConfig
import pandas as pd

from kygs.summarization import Summary, summarize_posts, summarize_summaries
from kygs.utils.common import get_config_path
from kygs.utils.report import CsvReport 


CONFIG_NAME = "config_summarize_posts"


def _dump_summaries(summary_collection: list[Summary], summaries_path: str) -> None:
    summarized_messages_report = CsvReport(summaries_path)
    summarized_messages_report.add_columns(
        summarized_messages=[s.text for s in summary_collection],
        start_time=[s.start_dt.strftime("%Y-%m-%d %H:%M:%S") for s in summary_collection],
        end_time=[s.end_dt.strftime("%Y-%m-%d %H:%M:%S") for s in summary_collection],
    )
    summarized_messages_report.dump()



def run_summarize_posts(cfg: DictConfig) -> None:
    mp = hydra.utils.call(cfg.message_provider)
    llm = hydra.utils.instantiate(cfg.llm)

    # Summarize within the specified time interval
    summary_collection: list[Summary] = summarize_posts(
        mp=mp,
        summarization_time_interval=cfg.original_post_summarization.time_interval,
        llm=llm,
        system_prompt=cfg.system_prompt,
        user_prompt_template=cfg.original_post_summarization.user_prompt_template,
        verbose=True,
    )

    _dump_summaries(
        summary_collection=summary_collection,
        summaries_path=cfg.original_post_summarization.output.summaries_path,
    )

    # Prepare overall summary by recursive summarization
    while len(summary_collection) > 1:
        summary_collection: list[Summary] = summarize_summaries(
            summary_collection=summary_collection,
            llm=llm,
            system_prompt=cfg.system_prompt,
            user_prompt_template=cfg.recursive_summarization.user_prompt_template,
            max_characters_in_prompt=cfg.recursive_summarization.max_characters_in_prompt,
            verbose=True,
        )
    
    _dump_summaries(
        summary_collection=summary_collection,
        summaries_path=cfg.recursive_summarization.output.summary_path,
    )


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(run_summarize_posts)()

