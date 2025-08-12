import datetime
import json

import hydra
from omegaconf import DictConfig
import pandas as pd

from kygs.summarization import Summary, summarize_posts
from kygs.utils.common import get_config_path
from kygs.utils.report import CsvReport 


CONFIG_NAME = "config_summarize_posts"


def run_summarize_posts(cfg: DictConfig) -> None:
    mp = hydra.utils.call(cfg.message_provider)
    llm = hydra.utils.instantiate(cfg.llm)

    summary_collection: list[Summary] = summarize_posts(
        mp=mp,
        summarization_time_interval=cfg.summarization_time_interval,
        llm=llm,
        system_prompt=cfg.system_prompt,
        user_prompt_template=cfg.user_prompt_template,
    )

    summarized_messages_report = CsvReport(cfg.output.summarized_messages_path)
    summarized_messages_report.add_columns(
        summarized_messages=[s.text for s in summary_collection],
        start_time=[s.start_dt.strftime("%Y-%m-%d %H:%M:%S") for s in summary_collection],
        end_time=[s.end_dt.strftime("%Y-%m-%d %H:%M:%S") for s in summary_collection],
    )
    summarized_messages_report.dump()


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(run_summarize_posts)()

