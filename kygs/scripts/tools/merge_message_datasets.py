from pathlib import Path
import json
import datetime

import pandas as pd
from omegaconf import DictConfig

import hydra
from kygs.utils.common import get_config_path

CONFIG_NAME = "config_merge_message_datasets"


def merge_message_datasets(cfg: DictConfig) -> None:
    df_collection = []
    for csv_filename in cfg.message_datasets_to_merge:
        df = pd.read_csv(csv_filename)
        df_collection.append(df)

    df_merged = pd.concat(df_collection, ignore_index=True)

    if cfg.drop_text_duplicates:
        df_merged.drop_duplicates(subset="text", keep="first", inplace=True)

    df_merged.to_csv(cfg.output_dataset_path, index=False)

    print("Merged datasets:")
    for csv_filename in cfg.message_datasets_to_merge:
        print(f"\t{csv_filename}")

    print()
    print(f"Output dataset: {cfg.output_dataset_path}")


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(merge_message_datasets)()

