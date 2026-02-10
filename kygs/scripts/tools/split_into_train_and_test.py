from pathlib import Path
import json
import datetime
import random

import pandas as pd
from omegaconf import DictConfig

import hydra
from kygs.utils.common import get_config_path

CONFIG_NAME = "config_split_into_train_and_test"


def split_into_train_and_test(cfg: DictConfig) -> None:
   df = pd.read_csv(cfg.input_dataset_path)
   shuffled_indices = list(df.index)
   random.shuffle(shuffled_indices)

   n_test = int(cfg.test_ratio * df.shape[0])
   df_test = df.loc[shuffled_indices[:n_test]]
   df_train = df.loc[shuffled_indices[n_test:]]

   # Save two separate datasets
   df_train.to_csv(cfg.output_train_dataset_path, index=False)
   df_test.to_csv(cfg.output_test_dataset_path, index=False)

   print(f"Original dataset: {cfg.input_dataset_path}")
   print(f"Train dataset: {cfg.output_train_dataset_path}")
   print(f"Test dataset: {cfg.output_test_dataset_path}")


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(split_into_train_and_test)()
