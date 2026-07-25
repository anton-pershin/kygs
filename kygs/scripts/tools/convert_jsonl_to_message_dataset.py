import json
import logging
from datetime import datetime

import hydra
import pandas as pd
from omegaconf import DictConfig

from kygs.utils.common import get_config_path

CONFIG_NAME = "config_convert_jsonl_to_message_dataset"


def convert_jsonl_to_message_dataset(cfg: DictConfig) -> None:
    rows: list[dict[str, str | int]] = []

    with open(cfg.input_dataset_path, "r", encoding="utf-8") as f:
        for l in f:
            synth_msg: dict[str, str] = json.loads(l)
            text = synth_msg["text"]
            rows.append(
                {
                    "id": 0,
                    "offset": 0,
                    "text": text,
                    "sender_chat": "synth",
                    "date": datetime.now().isoformat(sep=" ", timespec="seconds"),
                    "n_views": 0,
                    "type": "text",
                    "label": cfg.label,
                }
            )

    pd.DataFrame(rows).to_csv(cfg.output_dataset_path, index=False)

    logging.info("JSONL to CSV conversion:")
    logging.info("JSONL: %s", cfg.input_dataset_path)
    logging.info("CSV: %s", cfg.output_dataset_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(convert_jsonl_to_message_dataset)()
