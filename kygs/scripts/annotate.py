import hydra
from omegaconf import DictConfig

from kygs.annotation.base import MessageAnnotation
from kygs.utils.common import get_config_path
from kygs.utils.console import console
from kygs.utils.report import CsvReport

CONFIG_NAME = "config_annotate"


def annotate(cfg: DictConfig) -> None:
    mp = hydra.utils.call(cfg.message_provider)

    # Instantiate annotator from config
    annotator: MessageAnnotation = hydra.utils.instantiate(cfg.annotator)

    # Get annotations
    labels = annotator(mp.messages, dict(cfg.labels))

    # Dump 'em
    annotated_dataset_csv = CsvReport(cfg.annotated_dataset_path)
    annotated_dataset_csv.add_columns(
        messages=[m.text for m in mp.messages],
        labels=labels,
    )
    annotated_dataset_csv.dump()

    console.print(
        f"Saved {len(labels)} annotated messages to {cfg.annotated_dataset_path}"
    )


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(annotate)()
