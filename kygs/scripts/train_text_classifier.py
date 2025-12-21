from pathlib import Path

import hydra
import hydra.utils
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rich.table import Table

from kygs.classifier import TextClassifier
from kygs.message_provider import MessageProvider
from kygs.text_embedding import TextEmbeddingModel
from kygs.utils.common import get_config_path, set_cuda_visible_devices
from kygs.utils.console import console
from kygs.utils.typing import NDArrayFloat, NDArrayInt

CONFIG_NAME = "config_train_text_classifier"


def print_dataset_stats(name: str, mp: MessageProvider, labels: list[str]) -> None:
    total_samples = len(mp.messages)

    # Count samples per label
    label_counts = {}
    for label in labels:
        count = sum(1 for m in mp.messages if m.label == label)
        percentage = (count / total_samples) * 100
        label_counts[label] = (count, percentage)

    # Create and populate table
    table = Table(title=f"{name} Dataset Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total samples", str(total_samples))
    for label, (count, percentage) in label_counts.items():
        table.add_row(f"Label '{label}'", f"{count} ({percentage:.1f}%)")

    console.print(table)
    console.print()


def get_text_sequences(mp: MessageProvider) -> list[str]:
    return [m.text for m in mp.messages]


def ensure_labels_are_valid(
    mp: MessageProvider,
    labels: list[str],
) -> None:
    for i, m in enumerate(mp.messages):
        assert m.label is not None, f"Label is None for message #{i}"
        assert (
            m.label in labels
        ), f"Label {m.label} not found in the list of available labels"


def prepare_xy(
    mp: MessageProvider,
    labels: list[str],
    text_embedding_model: TextEmbeddingModel,
) -> tuple[NDArrayFloat, NDArrayInt]:
    text_sequences = get_text_sequences(mp)
    text_embeddings = text_embedding_model.predict(text_sequences)
    numeric_labels = np.array(
        [labels.index(m.label) for m in mp.messages],  # type: ignore
        dtype=np.int32,
    )  # TODO: slow and clumsy wrt. type hints

    return text_embeddings, numeric_labels


def save_prediction_details(
    text_sequences: list[str],
    y_true: NDArrayInt,
    y_pred: NDArrayInt,
    file_path: str | Path,
):
    df = pd.DataFrame(
        {
            "text_sequences": text_sequences,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    df.to_csv(file_path, index=False)


def train_text_classifier(cfg: DictConfig) -> None:
    set_cuda_visible_devices(cfg.cuda_visible_devices)

    train_mp = hydra.utils.call(cfg.train_message_provider)
    test_mp = hydra.utils.call(cfg.test_message_provider)

    labels = list(cfg.labels)
    ensure_labels_are_valid(train_mp, labels)
    ensure_labels_are_valid(test_mp, labels)

    # Print dataset statistics if verbose is enabled
    if cfg.verbose:
        print_dataset_stats("Training", train_mp, labels)
        print_dataset_stats("Test", test_mp, labels)

    text_embedding_model = hydra.utils.instantiate(cfg.embedding)

    x_train, y_train = prepare_xy(
        mp=train_mp,
        labels=labels,
        text_embedding_model=text_embedding_model,
    )

    x_test, y_test = prepare_xy(
        mp=test_mp,
        labels=labels,
        text_embedding_model=text_embedding_model,
    )

    classifier = TextClassifier.create_model(
        hidden_layer_size=cfg.classifier.hidden_layer_size,
        max_iter=cfg.classifier.max_iter,
        model_path=cfg.classifier.model_path,
        labels=labels,
    )
    classifier.fit(x_train, y_train)
    classifier.print_classification_report(
        title="train",
        x=x_train,
        y_true=y_train,
    )
    classifier.print_classification_report(
        title="test",
        x=x_test,
        y_true=y_test,
    )

    classifier.save_model()

    y_train_pred = classifier.predict(x_train)
    save_prediction_details(
        text_sequences=get_text_sequences(train_mp),
        y_true=y_train,
        y_pred=y_train_pred,
        file_path=Path(cfg.result_dir) / "train_prediction_details.csv",
    )

    y_test_pred = classifier.predict(x_test)
    save_prediction_details(
        text_sequences=get_text_sequences(test_mp),
        y_true=y_test,
        y_pred=y_test_pred,
        file_path=Path(cfg.result_dir) / "test_prediction_details.csv",
    )


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(train_text_classifier)()
