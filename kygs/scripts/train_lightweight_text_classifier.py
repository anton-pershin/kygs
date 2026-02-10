from __future__ import annotations

from pathlib import Path

import hydra
import hydra.utils
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rich.table import Table
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from kygs.lightweight_classifier import LightweightTextClassifier, SpacyLemmaTokenizer
from kygs.message_provider import MessageProvider
from kygs.utils.common import get_config_path, set_cuda_visible_devices
from kygs.utils.console import console

CONFIG_NAME = "config_train_lightweight_text_classifier"


def ensure_labels_are_valid(mp: MessageProvider, labels: list[str]) -> None:
    for i, m in enumerate(mp.messages):
        assert m.label is not None, f"Label is None for message #{i}"
        assert m.label in labels, f"Label {m.label} not found in the list of labels"


def print_dataset_stats(name: str, mp: MessageProvider, labels: list[str]) -> None:
    total_samples = len(mp.messages)
    table = Table(title=f"{name} dataset stats")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total samples", str(total_samples))
    for label in labels:
        count = sum(1 for m in mp.messages if m.label == label)
        table.add_row(f"Label '{label}'", str(count))

    console.print(table)
    console.print()


def get_texts_and_labels(mp: MessageProvider, labels: list[str]) -> tuple[list[str], list[int]]:
    texts = [m.text for m in mp.messages]
    y = np.array([labels.index(m.label) for m in mp.messages], dtype=np.int32)
    return texts, y.tolist()


def save_prediction_details(
    texts: list[str],
    y_true: list[int],
    y_pred: list[int],
    file_path: str | Path,
) -> None:
    df = pd.DataFrame({"text": texts, "y_true": y_true, "y_pred": y_pred})
    df.to_csv(file_path, index=False)


def train_lightweight_text_classifier(cfg: DictConfig) -> None:
    set_cuda_visible_devices(cfg.cuda_visible_devices)

    train_mp: MessageProvider = hydra.utils.call(cfg.train_message_provider)
    test_mp: MessageProvider = hydra.utils.call(cfg.test_message_provider)

    labels = list(cfg.labels)
    ensure_labels_are_valid(train_mp, labels)
    ensure_labels_are_valid(test_mp, labels)

    if cfg.verbose:
        print_dataset_stats("Train", train_mp, labels)
        print_dataset_stats("Test", test_mp, labels)

    tokenizer = SpacyLemmaTokenizer(
        model_name=cfg.spacy_model,
        keep_pos=cfg.vectorizer.get("keep_pos"),
        lowercase=cfg.vectorizer.get("lowercase", True),
        strip_punct=cfg.vectorizer.get("strip_punct", True),
        strip_numeric=cfg.vectorizer.get("strip_numeric", True),
        lemmatize=cfg.vectorizer.get("lemmatize", True),
    )

    vectorizer_params = {
        k: v
        for k, v in cfg.vectorizer.items()
        if k
        not in {
            "keep_pos",
            "lowercase",
            "strip_punct",
            "strip_numeric",
            "lemmatize",
        }
    }
    # For some reason, TfidfVectorizer expects a tuple
    vectorizer_params["ngram_range"] = tuple(vectorizer_params["ngram_range"])

    vectorizer = TfidfVectorizer(tokenizer=tokenizer, **vectorizer_params)
    classifier = MultinomialNB(**cfg.classifier)

    x_train, y_train = get_texts_and_labels(train_mp, labels)
    x_test, y_test = get_texts_and_labels(test_mp, labels)

    lightweight_classifier = LightweightTextClassifier(
        vectorizer=vectorizer,
        classifier=classifier,
        labels=labels,
        model_path=str(cfg.model_path),
    )

    lightweight_classifier.fit(x_train, y_train)
    lightweight_classifier.print_classification_report("train", x_train, y_train)
    lightweight_classifier.print_classification_report("test", x_test, y_test)

    lightweight_classifier.save()

    y_train_pred = lightweight_classifier.predict(x_train)
    save_prediction_details(
        texts=x_train,
        y_true=y_train,
        y_pred=y_train_pred,
        file_path=Path(cfg.result_dir) / "train_lightweight_prediction_details.csv",
    )

    y_test_pred = lightweight_classifier.predict(x_test)
    save_prediction_details(
        texts=x_test,
        y_true=y_test,
        y_pred=y_test_pred,
        file_path=Path(cfg.result_dir) / "test_lightweight_prediction_details.csv",
    )


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(train_lightweight_text_classifier)()
