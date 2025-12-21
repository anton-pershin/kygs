from __future__ import annotations

import json
import pickle
from pathlib import Path

from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

from kygs.utils.console import console
from kygs.utils.typing import NDArrayFloat, NDArrayInt

MODEL_FILENAME = "model.pkl"
METADATA_FILENAME = "metadata.json"


class TextClassifier:
    def __init__(
        self,
        model: MLPClassifier,
        hidden_layer_size: int,
        labels: list[str],
        model_path: str,
    ):
        self.model = model
        self.hidden_layer_size = hidden_layer_size
        self.labels = labels
        self.model_path = model_path

    def fit(self, x: NDArrayFloat, y: NDArrayInt) -> None:
        self.model.fit(x, y)

    def predict(self, x: NDArrayFloat) -> NDArrayInt:
        y_pred = self.model.predict(x)
        return y_pred

    def print_classification_report(
        self, title: str, x: NDArrayFloat, y_true: NDArrayInt
    ) -> None:
        y_pred = self.predict(x)

        console.print()
        console.print(f"[bold]{title.upper()} CLASSIFICATION REPORT[/bold]")
        console.print(classification_report(y_true, y_pred, target_names=self.labels))

    @classmethod
    def create_model(
        cls,
        hidden_layer_size: int,
        max_iter: int,
        labels: list[str],
        model_path: str,
    ) -> TextClassifier:
        model = MLPClassifier(
            hidden_layer_sizes=(hidden_layer_size,),
            max_iter=max_iter,
            random_state=1,
            verbose=True,
        )
        return cls(
            model=model,
            hidden_layer_size=hidden_layer_size,
            labels=labels,
            model_path=model_path,
        )

    @classmethod
    def load_model(cls, model_path: str) -> TextClassifier:
        p = Path(model_path)
        with open(p / MODEL_FILENAME, "rb") as f:
            model = pickle.load(f)

        with open(p / METADATA_FILENAME, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return cls(
            model=model,
            hidden_layer_size=metadata["hidden_layer_size"],
            labels=metadata["labels"],
            model_path=metadata["model_path"],
        )

    def save_model(self) -> None:
        p = Path(self.model_path)
        p.mkdir(parents=True, exist_ok=False)
        with open(p / MODEL_FILENAME, "wb") as f:
            pickle.dump(self.model, f, protocol=5)

        metadata = {
            "labels": self.labels,
            "hidden_layer_size": self.hidden_layer_size,
            "model_path": self.model_path,
        }
        with open(p / METADATA_FILENAME, "w", encoding="utf-8") as f:
            json.dump(metadata, f)
