from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

from kygs.utils.console import console

MODEL_FILENAME = "model.pkl"
VECTORIZER_FILENAME = "tfidf.pkl"
METADATA_FILENAME = "metadata.json"


class SpacyLemmaTokenizer:
    def __init__(
        self,
        model_name: str,
        keep_pos: Sequence[str] | None = None,
        lowercase: bool = True,
        strip_punct: bool = True,
        strip_numeric: bool = True,
        lemmatize: bool = True,
    ) -> None:
        import spacy

        self.model_name = model_name
        self.keep_pos = tuple(keep_pos) if keep_pos else None
        self.lowercase = lowercase
        self.strip_punct = strip_punct
        self.strip_numeric = strip_numeric
        self.lemmatize = lemmatize

        try:
            self.nlp = spacy.load(self.model_name, disable=("ner", "textcat"))
        except OSError as exc:
            raise RuntimeError(
                "spaCy model 'ru_core_news_sm' is required. Install it via\n"
                "    python -m spacy download ru_core_news_sm"
            ) from exc

    def __call__(self, text: str) -> list[str]:
        processed_text = text
        if self.lowercase:
            processed_text = processed_text.lower()

        doc = self.nlp(processed_text)
        tokens: list[str] = []
        for token in doc:
            if token.is_space:
                continue
            if self.strip_punct and token.is_punct:
                continue
            if self.strip_numeric and token.like_num:
                continue
            if token.is_stop:
                continue
            if self.keep_pos and token.pos_ not in self.keep_pos:
                continue

            lemma = token.lemma_ if self.lemmatize else token.text
            lemma = lemma.strip()
            if lemma:
                tokens.append(lemma)

        return tokens



class LightweightTextClassifier:
    def __init__(
        self,
        vectorizer: TfidfVectorizer,
        classifier: MultinomialNB,
        labels: list[str],
        model_path: str,
    ) -> None:
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.labels = labels
        self.model_path = Path(model_path)

    def fit(self, texts: Sequence[str], targets: Sequence[int]) -> None:
        x_train = self.vectorizer.fit_transform(texts)
        breakpoint()
        self.classifier.fit(x_train, targets)

    def predict(self, texts: Sequence[str]) -> list[int]:
        x_data = self.vectorizer.transform(texts)
        return self.classifier.predict(x_data).tolist()

    def print_classification_report(
        self,
        title: str,
        texts: Sequence[str],
        targets: Sequence[int],
    ) -> None:
        predictions = self.predict(texts)
        console.print()
        console.print(f"[bold]{title.upper()} CLASSIFICATION REPORT[/bold]")
        console.print(classification_report(targets, predictions, target_names=self.labels))

    def save(self) -> None:
        self.model_path.mkdir(parents=True, exist_ok=False)
        with open(self.model_path / VECTORIZER_FILENAME, "wb") as f:
            joblib.dump(self.vectorizer, f)

        with open(self.model_path / MODEL_FILENAME, "wb") as f:
            joblib.dump(self.classifier, f)

        metadata = {
            "labels": self.labels,
            "vectorizer_params": self.vectorizer.get_params(deep=True),
            "classifier_params": self.classifier.get_params(deep=True),
            "model_path": str(self.model_path),
        }
        with open(self.model_path / METADATA_FILENAME, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, model_path: str) -> LightweightTextClassifier:
        model_dir = Path(model_path)
        with open(model_dir / VECTORIZER_FILENAME, "rb") as f:
            vectorizer = joblib.load(f)

        with open(model_dir / MODEL_FILENAME, "rb") as f:
            classifier = joblib.load(f)

        with open(model_dir / METADATA_FILENAME, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return cls(
            vectorizer=vectorizer,
            classifier=classifier,
            labels=metadata["labels"],
            model_path=model_path,
        )
