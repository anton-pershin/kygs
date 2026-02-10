from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from functools import reduce
from typing import Optional

import pandas as pd
from rich.markdown import Markdown
from rich.panel import Panel

from kygs.utils.console import console
from kygs.utils.time import datetime_ceil, datetime_floor, increment_datetime
from kygs.utils.typing import TimeUnit


@dataclass
class Message:
    text: str
    time: datetime.datetime
    author: str
    label: Optional[str]
    true_label: Optional[str]
    title: Optional[str] = None
    source: Optional[str] = None
    url: Optional[str] = None
    score: Optional[int] = None


@dataclass
class MessageCollection:
    messages: list[Message]
    start_dt: datetime.datetime
    end_dt: datetime.datetime


class MessageProvider:
    def __init__(self, messages: list[Message]) -> None:
        self.messages = messages

    @classmethod
    def create_empty(cls) -> MessageProvider:
        return cls([])

    @classmethod
    def from_synthetic_messages_json(cls, json_path: str) -> MessageProvider:
        with open(json_path, "r", encoding="utf-8") as f:
            d = json.load(f)

        messages = []
        for m in d["messages"]:
            text = m["text"]
            time = datetime.datetime.now()
            author = "synthetic"
            true_label: str | None = None
            if "label" in m:
                true_label = m["label"]

            msg = Message(text, time, author, label=None, true_label=true_label)
            messages.append(msg)

        return cls(messages)

    @classmethod
    def from_telegram_messages_json(cls, json_path: str) -> MessageProvider:
        with open(json_path, "r", encoding="utf-8") as f:
            d = json.load(f)

        messages = []
        for m in d["messages"]:
            if "action" in m:
                continue

            text = "".join([te["text"] for te in m["text_entities"]])
            time = datetime.datetime.strptime(m["date"], "%Y-%m-%dT%H:%M:%S")
            author = m["from"]
            msg = Message(text, time, author, label=None, true_label=None)
            messages.append(msg)

        return cls(messages)

    @classmethod
    def from_telegram_messages_csv(cls, csv_path: str) -> MessageProvider:
        df = pd.read_csv(csv_path)

        messages = []
        for _, row in df.iterrows():
            text = row["text"]
            if not text:
                raise ValueError("No text in message")

            time = datetime.datetime.fromisoformat(row["date"])
            author = row["sender_chat"]
            label = row["label"]
            msg = Message(text, time, author, label=label, true_label=None)
            messages.append(msg)

        return cls(messages)

    @classmethod
    def from_reddit_posts_json(
        cls, json_path: str, stores_true_labels: bool
    ) -> MessageProvider:
        with open(json_path, "r", encoding="utf-8") as f:
            d = json.load(f)

        messages = []
        for m in d["posts"]:
            # Skip empty messages (typically, images/video with title only)
            text = m["selftext"]
            if not text:
                continue

            time = datetime.datetime.utcfromtimestamp(int(m["created_utc"]))
            author = m["author"]
            label = m["label"] if "label" in m else None

            if stores_true_labels:
                msg = Message(text, time, author, label=None, true_label=label)
            else:
                msg = Message(text, time, author, label=label, true_label=None)

            messages.append(msg)

        return cls(messages)

    @classmethod
    def from_reddit_posts_csv(
        cls, csv_path: str, stores_true_labels: bool
    ) -> MessageProvider:
        df = pd.read_csv(csv_path)

        messages = []
        for _, row in df.iterrows():
            # Skip empty messages (typically, images/video with title only)
            text = row["messages"]
            if not text:
                continue

            time = datetime.datetime.now()
            author = "Unknown"
            label = row["labels"]

            if stores_true_labels:
                msg = Message(text, time, author, label=None, true_label=label)
            else:
                msg = Message(text, time, author, label=label, true_label=None)

            messages.append(msg)

        return cls(messages)

    @classmethod
    def from_message_providers(cls, *mps: MessageProvider) -> MessageProvider:
        messages: list[Message] = reduce(
            lambda x, y: x + y,
            [mp.messages for mp in mps],
            [],
        )
        return cls(messages)

    def append_messages(self, other_mp: MessageProvider):
        self.messages.extend(other_mp.messages)

    def display_messages(self) -> None:
        for i, message in enumerate(self.messages, 1):
            console.print(f"Item {i} of {len(self.messages)}")
            content = f"## {message.author}\n\n"
            content += f"{message.text}\n\n"

            panel = Panel(
                Markdown(content),
                title=f"[bold]{message.time}[/bold]",
                subtitle=f"[bold]Label: {message.label}[/bold]",
            )
            console.print(panel)
            console.print()

    def times(self) -> list[datetime.datetime]:
        return [m.time for m in self.messages]

    def filter(self, start: datetime.datetime, end: datetime.datetime) -> list[Message]:
        return [m for m in self.messages if start <= m.time <= end]

    def split_by(self, time_unit: TimeUnit) -> list[MessageCollection]:
        times = self.times()
        times.sort()
        start_dt = datetime_floor(times[0], time_unit)
        end_dt = datetime_ceil(times[-1], time_unit)

        splits = []
        split_start_dt = start_dt
        split_end_dt = increment_datetime(start_dt, time_unit)

        # Cover the case where times contain the time interval
        # shorter than time_unit
        end_dt = max(end_dt, split_end_dt)

        i = 0
        while split_start_dt < end_dt:
            split = self.filter(split_start_dt, split_end_dt)
            splits.append(
                MessageCollection(
                    messages=split,
                    start_dt=split_start_dt,
                    end_dt=split_end_dt,
                )
            )

            i += 1
            split_start_dt = increment_datetime(start_dt, time_unit, amount=i)
            split_end_dt = increment_datetime(start_dt, time_unit, amount=i + 1)

        return splits
