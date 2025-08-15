from __future__ import annotations
from dataclasses import dataclass
import json
import datetime
import math

from rich.panel import Panel
from rich.markdown import Markdown

from kygs.utils.console import console
from kygs.utils.typing import TimeUnit
from kygs.utils.time import (
    datetime_floor,
    datetime_ceil,
    increment_datetime,
)

@dataclass
class Message:
    text: str
    time: datetime.datetime
    author: str


@dataclass
class MessageCollection:
    messages: list[Message]
    start_dt: datetime.datetime
    end_dt: datetime.datetime


class MessageProvider:
    def __init__(self, messages: list[Message]) -> None:
        self.messages = messages

    @classmethod
    def from_telegram_messages_json(cls, json_path: str) -> MessageProvider:
        with open(json_path, "r") as f:
            d = json.load(f)

        messages = []
        for m in d["posts"]:
            if "action" in m:
                continue

            text = "".join([te["text"] for te in m["text_entities"]])
            time = datetime.datetime.strptime(m["date"], "%Y-%m-%dT%H:%M:%S")
            author = m["from"]
            msg = Message(text, time, author)
            messages.append(msg)

        return cls(messages)
        
    @classmethod
    def from_reddit_posts_json(cls, json_path:str) -> MessageProvider:
        with open(json_path, "r") as f:
            d = json.load(f)

        messages = []
        for m in d["posts"]:
            # Skip empty messages (typically, images/video with title only)
            text = m["selftext"]
            if not text: 
                continue

            time = datetime.datetime.utcfromtimestamp(int(m["created_utc"]))
            author = m["author"]
            msg = Message(text, time, author)
            messages.append(msg)

        return cls(messages)

    def display_messages(self) -> None:
        for i, message in enumerate(self.messages, 1):
            console.print(f"Item {i} of {len(self.messages)}")
            content = f"## {message.author}\n\n"
            content += f"{message.text}\n\n"
        
            panel = Panel(
                Markdown(content),
                title=f"[bold]{message.time}[/bold]",
                subtitle=None,
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

