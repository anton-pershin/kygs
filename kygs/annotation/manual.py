from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
from datetime import datetime

from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt

from kygs.utils.console import console


@dataclass
class MessageForAnnotation:
    text: str
    author: str
    title: Optional[str]
    source: str
    url: Optional[str]
    score: int
    time: datetime
    label: Optional[str] = None


def display_message(message: MessageForAnnotation) -> None:
    if message.title:
        content = f"## {message.title}\n\n"
    else:
        content = f"## Posted on {message.time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    content += f"{message.text}\n\n"
    content += f"*Posted by {message.author} in {message.source}*"
    
    panel = Panel(
        Markdown(content),
        title=f"Score: {message.score}",
        subtitle=f"URL: {message.url}" if message.url else None
    )
    console.print(panel)
    console.print()


def get_valid_label(labels: List[str]) -> str:
    choices = {str(i): label for i, label in enumerate(labels, 1)}
    
    console.print("Available labels:")
    for number, label in choices.items():
        console.print(f"  {number}. {label}")
    console.print()

    choice = Prompt.ask(
        "Choose label",
        choices=choices.keys(),
        show_choices=False
    )
    return choices[choice]


def annotate_items(messages: List[MessageForAnnotation], labels: List[str]) -> List[str]:
    console.print(f"Loaded {len(messages)} items for annotation")
    console.print()
    
    annotations = []
    for i, message in enumerate(messages, 1):
        console.print(f"Item {i} of {len(messages)}")
        display_message(message)
        
        label = get_valid_label(labels)
        annotations.append(label)
        
        console.print()
    
    return annotations
