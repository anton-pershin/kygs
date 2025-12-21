from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from kygs.annotation.base import MessageAnnotation
from kygs.message_provider import Message
from kygs.utils.console import console


def display_message(message: Message) -> None:
    if message.title:
        content = f"## {message.title}\n\n"
    else:
        content = f"## Posted on {message.time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    content += f"{message.text}\n\n"
    content += f"*Posted by {message.author} in {message.source}*"

    panel = Panel(
        Markdown(content),
        title=f"Score: {message.score}",
        subtitle=f"URL: {message.url}" if message.url else None,
    )
    console.print(panel)
    console.print()


def get_valid_label(labels: dict[str, str]) -> str:
    choices = {
        str(i): (name, descr) for i, (name, descr) in enumerate(labels.items(), 1)
    }

    console.print("Available labels:")
    for number, (name, descr) in choices.items():
        console.print(f"  {number}. **{name}**. {descr}")

    console.print()

    choice = Prompt.ask(
        "Choose label",
        choices=choices.keys(),
        show_choices=False,
    )  # type: ignore
    return choices[choice][0]


class ManualAnnotation(MessageAnnotation):
    def __call__(
        self, messages: list[Message], labels: dict[str, str]
    ) -> list[str | None]:
        console.print(f"Loaded {len(messages)} items for annotation")
        console.print()

        annotations: list[str | None] = []
        for i, message in enumerate(messages, 1):
            console.print(f"Item {i} of {len(messages)}")
            display_message(message)

            label = get_valid_label(labels)
            annotations.append(label)

            console.print()

        return annotations
