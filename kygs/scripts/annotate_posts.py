from pathlib import Path
from datetime import datetime
import json

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt

from kygs.utils.common import get_config_path
from kygs.utils.console import console, prompt_user


CONFIG_NAME = "config_annotate_posts"


def display_post(post: dict) -> None:
    content = f"## {post['title']}\n\n"
    if post['selftext']:
        content += f"{post['selftext']}\n\n"
    content += f"*Posted by u/{post['author']} in r/{post['subreddit']}*"
    
    panel = Panel(
        Markdown(content),
        title=f"Score: {post['score']}",
        subtitle=f"URL: {post['url']}"
    )
    console.print(panel)
    console.print()


def get_valid_label(labels: list[str]) -> str:
    """Prompt user for a valid label using numbered options."""
    choices = {str(i): label for i, label in enumerate(labels, 1)}
    
    # Display available labels first
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


def annotate_posts(cfg: DictConfig) -> None:
    # Load collected posts
    with open(cfg.reddit.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    posts = data['posts']
    metadata = data['metadata']
    
    console.print(f"Loaded {len(posts)} posts for annotation")
    console.print()
    
    # Annotate posts
    annotated_posts = []
    for i, post in enumerate(posts, 1):
        console.print(f"Post {i} of {len(posts)}")
        display_post(post)
        
        label = get_valid_label(cfg.reddit.labels)
        post['label'] = label
        annotated_posts.append(post)
        
        console.print()
    
    # Save annotated dataset
    output_file = Path(cfg.reddit.annotated_dataset_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Update metadata
    metadata.update({
        'annotation_time': datetime.now().isoformat(),
        'available_labels': list(cfg.reddit.labels)
    })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': metadata,
            'posts': annotated_posts
        }, f, indent=2, ensure_ascii=False)
    
    console.print(f"Saved {len(annotated_posts)} annotated posts to {output_file}")


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(annotate_posts)()

