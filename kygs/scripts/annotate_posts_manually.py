from pathlib import Path
from datetime import datetime, UTC
import json

import hydra
from omegaconf import DictConfig

from kygs.utils.common import get_config_path
from kygs.utils.console import console
from kygs.annotation.manual import MessageForAnnotation, annotate_items


CONFIG_NAME = "config_annotate_posts_manually"


def annotate_posts_manually(cfg: DictConfig) -> None:
    # Load collected posts
    with open(cfg.reddit.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert posts to MessageForAnnotation objects
    messages = [
        MessageForAnnotation(
            text=post['selftext'],
            author=post['author'],
            title=post['title'],
            source=post['subreddit'],
            url=post['url'],
            score=post['score'],
            time=datetime.fromtimestamp(int(post['created_utc']), UTC)
        ) for post in data['posts']
    ]

    # Get annotations
    labels = annotate_items(messages, cfg.reddit.labels)

    # Update original posts with labels
    for post, label in zip(data['posts'], labels):
        post['label'] = label

    # Handle metadata and saving
    output_path = Path(cfg.reddit.annotated_dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = data['metadata']
    metadata.update({
        'annotation_time': datetime.now().isoformat(),
        'available_labels': list(cfg.reddit.labels)
    })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': metadata,
            'posts': data['posts']
        }, f, indent=2, ensure_ascii=False)
    
    console.print(f"Saved {len(labels)} annotated posts to {output_path}")


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(annotate_posts_manually)()

