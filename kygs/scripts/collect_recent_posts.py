from pathlib import Path
from datetime import datetime, timedelta, UTC
import json

import hydra
from hydra.utils import instantiate 
from omegaconf import DictConfig
import praw
from praw.models import Submission
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import track

from kygs.utils.common import get_config_path
from kygs.utils.console import console


CONFIG_NAME = "config_collect_recent_posts"


def collect_recent_posts(cfg: DictConfig) -> None:
    # Initialize Reddit API client
    reddit = praw.Reddit(
        client_id=cfg.user_settings.reddit.client_id,
        client_secret=cfg.user_settings.reddit.client_secret,
        user_agent=cfg.user_settings.reddit.user_agent
    )
    
    # Calculate the cutoff time
    time_window = timedelta(**cfg.reddit.time_window)
    cutoff_time = datetime.now(UTC) - time_window
    
    # Collect posts from each subreddit
    all_posts = []
    for subreddit_name in track(cfg.reddit.subreddits, description="Post collection"):
        subreddit = reddit.subreddit(subreddit_name)
        # Get new posts
        for post in subreddit.new(limit=None):
            post_time = datetime.fromtimestamp(post.created_utc, tz=UTC)
            if post_time < cutoff_time:
                break
            all_posts.append({
                'subreddit': subreddit_name,
                'title': post.title,
                'author': str(post.author),
                'created_utc': post.created_utc,
                'url': post.url,
                'selftext': post.selftext,
                'score': post.score
            })
    
    console.print()
    console.print(f"Collected {len(all_posts)} posts from {len(cfg.reddit.subreddits)} subreddits")
    console.print()
    
    # Save posts to JSON file
    result_dir = Path(cfg.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = result_dir / f"reddit_posts_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'collection_time': datetime.now().isoformat(),
                'subreddits': list(cfg.reddit.subreddits),
                'time_window_seconds': time_window.total_seconds(), 
            },
            'posts': all_posts
        }, f, indent=2, ensure_ascii=False)
    
    console.print(f"Saved {len(all_posts)} posts to {output_file}")
    console.print()


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(collect_recent_posts)()
