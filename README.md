# kygs
LLM-powered opinion grabber

## Getting started

1. Create a virtual environment, e.g.
```bash
conda create -n infoscout python=3.11
conda activate infoscout
```
2. Install necessary packages
```bash
pip install -r requirements.txt
```
3. Set up `/config/user_settings/user_settings.yaml`
4. Run one of the scripts `/kygs/scripts/XXX.py` and do not forget to modify the corresponding config file in `/config/config_XXX.yaml'
```bash
python kygs/scripts/XXX.py
```

⚠️  DO NOT commit your `user_settings.yaml`

## Scripts

### `collect_recent_posts.py`

Collects recent posts from specified subreddits within a given time window. Posts are saved in JSON format with metadata about the collection.

#### Configuration

1. In `user_settings.yaml`, set up your Reddit API credentials:
   ```yaml
   reddit:
     client_id: YOUR_CLIENT_ID
     client_secret: YOUR_CLIENT_SECRET
     user_agent: YOUR_USER_AGENT
   ```

2. In `config_collect_recent_posts.yaml`, modify:
   - `reddit.subreddits`: list of subreddits to monitor
   - `reddit.time_window`: time window for "recent" posts (uses `timedelta` format, e.g., `hours: 24`)

#### Output

Creates a timestamped JSON file (`reddit_posts_YYYYMMDD_HHMMSS.json`) in the result directory containing:
- Collection metadata (time, subreddits, time window)
- List of posts with title, author, URL, text content, and score

