# kygs
LLM-powered opinion grabber

## Getting started

1. Create a virtual environment, e.g.
```bash
conda create -n kygs python=3.11
conda activate kygs
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

Creates a timestamped JSON file (`reddit_posts_YYYYMMDD_HHMMSS.json`) in the result directory containing:
- Collection metadata (time, subreddits, time window)
- List of posts with title, author, URL, text content, and score

### `train_lightweight_text_classifier.py`

Trains a fast TF-IDF + Naive Bayes classifier using spaCy lemmatization (default `ru_core_news_sm`).
Run it with `python kygs/scripts/train_lightweight_text_classifier.py` and configure parameters via `config/config_train_lightweight_text_classifier.yaml` (message providers, labels, vectorizer/model settings).
Ensure you have the spaCy model installed with `python -m spacy download ru_core_news_sm` before running the script.

Generates CSVs with per-sample predictions for both train and test datasets inside `${user_settings.hydra_dir}`.

### `train_text_classifier.py`

Uses text embeddings with an MLP classifier for higher-capacity models. Configure via `config/config_train_text_classifier.yaml` to control embeddings, labels, and message providers.

### `annotate_posts_manually.py`

Allows manual annotation/labeling of previously collected Reddit posts. Displays posts one by one and prompts for label selection from predefined options.

#### Configuration

1. In `config_annotate_posts_manually.yaml`, modify:
   - `reddit.input_file`: path to the JSON file containing collected posts
   - `reddit.labels`: list of available labels for annotation (e.g., 'gpt5', 'gpt4o', 'gpt-oss', etc.)
   - `reddit.annotated_dataset_path`: where to save the annotated dataset

#### Output

Creates a JSON file at the specified output path containing:
- Original collection metadata
- Annotation metadata (time, available labels)
- List of posts with all original fields plus added 'label' field

