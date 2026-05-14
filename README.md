# kygs

LLM-powered tools for natural text processing including text classification, clustering, annotation, summarization and similarity/relevance analysis.

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

## Subpackages

- [Summarization](docs/summarization.md): `kygs.summarization`
- Annotation: `kygs.annotation`
- Clustering: `kygs.clustering`
- Relevance analysis: `kygs.relevance`

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

### `score_relevance.py`

Computes pairwise relevance scores between query and document message sets. Configure via `config/config_score_relevance.yaml` to choose message providers, embedding backend, and scoring strategy (currently cosine similarity over embeddings). Results are saved as a JSON payload containing per-query ranked document IDs with scores.

### `tools/print_relevant_messages.py`

Loads two message providers (for queries and documents) along with a JSON relevance file and prints human-readable summaries of the top matches. Configure via `config/config_print_relevant_messages.yaml`.

### `summarize_posts.py`

Summarizes collections of messages using recursive (map-reduce) LLM summarization. Messages are first split into groups via a configurable `SplitStrategy`, then each group is summarized independently. Finally, all per-split summaries are combined into an overall summary. Configure via `config/config_summarize_posts.yaml` to control the LLM, message provider, split strategy, summarization prompts, summary builders, and prompt size limit (`max_characters_in_prompt`).

#### Output

Two JSON files in the Hydra output directory:
- `summarized_messages_per_split.json` — one summary per message split (e.g., per time period or label)
- `summarized_messages_overall.json` — a single summary synthesizing all per-split summaries

### `summarize_and_annotate_posts.py`

Summarizes collections of messages and assigns multilabel annotations to each collection in a single LLM pass. Uses the same recursive map-reduce architecture as `summarize_posts.py`, but the first-pass prompt instructs the LLM to return a JSON response containing both a summary and a list of labels from configurable annotation classes. Subsequent recursive passes use plain summarization without annotation. Configure via `config/config_summarize_and_annotate_posts.yaml` to control the LLM, message provider, split strategy, annotation labels, summarization prompts, summary builders, and prompt size limit.

#### Output

Two JSON files in the Hydra output directory:
- `summarized_and_annotated_messages_per_split.json` — one summary per message split, each with an `annotation_labels` field in metadata containing the assigned multilabel annotations (configurable via `metadata_key` parameter in `config/summary_builder/annotated.yaml`)
- `summarized_and_annotated_messages_overall.json` — a single overall summary

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

