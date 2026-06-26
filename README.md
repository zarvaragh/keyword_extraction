# keyword_extraction

Extract keywords from text using **TF-IDF** and **RAKE** — two classic NLP techniques — with word cloud and bar chart visualizations.

## Requirements

```bash
pip install scikit-learn pandas matplotlib seaborn wordcloud rake-nltk
```

## Dataset

Expects a CSV with `Title` and `Body` columns (e.g. a Stack Overflow questions dataset).

## Scripts

### `tf_idf.py` — TF-IDF Keyword Extraction

Uses scikit-learn's `TfidfTransformer` + `CountVectorizer`:

1. Loads and cleans 20 000 rows
2. Generates a word cloud of the corpus
3. Plots the top 20 uni-grams, bi-grams, and tri-grams
4. Fits a TF-IDF model on the training split
5. Extracts the top 10 keywords per document in the test split

### `rake.py` — RAKE Keyword Extraction

Uses `rake-nltk` to extract top keyword phrases from each document row.

## Visualizations

- Word cloud of the entire corpus
- Bar charts for most frequent uni-grams, bi-grams, and tri-grams

## Tech Stack

| Tool | Purpose |
|------|---------|
| scikit-learn | TF-IDF vectorization (`get_feature_names_out` — sklearn 1.2+) |
| rake-nltk | RAKE keyword extraction |
| pandas | Data loading and manipulation |
| matplotlib / seaborn | Visualization |
| wordcloud | Word cloud generation |