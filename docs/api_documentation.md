# HotelInsight — API Documentation

All public functions, classes, and their signatures are listed here by module.

---

## `src.data_processing.data_loader`

### `load_hotel_reviews(dataset="main") -> pd.DataFrame`
Load the full hotel reviews CSV.

| Parameter | Type | Description |
|-----------|------|-------------|
| `dataset` | str  | Currently only `"main"` is supported |

Returns a DataFrame with at minimum: `Hotel_Name`, `Reviewer_Score`, `Review_Date`,
`Negative_Review`, `Positive_Review`.

---

### `load_hotel_by_name(hotel_name: str) -> pd.DataFrame`
Filter the dataset to reviews for a single property.

---

### `get_hotel_list() -> list[str]`
Return a sorted list of unique `Hotel_Name` values.

---

### `get_hotel_stats(hotel_name: str) -> dict`
Return a summary dict:
```python
{
    "total_reviews": int,
    "avg_rating":    float,   # 1–10 scale
    "date_range":    str,     # "Jan 2015 – Aug 2017"
}
```

---

## `src.data_processing.data_cleaner`

### `clean_reviews(df: pd.DataFrame) -> pd.DataFrame`
Add derived columns: `review_text`, `rating` (1–5), `date`, `review_length`, `word_count`.
Drops rows with empty review text.

---

### `preprocess_text(text: str) -> str`
Lowercase, punctuation removal, whitespace normalisation.

---

### `split_train_test(df, test_size=0.2) -> tuple[pd.DataFrame, pd.DataFrame]`
Stratified split. Returns `(train_df, test_df)`.

---

## `src.data_processing.feature_extractor`

### `class FeatureExtractor`
TF-IDF vectoriser wrapper (`max_features=10 000`, `ngram_range=(1,2)`).

| Method | Signature | Description |
|--------|-----------|-------------|
| `fit_transform` | `(texts) -> scipy.sparse` | Fit and transform |
| `transform`     | `(texts) -> scipy.sparse` | Transform only |
| `save`          | `(path=None)` | Pickle to `models/` |
| `load`          | `(path=None)` | Restore from pickle |

---

### `add_topic_indicator_columns(df: pd.DataFrame) -> pd.DataFrame`
Adds boolean columns `has_cleanliness`, `has_staff`, etc. via keyword matching.

---

## `src.analysis.sentiment_analyzer`

### `class SentimentAnalyzer(method="bert")`
Lazy-loading sentiment scorer.

| Parameter | Values | Description |
|-----------|--------|-------------|
| `method`  | `"bert"` / `"vader"` | BERT = higher accuracy; VADER = faster |

| Method | Signature | Returns |
|--------|-----------|---------|
| `analyze` | `(text: str) -> float` | Score in [−1, +1] |
| `analyze_batch` | `(texts: list[str]) -> list[float]` | Batch scores |
| `get_sentiment_label` | `(score: float) -> str` | `"positive"` / `"neutral"` / `"negative"` |

**Thresholds:** > 0.3 → positive; < −0.3 → negative.

---

## `src.analysis.nlp_features`

### `class NLPFeaturesExtractor`
Advanced NLP feature extraction using spaCy for entity recognition, POS tagging, and dependency parsing.

| Method | Signature | Returns |
|--------|-----------|---------|
| `extract_entities` | `(text: str) -> dict[str, list[str]]` | Named entities by type (PERSON, ORG, GPE, PRODUCT, FACILITY) |
| `extract_noun_phrases` | `(text: str) -> list[str]` | Multi-word noun chunks |
| `extract_adjectives` | `(text: str) -> list[tuple[str, str]]` | (adjective, noun) pairs |
| `extract_pos_tags` | `(text: str) -> dict[str, list[str]]` | Grouped by POS type (NOUN, VERB, ADJ, ADV, PROPN) |
| `extract_lemmas` | `(text: str) -> list[str]` | Lemmatized word tokens |
| `analyze_syntactic_complexity` | `(text: str) -> dict` | sentence_count, avg_sentence_length, avg_dependency_depth |
| `comprehensive_analysis` | `(text: str) -> dict` | Combines all extraction methods |
| `batch_analysis` | `(texts: list[str]) -> list[dict]` | Efficient batch processing |

**Features:** Identifies specific complaint topics (e.g. "dirty bathroom", "rude staff") using linguistic patterns.

---

## `src.analysis.nltk_sentiment`

### `class NLTKSentimentAnalyzer`
NLTK-based sentiment analysis using TextBlob, tokenization, and subjectivity scoring.

| Method | Signature | Returns |
|--------|-----------|---------|
| `analyze_textblob` | `(text: str) -> dict` | polarity, subjectivity scores |
| `analyze_sentence_level` | `(text: str) -> dict` | Per-sentence sentiment analysis |
| `extract_subjective_words` | `(text: str) -> list[str]` | Opinion-bearing words |
| `word_frequency` | `(text: str, top_n=10) -> list[tuple]` | (word, frequency) pairs |
| `extract_keywords` | `(text: str, top_n=15) -> tuple` | (keywords_list, keyword_scores_dict) |
| `tokenize_and_analyze` | `(text: str) -> dict` | Token count, word count, vocabulary richness |
| `comprehensive_nltk_analysis` | `(text: str) -> dict` | Combines all NLTK analyses |

### `class EnsembleSentimentAnalyzer`
Combines BERT, VADER, and TextBlob for robust sentiment prediction with confidence scoring.

| Method | Signature | Returns |
|--------|-----------|---------|
| `ensemble_score` | `(text: str, weights=None) -> dict` | ensemble_score, component scores, agreement level, consensus_label |

**Weights (default):** VADER 0.33, BERT 0.33, TextBlob 0.34.
**Agreement:** Normalized standard deviation of component scores (higher = more consensus).

---

## `src.analysis.xgboost_classifier`

### `class XGBoostTopicClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8)`
Multi-label topic classifier using XGBoost gradient boosting (enhanced alternative to RandomForest).

| Method | Signature | Returns |
|--------|-----------|---------|
| `train` | `(texts, labels, test_size=0.2) -> dict` | Training metrics per category |
| `predict` | `(texts: list[str]) -> list[list[str]]` | Predicted labels for each text |
| `predict_proba` | `(texts: list[str]) -> dict` | Probability array per category |
| `get_feature_importance` | `(category: str, top_n=20) -> list[tuple]` | Top TF-IDF features by importance |
| `save` | `(model_path=None)` | Pickle models to disk |
| `load` | `(model_path=None)` | Restore models from disk |

**Binary relevance:** One XGBoost classifier per category for multi-label predictions.

### `class XGBoostEnsembleClassifier(xgboost_weight=0.6, random_forest_weight=0.4)`
Ensemble combining XGBoost and RandomForest predictions via weighted voting.

| Method | Signature | Returns |
|--------|-----------|---------|
| `fit` | `(xgboost_clf, rf_clf)` | Sets component classifiers |
| `predict` | `(texts: list[str]) -> list[list[str]]` | Union of both methods' predictions |
| `predict_with_confidence` | `(texts: list[str]) -> list[tuple]` | (predicted_labels, confidence_score) |

---

## `src.analysis.topic_classifier`

### `class TopicClassifier(n_estimators=200)`
Keyword + ML topic classifier for 8 complaint categories.

| Method | Signature | Returns |
|--------|-----------|---------|
| `extract_topics_keyword` | `(text: str) -> list[str]` | Keyword-only matches |
| `predict` | `(text: str) -> list[str]` | Union of keyword + ML |
| `predict_batch` | `(texts: list[str]) -> list[list[str]]` | Batch prediction |
| `train` | `(X_train, y_train)` | Fit RandomForest |
| `save` | `(path=None)` | Pickle model |
| `load` | `(path=None)` | Restore model |

---

## `src.analysis.pattern_detector`

### `analyze_hotel(hotel_name: str) -> dict`
```python
{
    "hotel_name":      str,
    "total_reviews":   int,
    "avg_rating":      float,
    "avg_sentiment":   float,
    "topic_stats": {
        "<topic>": {
            "complaint_count": int,
            "complaint_rate":  float,  # 0–1
            "avg_rating":      float,
            "trend":           str,    # INCREASING | DECREASING | STABLE
            "priority_score":  float,
        }, ...
    }
}
```

---

### `analyze_trends(hotel_name: str, months: int = 12) -> dict`
Returns per-topic trend info comparing recent 3 months vs prior 3 months.

---

### `calculate_priority_score(frequency: float, impact: float, trend: str) -> dict`
Returns `{"score": float, "category": str}` where category is one of Critical / High / Medium / Low.

---

## `src.analysis.root_cause_analyzer`

### `infer_root_causes(hotel_df: pd.DataFrame, topic: str) -> list[dict]`
```python
[
    {
        "cause":       str,    # e.g. "UNDERSTAFFING"
        "confidence":  float,  # 0–100
        "description": str,
        "evidence":    list[str],  # up to 3 verbatim snippets
    }, ...
]
```
Sorted descending by confidence. Minimum 10% threshold to appear.

---

## `src.analysis.impact_calculator`

### `build_impact_table(hotel_df: pd.DataFrame) -> pd.DataFrame`
Columns: `topic`, `complaint_count`, `complaint_rate`, `avg_rating_with`,
`avg_rating_without`, `impact`.

---

### `rank_topics_by_impact(hotel_df: pd.DataFrame) -> list[tuple[str, float]]`
Returns `[("cleanliness", 0.82), ...]` sorted descending by impact.

---

## `src.planning.cost_calculator`

### `evaluate_cost_formula(formula: str, rooms: int) -> float`
Safe-eval a cost formula string. Raises `ValueError` on invalid input.

**Allowed variables:** `rooms`, `TRAINING_COST_BASE`, `EQUIPMENT_COST_PER_ROOM`, `STAFF_COST_PER_ROOM`.

---

### `format_currency(amount: float) -> str`
Returns e.g. `"£1,250"`.

---

### `calculate_total_costs(immediate, short_term, long_term) -> dict`
Returns `{"one_time": float, "monthly": float, "three_month_total": float}`.

---

## `src.planning.action_generator`

### `class ActionPlanGenerator`
Loads `action_templates.json` and generates tailored plans.

### `generate_plan(hotel_name, topic, root_causes, num_rooms=100) -> dict`
```python
{
    "immediate_actions":  list[ActionDict],
    "short_term_actions": list[ActionDict],
    "long_term_actions":  list[ActionDict],
    "total_cost": {"one_time": float, "monthly": float, "three_month_total": float},
}
```
Each `ActionDict` has: `description`, `cost`, `timeline`, `expected_impact`, `root_cause`, `confidence`.

---

## `src.planning.roi_predictor`

### `calculate_roi(hotel_name, topic, action_plan, num_rooms=100, avg_room_rate=150, current_occupancy=0.65) -> dict`
```python
{
    "current_rating":          str,
    "expected_rating":         str,
    "rating_improvement":      str,
    "current_occupancy":       str,
    "expected_occupancy":      str,
    "monthly_revenue_increase": str,
    "total_investment":        str,
    "net_profit_3_months":     str,
    "roi_percentage":          str,
    "payback_months":          str,
}
```
All values are formatted strings (e.g. `"£2,400"` or `"+0.3"`).

---

## `src.visualization.charts`

All functions return `plotly.graph_objects.Figure`.

| Function | Parameters | Description |
|----------|------------|-------------|
| `rating_trend_chart` | `hotel_df` | Monthly average rating line chart |
| `complaint_frequency_bar` | `topic_stats: dict` | Horizontal bar (complaint %) |
| `sentiment_distribution_pie` | `labels: list[str]` | Positive/Neutral/Negative pie |
| `impact_heatmap` | `impact_table: pd.DataFrame` | Rating-drop heatmap |
| `trend_line_chart` | `trends: dict` | Each topic's trend arrows |
| `roi_waterfall_chart` | `roi_data: dict` | Investment vs revenue waterfall |

---

## `src.visualization.report_generator`

### `export_analysis_excel(hotel_name, analysis, impact_table, action_plan=None, roi_data=None) -> str`
Writes a multi-sheet `.xlsx` to `data/results/` and returns the file path.

Sheets: `Summary`, `Impact Analysis`, `Action Plan`, `ROI Projection`.
