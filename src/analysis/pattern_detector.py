"""
HotelInsight - Pattern Detector
==================================
Analyses hotel reviews to detect recurring complaint patterns, track trends
over time, measure rating impact, and calculate priority scores for each issue.

All functions are designed to work on DataFrames that have already been
cleaned by :mod:`src.data_processing.data_cleaner` and enriched with
``has_<topic>`` indicator columns by
:mod:`src.data_processing.feature_extractor`.
"""

from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config.settings import (
    COMPLAINT_CATEGORIES,
    PRIORITY_CRITICAL_THRESHOLD,
    PRIORITY_HIGH_THRESHOLD,
    PRIORITY_MEDIUM_THRESHOLD,
)
from src.data_processing.data_cleaner import clean_reviews
from src.data_processing.data_loader import load_hotel_by_name
from src.utils.date_utils import filter_recent
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _ensure_topic_columns(hotel_df: pd.DataFrame) -> pd.DataFrame:
    """Add ``has_<topic>`` columns to *hotel_df* if they are missing.

    Uses keyword-based topic extraction when the column is absent.

    Args:
        hotel_df: Cleaned hotel reviews DataFrame.

    Returns:
        DataFrame with ``has_<topic>`` boolean columns for every category.
    """
    missing_cats = [
        c for c in COMPLAINT_CATEGORIES
        if f"has_{c}" not in hotel_df.columns
    ]

    if not missing_cats:
        return hotel_df

    from src.analysis.topic_classifier import TopicClassifier
    from src.data_processing.feature_extractor import add_topic_indicator_columns

    logger.info(
        "Topic indicator columns missing; running keyword classification …"
    )
    classifier = TopicClassifier()

    if "review_text" not in hotel_df.columns:
        for cat in missing_cats:
            hotel_df[f"has_{cat}"] = False
        return hotel_df

    hotel_df = hotel_df.copy()
    hotel_df["topics"] = hotel_df["review_text"].apply(
        classifier.extract_topics_keyword
    )
    hotel_df = add_topic_indicator_columns(hotel_df)

    return hotel_df


def _prepare_hotel_df(hotel_name: str) -> pd.DataFrame:
    """Load, clean, and enrich reviews for *hotel_name*.

    Args:
        hotel_name: Exact hotel name.

    Returns:
        Enriched DataFrame ready for analysis functions.
    """
    raw = load_hotel_by_name(hotel_name)
    cleaned = clean_reviews(raw)
    enriched = _ensure_topic_columns(cleaned)
    return enriched


def analyze_hotel(hotel_name: str) -> Dict:
    """Compute a comprehensive analysis summary for a single hotel.

    Loads, cleans, and analyses all reviews for *hotel_name*.

    Args:
        hotel_name: Exact hotel name as it appears in the dataset.

    Returns:
        Dictionary with the keys:

        - ``hotel_name`` (str)
        - ``total_reviews`` (int)
        - ``avg_rating`` (float)
        - ``avg_sentiment`` (float) – average VADER compound score
        - ``topic_stats`` (dict) – per-topic ``count`` and ``percentage``
    """
    logger.info("Analysing hotel: '%s'", hotel_name)

    hotel_df = _prepare_hotel_df(hotel_name)

    if hotel_df.empty:
        logger.warning("No reviews for '%s'.", hotel_name)
        return {
            "hotel_name": hotel_name,
            "total_reviews": 0,
            "avg_rating": 0.0,
            "avg_sentiment": 0.0,
            "topic_stats": {},
        }

    total = len(hotel_df)
    avg_rating = float(hotel_df["rating"].mean()) if "rating" in hotel_df.columns else 0.0

    # Lightweight sentiment via VADER (fast; full BERT done separately)
    avg_sentiment = 0.0
    if "sentiment_score" in hotel_df.columns:
        avg_sentiment = float(hotel_df["sentiment_score"].mean())
    else:
        try:
            from src.analysis.sentiment_analyzer import SentimentAnalyzer
            analyser = SentimentAnalyzer(method="vader")
            sample = hotel_df["review_text"].head(200).tolist()
            scores = analyser.analyze_batch(sample)
            avg_sentiment = float(np.mean(scores)) if scores else 0.0
        except Exception as exc:
            logger.debug("VADER sentiment skipped: %s", exc)

    # Per-topic stats
    topic_stats: Dict[str, Dict] = {}
    for cat in COMPLAINT_CATEGORIES:
        col = f"has_{cat}"
        if col in hotel_df.columns:
            count = int(hotel_df[col].sum())
            pct = round(count / total * 100, 1) if total > 0 else 0.0
            topic_stats[cat] = {"count": count, "percentage": pct}

    return {
        "hotel_name": hotel_name,
        "total_reviews": total,
        "avg_rating": round(avg_rating, 2),
        "avg_sentiment": round(avg_sentiment, 3),
        "topic_stats": topic_stats,
    }


def analyze_trends(hotel_name: str, months: int = 12) -> Dict:
    """Analyse how complaint rates trend over the last *months* months.

    Compares the most recent 3 months to the preceding 3 months to
    determine whether each topic is ``INCREASING``, ``DECREASING``, or
    ``STABLE``.

    Args:
        hotel_name: Exact hotel name.
        months: How far back to look (default 12).

    Returns:
        Dictionary keyed by topic name with sub-keys:

        - ``trend``: ``"INCREASING"``, ``"DECREASING"``, or ``"STABLE"``
        - ``trend_icon``: ``"↑"``, ``"↓"``, or ``"→"``
        - ``current_rate``: Complaint rate in the most-recent period (0–100)
        - ``monthly_data``: Dict of ``"YYYY-MM"`` → complaint rate
    """
    hotel_df = _prepare_hotel_df(hotel_name)

    if hotel_df.empty or "date" not in hotel_df.columns:
        return {}

    recent_df = filter_recent(hotel_df, "date", months=months)
    if recent_df.empty:
        return {}

    recent_df = recent_df.copy()
    recent_df["year_month"] = recent_df["date"].dt.to_period("M").astype(str)

    results: Dict[str, Dict] = {}

    for cat in COMPLAINT_CATEGORIES:
        col = f"has_{cat}"
        if col not in recent_df.columns:
            continue

        # Monthly complaint rates
        monthly = (
            recent_df.groupby("year_month")[col]
            .mean()
            .mul(100)
            .round(1)
        )

        if monthly.empty:
            continue

        monthly_dict = monthly.to_dict()

        # Trend: compare last 3 months vs previous 3 months
        sorted_months = sorted(monthly_dict.keys())
        recent_3 = [monthly_dict[m] for m in sorted_months[-3:]]
        older_3 = [monthly_dict[m] for m in sorted_months[-6:-3]] if len(sorted_months) >= 6 else []

        recent_avg = np.mean(recent_3) if recent_3 else 0.0
        older_avg = np.mean(older_3) if older_3 else recent_avg

        diff = recent_avg - older_avg
        if diff > 3:
            trend, icon = "INCREASING", "↑"
        elif diff < -3:
            trend, icon = "DECREASING", "↓"
        else:
            trend, icon = "STABLE", "→"

        results[cat] = {
            "trend": trend,
            "trend_icon": icon,
            "current_rate": round(float(recent_avg), 1),
            "monthly_data": monthly_dict,
        }

    return results


def calculate_impact(hotel_df: pd.DataFrame, topic: str) -> float:
    """Calculate the average rating impact of a complaint topic.

    Computes: ``avg_rating_without_topic - avg_rating_with_topic``.
    A positive return value indicates the topic negatively impacts ratings.

    Args:
        hotel_df: Reviews DataFrame with ``has_<topic>`` and ``rating`` columns.
        topic: Topic name (e.g. ``"cleanliness"``).

    Returns:
        Rating impact as a positive float (0 if the topic has no data).
    """
    col = f"has_{topic}"
    if col not in hotel_df.columns or "rating" not in hotel_df.columns:
        return 0.0

    with_topic = hotel_df[hotel_df[col] == True]["rating"]
    without_topic = hotel_df[hotel_df[col] == False]["rating"]

    if with_topic.empty or without_topic.empty:
        return 0.0

    impact = float(without_topic.mean() - with_topic.mean())
    return round(max(impact, 0.0), 2)


def get_all_impacts(hotel_name: str) -> Dict[str, float]:
    """Calculate rating impact for every complaint category for a hotel.

    Args:
        hotel_name: Exact hotel name.

    Returns:
        Dictionary mapping category name → impact score (0–5 scale).
    """
    hotel_df = _prepare_hotel_df(hotel_name)
    return {cat: calculate_impact(hotel_df, cat) for cat in COMPLAINT_CATEGORIES}


def calculate_priority_score(
    frequency: float,
    impact: float,
    trend: str,
) -> Dict:
    """Calculate a weighted priority score for a hotel issue.

    Formula::

        score = (frequency × 0.4) + (impact × 20 × 0.4) + (trend_score × 0.2)

    Where ``trend_score`` is 10 for ``INCREASING``, 5 for ``STABLE``, and 0
    for ``DECREASING``.

    The raw score is capped at 100 and then bucketed into severity categories
    using the thresholds defined in ``settings.py``.

    Args:
        frequency: Percentage of reviews mentioning the issue (0–100).
        impact: Star-rating reduction caused by the issue (0–5).
        trend: ``"INCREASING"``, ``"STABLE"``, or ``"DECREASING"``.

    Returns:
        Dictionary with:

        - ``score`` (float): Weighted score in ``[0, 100]``.
        - ``category`` (str): ``"CRITICAL"``, ``"HIGH"``, ``"MEDIUM"``,
          or ``"LOW"``.
    """
    trend_scores = {"INCREASING": 10, "STABLE": 5, "DECREASING": 0}
    trend_score = trend_scores.get(trend, 5)

    score = (
        (frequency * 0.4)
        + (impact * 20 * 0.4)
        + (trend_score * 0.2)
    )
    score = min(round(score, 1), 100.0)

    if score >= PRIORITY_CRITICAL_THRESHOLD:
        category = "CRITICAL"
    elif score >= PRIORITY_HIGH_THRESHOLD:
        category = "HIGH"
    elif score >= PRIORITY_MEDIUM_THRESHOLD:
        category = "MEDIUM"
    else:
        category = "LOW"

    return {"score": score, "category": category}
