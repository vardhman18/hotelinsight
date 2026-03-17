"""
HotelInsight - Impact Calculator
===================================
Computes the quantitative impact that each complaint topic has on star ratings
and identifies the highest-impact issues for a hotel.

These functions are thin wrappers / convenience re-exports of the core
``calculate_impact`` function in :mod:`src.analysis.pattern_detector`,
with additional helpers for building impact tables and ranking issues.
"""

from typing import Dict, List, Tuple

import pandas as pd

from src.config.settings import COMPLAINT_CATEGORIES
from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_topic_impact(hotel_df: pd.DataFrame, topic: str) -> float:
    """Calculate the average rating reduction caused by a specific topic.

    Delegates to :func:`~src.analysis.pattern_detector.calculate_impact`.

    Args:
        hotel_df: Cleaned reviews DataFrame with ``has_<topic>`` and
            ``rating`` columns.
        topic: Topic name.

    Returns:
        Positive float representing star-rating reduction (0 if no data).
    """
    from src.analysis.pattern_detector import calculate_impact

    return calculate_impact(hotel_df, topic)


def build_impact_table(hotel_df: pd.DataFrame) -> pd.DataFrame:
    """Build a summary table of impact metrics for all complaint categories.

    Args:
        hotel_df: Hotel reviews DataFrame.

    Returns:
        DataFrame with columns:

        - ``topic``
        - ``complaint_count``
        - ``complaint_rate`` (as percentage)
        - ``avg_rating_with``
        - ``avg_rating_without``
        - ``impact`` (positive = reduces rating)
    """
    rows = []

    for cat in COMPLAINT_CATEGORIES:
        col = f"has_{cat}"
        if col not in hotel_df.columns or "rating" not in hotel_df.columns:
            continue

        with_mask = hotel_df[col] == True
        without_mask = ~with_mask

        count = int(with_mask.sum())
        total = len(hotel_df)
        rate = round(count / total * 100, 1) if total > 0 else 0.0

        avg_with = float(hotel_df.loc[with_mask, "rating"].mean()) if count > 0 else float("nan")
        avg_without = float(hotel_df.loc[without_mask, "rating"].mean()) if without_mask.sum() > 0 else float("nan")

        if pd.notna(avg_with) and pd.notna(avg_without):
            impact = round(max(avg_without - avg_with, 0.0), 2)
        else:
            impact = 0.0

        rows.append(
            {
                "topic": cat,
                "complaint_count": count,
                "complaint_rate": rate,
                "avg_rating_with": round(avg_with, 2) if pd.notna(avg_with) else None,
                "avg_rating_without": round(avg_without, 2) if pd.notna(avg_without) else None,
                "impact": impact,
            }
        )

    return pd.DataFrame(rows).sort_values("impact", ascending=False).reset_index(drop=True)


def rank_topics_by_impact(hotel_df: pd.DataFrame) -> List[Tuple[str, float]]:
    """Return topics sorted by their rating impact (highest first).

    Args:
        hotel_df: Hotel reviews DataFrame.

    Returns:
        List of ``(topic_name, impact_score)`` tuples, highest impact first.
    """
    table = build_impact_table(hotel_df)
    return list(zip(table["topic"], table["impact"]))
