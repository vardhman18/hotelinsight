"""
HotelInsight - Priority Scorer
=================================
Standalone priority scoring utilities used by the Streamlit dashboard
and the action generator when ordering recommendations.
"""

from typing import Dict, List

from src.config.settings import (
    PRIORITY_CRITICAL_THRESHOLD,
    PRIORITY_HIGH_THRESHOLD,
    PRIORITY_MEDIUM_THRESHOLD,
)


def score_issue(
    frequency: float,
    impact: float,
    trend: str,
) -> Dict:
    """Calculate a weighted priority score for a hotel issue.

    This is a thin wrapper around
    :func:`~src.analysis.pattern_detector.calculate_priority_score` kept
    here so the planning layer has no reverse-import dependency on the
    analysis layer.

    Formula::

        score = (frequency × 0.4) + (impact × 20 × 0.4) + (trend_score × 0.2)

    Args:
        frequency: Complaint frequency (0–100 %).
        impact: Rating impact (0–5 stars).
        trend: ``"INCREASING"``, ``"STABLE"``, or ``"DECREASING"``.

    Returns:
        Dictionary with ``score`` (float) and ``category`` (str).
    """
    trend_map = {"INCREASING": 10, "STABLE": 5, "DECREASING": 0}
    trend_score = trend_map.get(trend, 5)

    raw = (frequency * 0.4) + (impact * 20 * 0.4) + (trend_score * 0.2)
    score = min(round(raw, 1), 100.0)

    if score >= PRIORITY_CRITICAL_THRESHOLD:
        category = "CRITICAL"
    elif score >= PRIORITY_HIGH_THRESHOLD:
        category = "HIGH"
    elif score >= PRIORITY_MEDIUM_THRESHOLD:
        category = "MEDIUM"
    else:
        category = "LOW"

    return {"score": score, "category": category}


def rank_actions_by_confidence(actions: List[Dict]) -> List[Dict]:
    """Sort action plan items by their root-cause confidence (descending).

    Args:
        actions: List of action dictionaries each containing a
            ``confidence`` key.

    Returns:
        Sorted list (highest confidence first).
    """
    return sorted(actions, key=lambda a: a.get("confidence", 0), reverse=True)


PRIORITY_COLOURS = {
    "CRITICAL": "#ff4444",
    "HIGH":     "#ff8c00",
    "MEDIUM":   "#ffd700",
    "LOW":      "#4caf50",
}
"""Hex colour codes for each priority category (used in the Streamlit UI)."""


def priority_badge(category: str) -> str:
    """Return a coloured emoji badge for a priority category.

    Args:
        category: One of ``"CRITICAL"``, ``"HIGH"``, ``"MEDIUM"``,
            ``"LOW"``.

    Returns:
        Emoji string.
    """
    badges = {
        "CRITICAL": "🔴",
        "HIGH":     "🟠",
        "MEDIUM":   "🟡",
        "LOW":      "🟢",
    }
    return badges.get(category, "⚪")
