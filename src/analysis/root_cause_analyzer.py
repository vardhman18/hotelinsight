"""
HotelInsight - Root Cause Analyser
=====================================
Infers the most likely root causes for a specific complaint topic by
scanning review text for diagnostic keyword patterns.

Supported root cause types:
- UNDERSTAFFING
- TRAINING_ISSUES
- EQUIPMENT_ISSUES
- TIME_PRESSURE
- SUPPLY_CHAIN_ISSUES
- PROCESS_INEFFICIENCY
- COMMUNICATION_BREAKDOWN
"""

from typing import Dict, List

import pandas as pd

from src.config.settings import ROOT_CAUSE_CONFIDENCE_THRESHOLD
from src.utils.logger import get_logger
from src.utils.text_processing import clean_text

logger = get_logger(__name__)

# Minimum keyword-hit rate (as a fraction) required to include a cause
_MIN_CONFIDENCE_FRACTION = 0.10  # 10 %

# ---------------------------------------------------------------------------
# Keyword dictionaries for each root cause type
# ---------------------------------------------------------------------------

_ROOT_CAUSE_KEYWORDS: Dict[str, List[str]] = {
    "UNDERSTAFFING": [
        "understaffed", "not enough staff", "short staffed", "short-staffed",
        "overworked", "rushed staff", "too busy", "couldn't find anyone",
        "no one available", "waited ages", "nobody came", "staff missing",
        "overwhelmed", "too few staff",
    ],
    "TRAINING_ISSUES": [
        "didn't know", "didn't seem to know", "untrained", "new staff",
        "inconsistent", "hit or miss", "unprofessional", "clueless",
        "no idea", "poorly trained", "lack of training",
        "seems new", "inexperienced",
    ],
    "EQUIPMENT_ISSUES": [
        "broken", "not working", "stopped working", "out of order",
        "needs replacement", "old equipment", "outdated", "faulty",
        "keeps breaking", "malfunctioning", "worn out",
    ],
    "TIME_PRESSURE": [
        "rushed", "hurried", "not enough time", "quick clean",
        "fast checkout", "not given enough time", "in a rush",
        "cutting corners", "too fast",
    ],
    "SUPPLY_CHAIN_ISSUES": [
        "out of stock", "no supplies", "ran out", "missing items",
        "no toiletries", "no towels", "no amenities", "couldn't get",
        "supply problem", "not available",
    ],
    "PROCESS_INEFFICIENCY": [
        "disorganised", "disorganized", "chaotic", "no system",
        "wrong room", "booking error", "mistake", "confused",
        "mixed up", "unclear process", "poor process",
        "didn't follow up", "lost complaint",
    ],
    "COMMUNICATION_BREAKDOWN": [
        "didn't tell", "wasn't informed", "not notified",
        "no communication", "left in the dark", "no response",
        "ignored complaint", "nobody told us", "miscommunication",
        "mixed messages", "didn't get back", "no follow up",
    ],
}

# Human-readable descriptions for each cause
_CAUSE_DESCRIPTIONS: Dict[str, str] = {
    "UNDERSTAFFING": "Insufficient staff to handle guest volume",
    "TRAINING_ISSUES": "Staff lack skills or knowledge to perform tasks correctly",
    "EQUIPMENT_ISSUES": "Broken or outdated equipment impairs service delivery",
    "TIME_PRESSURE": "Staff are given insufficient time to complete tasks properly",
    "SUPPLY_CHAIN_ISSUES": "Lack of necessary supplies prevents service fulfilment",
    "PROCESS_INEFFICIENCY": "Poorly designed processes lead to errors or inconsistency",
    "COMMUNICATION_BREAKDOWN": "Poor internal communication causes guests to receive conflicting or no information",
}

# Topic-specific fallback causes used when very sparse data has no keyword hits.
_TOPIC_FALLBACK_CAUSES: Dict[str, List[str]] = {
    "cleanliness": ["UNDERSTAFFING", "TIME_PRESSURE", "PROCESS_INEFFICIENCY"],
    "staff": ["TRAINING_ISSUES", "COMMUNICATION_BREAKDOWN", "UNDERSTAFFING"],
    "maintenance": ["EQUIPMENT_ISSUES", "SUPPLY_CHAIN_ISSUES", "PROCESS_INEFFICIENCY"],
    "noise": ["PROCESS_INEFFICIENCY", "COMMUNICATION_BREAKDOWN", "TIME_PRESSURE"],
    "wifi": ["EQUIPMENT_ISSUES", "SUPPLY_CHAIN_ISSUES", "PROCESS_INEFFICIENCY"],
    "breakfast": ["SUPPLY_CHAIN_ISSUES", "TIME_PRESSURE", "PROCESS_INEFFICIENCY"],
    "value": ["PROCESS_INEFFICIENCY", "UNDERSTAFFING", "COMMUNICATION_BREAKDOWN"],
    "location": ["COMMUNICATION_BREAKDOWN", "PROCESS_INEFFICIENCY", "UNDERSTAFFING"],
}


def infer_root_causes(
    hotel_df: pd.DataFrame,
    topic: str,
) -> List[Dict]:
    """Infer root causes for a specific complaint topic.

    Analyses the review text of all reviews tagged with *topic* and looks
    for diagnostic keyword patterns to score each root cause type.

    Args:
        hotel_df: Cleaned DataFrame of hotel reviews.  Must contain a
            ``review_text`` column and a ``has_<topic>`` boolean column.
        topic: Complaint topic (e.g. ``"cleanliness"``).

    Returns:
        List of root cause dictionaries, sorted by confidence (highest
        first), each containing:

        - ``cause`` (str): Root cause type key.
        - ``confidence`` (float): Confidence percentage (0–100).
        - ``evidence`` (list[str]): Bullet-point strings explaining why.
        - ``description`` (str): Human-readable cause description.

        Only causes with confidence > 10 % are included.
    """
    col = f"has_{topic}"

    # Defensive checks
    if hotel_df.empty:
        logger.warning("infer_root_causes: empty DataFrame for topic '%s'.", topic)
        return []

    if col not in hotel_df.columns:
        logger.warning("Column '%s' not found; returning empty causes.", col)
        return []

    topic_df = hotel_df[hotel_df[col] == True].copy()

    if topic_df.empty:
        logger.info("No '%s' complaints found in dataset.", topic)
        return []

    if "review_text" not in topic_df.columns:
        logger.warning("'review_text' column missing.")
        return []

    total_topic_reviews = len(topic_df)
    logger.info(
        "Inferring root causes for '%s' from %d reviews …",
        topic,
        total_topic_reviews,
    )

    # Combine all review text for fast bulk scanning
    all_text_lower = " ".join(topic_df["review_text"].astype(str).str.lower().tolist())

    results = []

    for cause, keywords in _ROOT_CAUSE_KEYWORDS.items():
        # Count how many *individual reviews* contain at least one keyword
        hit_mask = topic_df["review_text"].astype(str).str.lower().apply(
            lambda text: any(kw in text for kw in keywords)
        )
        hit_count = int(hit_mask.sum())

        confidence = min(
            round((hit_count / total_topic_reviews) * 100, 1),
            100.0,
        )

        if confidence < (_MIN_CONFIDENCE_FRACTION * 100):
            continue

        # Build evidence strings from actual review snippets
        evidence = _build_evidence(
            topic_df[hit_mask],
            keywords,
            cause,
            hit_count,
            total_topic_reviews,
        )

        results.append(
            {
                "cause": cause,
                "confidence": confidence,
                "description": _CAUSE_DESCRIPTIONS.get(cause, cause),
                "evidence": evidence,
            }
        )

    # Sort by confidence descending
    results.sort(key=lambda x: x["confidence"], reverse=True)

    # Sparse-data fallback: still return likely causes so action plans can be
    # generated for hotels/topics with very few complaints (even a single review).
    if not results and total_topic_reviews > 0:
        results = _build_sparse_fallback_causes(topic, total_topic_reviews)

    logger.info(
        "Found %d root causes for '%s': %s",
        len(results),
        topic,
        [r["cause"] for r in results],
    )

    return results


def _build_sparse_fallback_causes(topic: str, total_topic_reviews: int) -> List[Dict]:
    """Return conservative fallback causes when keyword evidence is insufficient."""
    fallback_causes = _TOPIC_FALLBACK_CAUSES.get(
        topic,
        ["PROCESS_INEFFICIENCY", "COMMUNICATION_BREAKDOWN"],
    )

    # Keep confidence modest; for tiny samples we avoid overclaiming.
    if total_topic_reviews <= 2:
        base_conf = 35.0
    elif total_topic_reviews <= 5:
        base_conf = 30.0
    else:
        base_conf = 25.0

    results: List[Dict] = []
    for idx, cause in enumerate(fallback_causes[:3]):
        conf = max(base_conf - (idx * 5), 15.0)
        results.append(
            {
                "cause": cause,
                "confidence": conf,
                "description": _CAUSE_DESCRIPTIONS.get(cause, cause),
                "evidence": [
                    f"Sparse evidence mode: only {total_topic_reviews} complaint review(s) available for this topic.",
                    "Plan is generated using conservative fallback root-cause priors.",
                ],
                "fallback": True,
            }
        )

    logger.info(
        "Using sparse-data fallback causes for topic '%s' (%d review(s)).",
        topic,
        total_topic_reviews,
    )
    return results


def _build_evidence(
    hit_df: pd.DataFrame,
    keywords: List[str],
    cause: str,
    hit_count: int,
    total: int,
) -> List[str]:
    """Generate human-readable evidence strings for a root cause.

    Args:
        hit_df: DataFrame rows where keywords were found.
        keywords: Keyword list for this cause.
        cause: Root cause type (for context labels).
        hit_count: Number of reviews that triggered this cause.
        total: Total reviews mentioning the topic.

    Returns:
        List of brief evidence bullet strings.
    """
    evidence = []

    # Statistical summary
    pct = round(hit_count / total * 100, 1)
    evidence.append(
        f"{hit_count} of {total} reviews ({pct}%) contain "
        f"'{cause.lower().replace('_', ' ')}' indicators."
    )

    # Up to 3 verbatim keyword-match snippets
    snippets_added = 0
    for _, row in hit_df.iterrows():
        text = str(row.get("review_text", "")).lower()
        for kw in keywords:
            if kw in text:
                # Extract a short context window around the keyword
                idx = text.find(kw)
                start = max(0, idx - 40)
                end = min(len(text), idx + len(kw) + 60)
                snippet = text[start:end].strip()
                evidence.append(f'Guest wrote: "…{snippet}…"')
                snippets_added += 1
                break  # one snippet per review

        if snippets_added >= 3:
            break

    return evidence
