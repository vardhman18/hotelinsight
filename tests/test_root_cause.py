"""
Tests for the root cause analyser.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.root_cause_analyzer import infer_root_causes


# ───────────────────────────── Fixtures ──────────────────────────────


def _make_df(review_texts: list[str], topic: str) -> pd.DataFrame:
    """Build a minimal cleaned DataFrame with has_{topic} = True."""
    df = pd.DataFrame(
        {
            "review_text": review_texts,
            "rating":      [2.5] * len(review_texts),
            "date":        pd.to_datetime(["2023-01-01"] * len(review_texts)),
            f"has_{topic}": [True] * len(review_texts),
        }
    )
    return df


# ───────────────────────────── Basic contract ────────────────────────


def test_returns_list():
    df = _make_df(["Room was dirty and understaffed hotel"], "cleanliness")
    result = infer_root_causes(df, "cleanliness")
    assert isinstance(result, list)


def test_each_item_has_required_keys():
    df = _make_df(
        ["Room was dirty, the staff were short-handed and busy"] * 5, "cleanliness"
    )
    result = infer_root_causes(df, "cleanliness")
    for item in result:
        assert "cause" in item
        assert "confidence" in item
        assert "description" in item
        assert "evidence" in item


def test_confidence_in_valid_range():
    df = _make_df(
        ["The room was dirty due to not enough cleaning staff"] * 10, "cleanliness"
    )
    result = infer_root_causes(df, "cleanliness")
    for item in result:
        assert 0 <= item["confidence"] <= 100


def test_sorted_by_confidence_descending():
    df = _make_df(
        ["understaffed hotel not enough staff to clean rooms"] * 20, "cleanliness"
    )
    result = infer_root_causes(df, "cleanliness")
    if len(result) > 1:
        for i in range(len(result) - 1):
            assert result[i]["confidence"] >= result[i + 1]["confidence"]


def test_empty_df_returns_empty_list():
    df = _make_df([], "cleanliness")
    result = infer_root_causes(df, "cleanliness")
    assert result == []


def test_wrong_topic_column_missing():
    df = pd.DataFrame({"review_text": ["dirty room"], "rating": [2.0]})
    result = infer_root_causes(df, "cleanliness")
    # Should not raise; may return empty
    assert isinstance(result, list)


# ───────────────────────────── Cause detection ───────────────────────


def test_understaffing_detected():
    texts = [
        "Hotel was clearly understaffed with not enough cleaners available"
    ] * 15
    df = _make_df(texts, "cleanliness")
    result = infer_root_causes(df, "cleanliness")
    causes = [r["cause"] for r in result]
    assert "UNDERSTAFFING" in causes


def test_equipment_issues_detected():
    texts = [
        "The vacuum cleaner was broken and the equipment was faulty always"
    ] * 15
    df = _make_df(texts, "maintenance")
    result = infer_root_causes(df, "maintenance")
    causes = [r["cause"] for r in result]
    assert "EQUIPMENT_ISSUES" in causes


def test_evidence_contains_strings():
    df = _make_df(
        ["Understaffed hotel with broken equipment and poor training"] * 5,
        "cleanliness",
    )
    result = infer_root_causes(df, "cleanliness")
    for item in result:
        assert isinstance(item["evidence"], list)
        for e in item["evidence"]:
            assert isinstance(e, str)


def test_single_complaint_review_still_returns_causes():
    df = _make_df(["Room was not clean"], "cleanliness")
    result = infer_root_causes(df, "cleanliness")
    assert isinstance(result, list)
    assert len(result) >= 1
