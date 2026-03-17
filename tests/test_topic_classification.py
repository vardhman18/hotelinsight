"""
Tests for the TopicClassifier module (keyword path).
ML training tests are in a separate slow-test suite.
"""

import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.topic_classifier import TopicClassifier


@pytest.fixture(scope="module")
def clf():
    return TopicClassifier()


# ───────────────────────────── Keyword matching ──────────────────────


def test_detects_cleanliness(clf):
    topics = clf.extract_topics_keyword("The room was filthy and dirty with stains")
    assert "cleanliness" in topics


def test_detects_staff(clf):
    topics = clf.extract_topics_keyword("The receptionist was rude and unhelpful")
    assert "staff" in topics


def test_detects_noise(clf):
    topics = clf.extract_topics_keyword("It was extremely loud all night, very noisy")
    assert "noise" in topics


def test_detects_wifi(clf):
    topics = clf.extract_topics_keyword("The internet connection was slow and kept dropping")
    assert "wifi" in topics


def test_detects_breakfast(clf):
    topics = clf.extract_topics_keyword("The breakfast buffet had limited food choices")
    assert "breakfast" in topics


def test_detects_maintenance(clf):
    topics = clf.extract_topics_keyword("The air conditioning was broken and the lift was out of order")
    assert "maintenance" in topics


def test_detects_value(clf):
    topics = clf.extract_topics_keyword("The price was too expensive for what was offered")
    assert "value" in topics


def test_detects_location(clf):
    topics = clf.extract_topics_keyword("Far from the city centre and difficult to get to")
    assert "location" in topics


def test_multi_topic_text(clf):
    text = "The room was dirty and the wifi was very slow"
    topics = clf.extract_topics_keyword(text)
    assert "cleanliness" in topics
    assert "wifi" in topics


def test_no_topic_for_generic_text(clf):
    topics = clf.extract_topics_keyword("The hotel was fine.")
    # may return empty list or short list — no hard assertion on length,
    # just verify it's a list of valid categories
    from src.config.settings import COMPLAINT_CATEGORIES
    assert all(t in COMPLAINT_CATEGORIES for t in topics)


# ───────────────────────────── predict (with fallback) ───────────────


def test_predict_returns_list(clf):
    result = clf.predict("The room was filthy and the staff were rude")
    assert isinstance(result, list)


def test_predict_contents_valid(clf):
    from src.config.settings import COMPLAINT_CATEGORIES
    result = clf.predict("Dirty bathroom, slow wifi, rude staff")
    assert all(t in COMPLAINT_CATEGORIES for t in result)


def test_predict_batch(clf):
    texts = [
        "Dirty room and no hot water",
        "Staff were incredibly helpful and friendly",
        "Very noisy from the street all night",
    ]
    results = clf.predict_batch(texts)
    assert len(results) == len(texts)
    assert "cleanliness" in results[0] or "maintenance" in results[0]
    assert "staff" in results[1]
    assert "noise" in results[2]


def test_predict_empty_text(clf):
    result = clf.predict("")
    assert isinstance(result, list)


# ───────────────────────────── add_topic_indicators ──────────────────


def test_add_topic_indicator_columns():
    import pandas as pd
    from src.data_processing.feature_extractor import add_topic_indicator_columns

    df = pd.DataFrame({"review_text": ["The room was dirty", "Great wifi and service"]})
    result = add_topic_indicator_columns(df)
    assert "has_cleanliness" in result.columns
    assert "has_wifi" in result.columns
    assert result["has_cleanliness"].iloc[0]
