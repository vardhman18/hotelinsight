"""
Tests for the SentimentAnalyzer module.
BERT tests are skipped by default (slow; set HOTELINSIGHT_BERT_TESTS=1 to run).
"""

import os
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.sentiment_analyzer import SentimentAnalyzer

RUN_BERT = os.getenv("HOTELINSIGHT_BERT_TESTS", "0") == "1"


# ───────────────────────────── VADER tests ───────────────────────────


@pytest.fixture(scope="module")
def vader():
    return SentimentAnalyzer(method="vader")


def test_vader_positive(vader):
    score = vader.analyze("The hotel was absolutely fantastic and wonderful!")
    assert score > 0.0


def test_vader_negative(vader):
    score = vader.analyze("The room was filthy and the staff were rude and unhelpful.")
    assert score < 0.0


def test_vader_neutral_ish(vader):
    score = vader.analyze("The hotel had a reception desk.")
    assert -0.5 <= score <= 0.5


def test_vader_empty_string(vader):
    # Should not raise; returns neutral or near-zero
    score = vader.analyze("")
    assert isinstance(score, float)


def test_vader_batch(vader):
    texts = ["Great hotel", "Terrible service", "Average experience"]
    scores = vader.analyze_batch(texts)
    assert len(scores) == 3
    assert scores[0] > scores[1]  # Great > Terrible


def test_vader_get_label_positive(vader):
    assert vader.get_sentiment_label(0.5) == "positive"


def test_vader_get_label_negative(vader):
    assert vader.get_sentiment_label(-0.5) == "negative"


def test_vader_get_label_neutral(vader):
    assert vader.get_sentiment_label(0.0) == "neutral"


# ───────────────────────────── BERT tests (optional) ─────────────────


@pytest.mark.skipif(not RUN_BERT, reason="Set HOTELINSIGHT_BERT_TESTS=1 to run BERT tests")
def test_bert_positive():
    bert = SentimentAnalyzer(method="bert")
    score = bert.analyze("An absolutely superb stay. Everything was perfect!")
    assert score > 0.3


@pytest.mark.skipif(not RUN_BERT, reason="Set HOTELINSIGHT_BERT_TESTS=1 to run BERT tests")
def test_bert_negative():
    bert = SentimentAnalyzer(method="bert")
    score = bert.analyze("The room was disgusting. I would never return.")
    assert score < 0.0


@pytest.mark.skipif(not RUN_BERT, reason="Set HOTELINSIGHT_BERT_TESTS=1 to run BERT tests")
def test_bert_batch_length():
    bert = SentimentAnalyzer(method="bert")
    texts = ["Good", "Bad", "OK", "Amazing", "Awful"]
    scores = bert.analyze_batch(texts)
    assert len(scores) == len(texts)


# ───────────────────────────── Score range ───────────────────────────


@pytest.mark.parametrize("method", ["vader"])
def test_score_range(method):
    analyser = SentimentAnalyzer(method=method)
    texts = [
        "Perfect stay, outstanding service!",
        "Dirty rooms and rude staff.",
        "The hotel was okay.",
    ]
    for text in texts:
        score = analyser.analyze(text)
        assert -1.0 <= score <= 1.0, f"Score {score} out of range for: {text!r}"
