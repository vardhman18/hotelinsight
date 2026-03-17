"""
Tests for data processing pipeline.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_processing.data_cleaner import clean_reviews, preprocess_text, split_train_test
from src.utils.text_processing import clean_text, combine_review_parts


# ───────────────────────────── Fixtures ──────────────────────────────


@pytest.fixture
def raw_df():
    """Minimal DataFrame resembling Hotel_Reviews.csv."""
    return pd.DataFrame(
        {
            "Hotel_Name":            ["TestHotel", "TestHotel", "TestHotel"],
            "Hotel_Address":         ["123 Main St", "123 Main St", "123 Main St"],
            "Review_Date":           ["1/10/2023", "2/15/2023", "2/15/2023"],
            "Reviewer_Score":        [4.0, 8.0, 6.0],
            "Negative_Review":       ["Room was dirty", "No Negative", "Noisy corridors"],
            "Positive_Review":       ["Great staff", "Nice breakfast", ""],
            "Reviewer_Nationality":  ["UK", "FR", "UK"],
            "Tags":                  ["[' Solo traveler ']", "[' Couple ']", "[' Business ']"],
            "lat":                   [51.5, 51.5, 51.5],
            "lng":                   [-0.1, -0.1, -0.1],
        }
    )


# ───────────────────────────── clean_text ────────────────────────────


def test_clean_text_lowercases():
    assert clean_text("HELLO WORLD") == "hello world"


def test_clean_text_strips_punctuation():
    result = clean_text("Hello, World!!!")
    assert "," not in result and "!" not in result


def test_clean_text_empty_string():
    assert clean_text("") == ""


def test_clean_text_none():
    assert clean_text(None) == ""


# ───────────────────────────── combine_review_parts ──────────────────


def test_combine_filters_no_positive():
    result = combine_review_parts("Dirty room", "No Positive")
    assert "dirty room" in result.lower()
    assert "no positive" not in result.lower()


def test_combine_filters_no_negative():
    result = combine_review_parts("No Negative", "Great staff")
    assert "great staff" in result.lower()
    assert "no negative" not in result.lower()


def test_combine_both_parts():
    result = combine_review_parts("Dirty room", "Great staff")
    assert "dirty" in result.lower()
    assert "great" in result.lower()


# ───────────────────────────── preprocess_text ───────────────────────


def test_preprocess_text_returns_string():
    assert isinstance(preprocess_text("Hello World 123!"), str)


def test_preprocess_text_lowercases():
    assert preprocess_text("HOTEL") == preprocess_text("hotel")


# ───────────────────────────── clean_reviews ─────────────────────────


def test_clean_reviews_adds_rating_column(raw_df):
    result = clean_reviews(raw_df)
    assert "rating" in result.columns


def test_clean_reviews_rating_in_range(raw_df):
    result = clean_reviews(raw_df)
    assert result["rating"].between(1, 5).all()


def test_clean_reviews_adds_review_text(raw_df):
    result = clean_reviews(raw_df)
    assert "review_text" in result.columns
    assert result["review_text"].notna().all()


def test_clean_reviews_deduplication(raw_df):
    """Last two rows have same date; verifies dedup doesn't break counts."""
    result = clean_reviews(raw_df)
    assert len(result) >= 1


def test_clean_reviews_date_column(raw_df):
    result = clean_reviews(raw_df)
    assert "date" in result.columns
    assert pd.api.types.is_datetime64_any_dtype(result["date"])


def test_clean_reviews_adds_word_count(raw_df):
    result = clean_reviews(raw_df)
    assert "word_count" in result.columns
    assert (result["word_count"] >= 0).all()


# ───────────────────────────── split_train_test ──────────────────────


def test_split_correct_sizes(raw_df):
    cleaned = clean_reviews(raw_df)
    train, test = split_train_test(cleaned, test_size=0.33)
    total = len(train) + len(test)
    assert total == len(cleaned)


def test_split_no_overlap(raw_df):
    cleaned = clean_reviews(raw_df)
    train, test = split_train_test(cleaned, test_size=0.5)
    common = set(train.index) & set(test.index)
    assert len(common) == 0
