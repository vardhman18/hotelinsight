"""
HotelInsight - Data Cleaner
==============================
Cleans and preprocesses the raw review DataFrame produced by
:mod:`src.data_processing.data_loader`.  All transformations are logged
so the pipeline is fully auditable.
"""

import re
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger
from src.utils.text_processing import clean_text, combine_review_parts, word_count

logger = get_logger(__name__)

# Placeholder strings that indicate an empty review in the Kaggle dataset
_EMPTY_PLACEHOLDERS = frozenset(
    [
        "no negative",
        "nothing",
        "none",
        "n/a",
        "no positive",
        "nothing to note",
        "no comment",
        "no complaints",
        "nothing negative",
        "nothing positive",
    ]
)


def _is_empty_review(text: str) -> bool:
    """Return ``True`` if *text* is a placeholder or genuinely empty."""
    if not isinstance(text, str):
        return True
    return text.strip().lower() in _EMPTY_PLACEHOLDERS or not text.strip()


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the raw review DataFrame.

    Applies the following steps in order:

    1. Remove exact-duplicate rows.
    2. Combine ``Negative_Review`` and ``Positive_Review`` into one
       ``review_text`` column (ignoring placeholder strings).
    3. Drop rows whose combined ``review_text`` is empty.
    4. Ensure ``Review_Date`` is ``datetime64`` (already done by the loader,
       but re-applied defensively).
    5. Add ``review_length`` (character count) and ``word_count`` columns.
    6. Add ``rating`` as a 0–10 → 1–5 normalised column if not present.
    7. Reset the index.

    Args:
        df: Raw DataFrame from :func:`~src.data_processing.data_loader.load_hotel_reviews`.

    Returns:
        Cleaned DataFrame with additional feature columns.
    """
    logger.info("Starting review cleaning.  Input rows: %d", len(df))

    # Step 1 – Remove duplicates
    before = len(df)
    df = df.drop_duplicates().copy()
    logger.info("Dropped %d duplicate rows.", before - len(df))

    # Step 2 – Combine review parts
    if "Negative_Review" in df.columns and "Positive_Review" in df.columns:
        df["review_text"] = df.apply(
            lambda row: combine_review_parts(
                row.get("Negative_Review", ""),
                row.get("Positive_Review", ""),
            ),
            axis=1,
        )
    elif "review_text" not in df.columns:
        df["review_text"] = ""

    # Step 3 – Drop rows with no usable review text
    before = len(df)
    df = df[df["review_text"].str.strip().astype(bool)].copy()
    logger.info("Dropped %d rows with empty review text.", before - len(df))

    # Step 4 – Ensure datetime column
    if "Review_Date" in df.columns:
        df["Review_Date"] = pd.to_datetime(df["Review_Date"], errors="coerce")

    # Convenience alias: use lowercase 'date' throughout the app
    if "Review_Date" in df.columns and "date" not in df.columns:
        df["date"] = df["Review_Date"]

    # Step 5 – Text feature columns
    df["review_length"] = df["review_text"].str.len()
    df["word_count"] = df["review_text"].apply(word_count)

    # Step 6 – Normalise rating to 1–5 scale
    if "Reviewer_Score" in df.columns:
        # The Kaggle dataset uses a 1–10 scale; convert to 1–5
        df["rating"] = (df["Reviewer_Score"] / 2).clip(lower=1.0, upper=5.0)
    elif "rating" not in df.columns:
        df["rating"] = float("nan")

    # Step 7 – Reset index
    df = df.reset_index(drop=True)
    logger.info("Cleaned dataset: %d rows remaining.", len(df))
    return df


def preprocess_text(text: str) -> str:
    """Preprocess a single review text for NLP tasks.

    This is a convenience wrapper around :func:`~src.utils.text_processing.clean_text`
    that also strips very common noise specific to the hotel-review domain
    (e.g. leading "I stayed at this hotel…" preambles are left intact to
    preserve semantic context, but HTML entities and excess punctuation are
    removed).

    Steps:
    1. Delegate to :func:`~src.utils.text_processing.clean_text` for
       lowercasing, URL removal, and special-character removal.
    2. Collapse repeated punctuation (``!!!`` → ``!``).
    3. Strip the result.

    Args:
        text: Raw review string.

    Returns:
        Preprocessed text string.
    """
    cleaned = clean_text(text)
    # Collapse repeated punctuation
    cleaned = re.sub(r"([!?.,])\1+", r"\1", cleaned)
    return cleaned.strip()


def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset into training and test sets.

    Stratification is applied on ``rating`` (binned to integer stars) when
    the column is available, so both splits have a similar rating
    distribution.

    Args:
        df: Cleaned DataFrame to split.
        test_size: Fraction of rows reserved for the test set (default 0.2).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of ``(train_df, test_df)``.
    """
    stratify_col = None
    if "rating" in df.columns and df["rating"].notna().all():
        # Bin ratings to integer stars for stratification
        df = df.copy()
        df["_rating_bin"] = df["rating"].round().astype(int)
        # Only stratify if every class has at least 2 members
        if df["_rating_bin"].value_counts().min() >= 2:
            stratify_col = df["_rating_bin"]

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )

    # Drop the temporary bin column if created
    for split in (train_df, test_df):
        if "_rating_bin" in split.columns:
            split.drop(columns=["_rating_bin"], inplace=True)

    logger.info(
        "Train/test split: %d train rows, %d test rows (%.0f/%d%%)",
        len(train_df),
        len(test_df),
        (1.0 - test_size) * 100,
        int(test_size * 100),
    )
    # Preserve original indices to guarantee train/test index disjointness.
    return train_df, test_df
