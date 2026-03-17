"""
HotelInsight - Data Loader
============================
Handles loading of hotel review datasets from CSV files.  All file I/O is
centralised here so other modules never need to know the on-disk layout.
"""

import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.config.settings import (
    MAIN_DATASET_PATH,
    RAW_DATA_DIR,
)
from src.utils.date_utils import safe_to_datetime
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Optional in-memory dataset override used by the Streamlit UI when a user
# uploads their own file for analysis.
_RUNTIME_DATASET_DF: Optional[pd.DataFrame] = None
_RUNTIME_DATASET_NAME: str = ""

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dataset_path(dataset: str) -> str:
    """Resolve the filesystem path for a named dataset.

    Args:
        dataset: ``"main"`` for the 515K Kaggle European hotels dataset.

    Returns:
        Absolute path string.

    Raises:
        ValueError: If *dataset* is not a recognised name.
    """
    paths = {
        "main": MAIN_DATASET_PATH,
    }
    if dataset not in paths:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Valid choices: {list(paths.keys())}"
        )
    return paths[dataset]


def _pick_first(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first column from *df* that matches *candidates* (case-insensitive)."""
    lower_to_original = {c.lower().strip(): c for c in df.columns}
    for candidate in candidates:
        col = lower_to_original.get(candidate.lower().strip())
        if col:
            return col
    return None


def _normalise_uploaded_columns(df: pd.DataFrame, fallback_hotel_name: str) -> pd.DataFrame:
    """Map common user-upload column names to the app's canonical schema."""
    hotel_col = _pick_first(df, ["Hotel_Name", "hotel_name", "hotel", "property", "property_name", "hotel name"])
    date_col = _pick_first(df, ["Review_Date", "review_date", "date", "reviewed_at", "review date"])
    score_col = _pick_first(df, ["Reviewer_Score", "reviewer_score", "rating", "score", "review_score", "stars"])
    neg_col = _pick_first(df, ["Negative_Review", "negative_review", "negative", "cons", "complaint"])
    pos_col = _pick_first(df, ["Positive_Review", "positive_review", "positive", "pros", "praise"])
    text_col = _pick_first(df, ["review_text", "review", "comment", "feedback", "review body"])
    address_col = _pick_first(df, ["Hotel_Address", "hotel_address", "address", "location"])

    out = pd.DataFrame(index=df.index)

    if hotel_col:
        out["Hotel_Name"] = df[hotel_col]
    else:
        out["Hotel_Name"] = fallback_hotel_name

    if date_col:
        out["Review_Date"] = df[date_col]
    else:
        out["Review_Date"] = pd.NaT

    if score_col:
        score = pd.to_numeric(df[score_col], errors="coerce")
        # The cleaner expects Kaggle-style 1-10 scores before converting to 1-5.
        if score.notna().any() and score.max(skipna=True) <= 5:
            score = score * 2
        out["Reviewer_Score"] = score
    else:
        out["Reviewer_Score"] = float("nan")

    if neg_col:
        out["Negative_Review"] = df[neg_col]
    else:
        out["Negative_Review"] = ""

    if pos_col:
        out["Positive_Review"] = df[pos_col]
    elif text_col:
        out["Positive_Review"] = df[text_col]
    else:
        out["Positive_Review"] = ""

    if address_col:
        out["Hotel_Address"] = df[address_col]
    else:
        out["Hotel_Address"] = ""

    # Standardise to the same dtypes expected from the Kaggle CSV path.
    out["Hotel_Name"] = out["Hotel_Name"].fillna(fallback_hotel_name).astype(str)
    out["Negative_Review"] = out["Negative_Review"].fillna("").astype(str)
    out["Positive_Review"] = out["Positive_Review"].fillna("").astype(str)
    out["Hotel_Address"] = out["Hotel_Address"].fillna("").astype(str)
    out["Review_Date"] = safe_to_datetime(out["Review_Date"])
    out["Reviewer_Score"] = pd.to_numeric(out["Reviewer_Score"], errors="coerce")

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_hotel_reviews(dataset: str = "main") -> pd.DataFrame:
    """Load the hotel reviews dataset from a CSV file.

    Standardises column names so downstream modules can rely on a fixed
    schema regardless of which dataset variant is loaded.

    The returned DataFrame always contains:

    - ``Hotel_Name`` (str)
    - ``Review_Date`` (datetime64)
    - ``Reviewer_Score`` (float)
    - ``Negative_Review`` (str)
    - ``Positive_Review`` (str)
    - ``Hotel_Address`` (str)

    Args:
        dataset: ``"main"`` for the 515K Kaggle dataset.

    Returns:
        pandas DataFrame with standardised columns.

    Raises:
        FileNotFoundError: If the CSV file does not exist at the expected
            path.  Download the file first using ``scripts/download_data.py``.
    """
    if _RUNTIME_DATASET_DF is not None:
        logger.info("Loading reviews from runtime dataset override: %s", _RUNTIME_DATASET_NAME)
        return _RUNTIME_DATASET_DF.copy()

    path = _dataset_path(dataset)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "Download it from Kaggle (515K Hotel Reviews) and place it in "
            f"'{RAW_DATA_DIR}'."
        )

    logger.info("Loading '%s' dataset from: %s", dataset, path)

    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded %d rows, %d columns.", len(df), len(df.columns))

    # ------------------------------------------------------------------
    # Normalise the Kaggle column names to the project schema.
    # The raw CSV contains underscored names; map any deviations here.
    # ------------------------------------------------------------------
    rename_map = {
        "Hotel_Name": "Hotel_Name",
        "Review_Date": "Review_Date",
        "Reviewer_Score": "Reviewer_Score",
        "Negative_Review": "Negative_Review",
        "Positive_Review": "Positive_Review",
        "Hotel_Address": "Hotel_Address",
    }

    # Keep only recognised columns (extras are allowed through for exploration)
    present = {c: rename_map[c] for c in rename_map if c in df.columns}
    df.rename(columns=present, inplace=True)

    # Ensure required columns exist with sensible defaults
    required_str_cols = [
        "Hotel_Name", "Negative_Review", "Positive_Review", "Hotel_Address"
    ]
    for col in required_str_cols:
        if col not in df.columns:
            df[col] = ""
            logger.warning("Column '%s' missing — filled with empty strings.", col)

    if "Review_Date" in df.columns:
        df["Review_Date"] = safe_to_datetime(df["Review_Date"])
    else:
        df["Review_Date"] = pd.NaT
        logger.warning("'Review_Date' column missing — set to NaT.")

    if "Reviewer_Score" not in df.columns:
        df["Reviewer_Score"] = float("nan")
        logger.warning("'Reviewer_Score' column missing — set to NaN.")

    # Cast score to float
    df["Reviewer_Score"] = pd.to_numeric(df["Reviewer_Score"], errors="coerce")

    # Fill string NaN values
    for col in required_str_cols:
        df[col] = df[col].fillna("").astype(str)

    return df


def set_runtime_dataset(df: pd.DataFrame, dataset_name: str = "Uploaded dataset") -> None:
    """Enable in-memory dataset override for all loader calls in this process."""
    global _RUNTIME_DATASET_DF, _RUNTIME_DATASET_NAME
    if df is None or df.empty:
        raise ValueError("Uploaded dataset is empty.")

    normalised = _normalise_uploaded_columns(df.copy(), fallback_hotel_name="Uploaded Hotel")
    _RUNTIME_DATASET_DF = normalised.reset_index(drop=True)
    _RUNTIME_DATASET_NAME = dataset_name.strip() or "Uploaded dataset"
    logger.info(
        "Runtime dataset enabled (%s): %d rows, %d columns.",
        _RUNTIME_DATASET_NAME,
        len(_RUNTIME_DATASET_DF),
        len(_RUNTIME_DATASET_DF.columns),
    )


def clear_runtime_dataset() -> None:
    """Disable the in-memory dataset override and fall back to the default CSV."""
    global _RUNTIME_DATASET_DF, _RUNTIME_DATASET_NAME
    _RUNTIME_DATASET_DF = None
    _RUNTIME_DATASET_NAME = ""
    logger.info("Runtime dataset override cleared.")


def is_runtime_dataset_active() -> bool:
    """Return True if an uploaded runtime dataset is currently active."""
    return _RUNTIME_DATASET_DF is not None


def get_runtime_dataset_name() -> str:
    """Return current runtime dataset label, if any."""
    return _RUNTIME_DATASET_NAME


def load_hotel_by_name(hotel_name: str) -> pd.DataFrame:
    """Load all reviews for a specific hotel.

    Args:
        hotel_name: Exact hotel name as it appears in the dataset.

    Returns:
        DataFrame containing only reviews for the specified hotel.
        Returns an empty DataFrame with the standard schema if the hotel
        is not found.
    """
    df = load_hotel_reviews()
    hotel_df = df[df["Hotel_Name"].str.strip() == hotel_name.strip()].copy()

    if hotel_df.empty:
        logger.warning("No reviews found for hotel: '%s'", hotel_name)
    else:
        logger.info(
            "Loaded %d reviews for '%s'.", len(hotel_df), hotel_name
        )

    return hotel_df.reset_index(drop=True)


def get_hotel_list() -> List[str]:
    """Return a sorted list of all unique hotel names in the dataset.

    Returns:
        Alphabetically sorted list of hotel name strings.
    """
    df = load_hotel_reviews()
    hotels = sorted(df["Hotel_Name"].dropna().str.strip().unique().tolist())
    logger.info("Found %d unique hotels.", len(hotels))
    return hotels


def get_hotel_stats(hotel_name: str) -> Dict:
    """Compute basic statistics for a single hotel.

    Args:
        hotel_name: Exact hotel name.

    Returns:
        Dictionary with keys:

        - ``total_reviews`` (int)
        - ``avg_rating`` (float) – rounded to 2 decimal places
        - ``date_range`` (Tuple[str, str]) – earliest and latest review dates
          as ``"YYYY-MM-DD"`` strings, or ``("N/A", "N/A")`` if unavailable.
    """
    hotel_df = load_hotel_by_name(hotel_name)

    if hotel_df.empty:
        return {
            "total_reviews": 0,
            "avg_rating": 0.0,
            "date_range": ("N/A", "N/A"),
        }

    total_reviews = len(hotel_df)
    avg_rating = round(hotel_df["Reviewer_Score"].mean(), 2)

    valid_dates = hotel_df["Review_Date"].dropna()
    if not valid_dates.empty:
        start_date = valid_dates.min().strftime("%Y-%m-%d")
        end_date = valid_dates.max().strftime("%Y-%m-%d")
    else:
        start_date, end_date = "N/A", "N/A"

    return {
        "total_reviews": total_reviews,
        "avg_rating": float(avg_rating),
        "date_range": (start_date, end_date),
    }
