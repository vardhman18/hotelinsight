"""
HotelInsight - Date Utilities
================================
Helper functions for parsing, formatting, and grouping date values from hotel
review datasets.  The Kaggle dataset stores dates as strings in the format
``"MM/DD/YYYY"``; these helpers ensure consistent conversion throughout the
pipeline.
"""

from datetime import datetime
from typing import Optional, Tuple

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Date formats encountered in common hotel review datasets
_DATE_FORMATS = [
    "%m/%d/%Y",    # Kaggle 515K dataset  e.g. "08/03/2017"
    "%Y-%m-%d",    # ISO 8601
    "%d/%m/%Y",    # UK/European day-first
    "%B %d, %Y",   # "August 3, 2017"
    "%d %B %Y",    # "3 August 2017"
]


def parse_date(date_str: str) -> Optional[datetime]:
    """Try parsing a date string against several common formats.

    Args:
        date_str: Raw date string from the dataset.

    Returns:
        ``datetime`` object, or ``None`` if no format matches.
    """
    if not isinstance(date_str, str) or not date_str.strip():
        return None

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue

    logger.debug("Could not parse date string: %r", date_str)
    return None


def safe_to_datetime(series: pd.Series) -> pd.Series:
    """Convert a pandas Series of date strings to ``datetime64`` safely.

    Uses ``pd.to_datetime`` with ``errors='coerce'`` so unparseable values
    become ``NaT`` rather than raising.

    Args:
        series: Pandas Series of date strings.

    Returns:
        Series of ``datetime64`` values.
    """
    return pd.to_datetime(series, errors="coerce")


def get_year_month(dt: datetime) -> str:
    """Format a datetime as ``"YYYY-MM"`` for monthly grouping.

    Args:
        dt: A ``datetime`` or ``Timestamp`` object.

    Returns:
        String in ``"YYYY-MM"`` format.
    """
    return dt.strftime("%Y-%m")


def date_range_label(start: datetime, end: datetime) -> str:
    """Return a human-readable date range string.

    Args:
        start: Earliest date.
        end: Latest date.

    Returns:
        String like ``"Jan 2015 – Dec 2017"``.
    """
    return f"{start.strftime('%b %Y')} – {end.strftime('%b %Y')}"


def months_between(start: datetime, end: datetime) -> int:
    """Return the approximate number of whole months between two dates.

    Args:
        start: Earlier date.
        end: Later date.

    Returns:
        Non-negative integer count of months.
    """
    delta = (end.year - start.year) * 12 + (end.month - start.month)
    return max(delta, 0)


def filter_recent(df: pd.DataFrame, date_col: str, months: int = 12) -> pd.DataFrame:
    """Return only rows where *date_col* falls within the last *months* months.

    Args:
        df: DataFrame containing a datetime column.
        date_col: Name of the datetime column.
        months: Number of months to look back from the most recent date.

    Returns:
        Filtered DataFrame.  If the column contains no valid dates the
        original DataFrame is returned unchanged.
    """
    if date_col not in df.columns or df[date_col].isna().all():
        logger.warning("filter_recent: column '%s' has no valid dates.", date_col)
        return df

    max_date = df[date_col].max()
    # Use pandas DateOffset for accurate month arithmetic
    cutoff = max_date - pd.DateOffset(months=months)
    return df[df[date_col] >= cutoff].copy()
