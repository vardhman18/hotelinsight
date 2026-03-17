"""
HotelInsight - Reusable Filter UI Components
==============================================
Provides date-range pickers, rating sliders, and topic multi-selects
that can be embedded in any Streamlit page via sidebar or inline.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
from datetime import date, timedelta


def date_range_filter(
    df: pd.DataFrame,
    date_col: str = "date",
    sidebar: bool = True,
    key_prefix: str = "dr",
) -> pd.DataFrame:
    """
    Render a date-range selector and return a filtered DataFrame.

    Parameters
    ----------
    df : DataFrame with a datetime column named `date_col`
    date_col : column name containing datetime values
    sidebar : if True, render in st.sidebar; otherwise inline
    key_prefix : prefix for Streamlit widget keys (avoids conflicts)
    """
    target = st.sidebar if sidebar else st

    if date_col not in df.columns:
        return df

    min_date = df[date_col].min()
    max_date = df[date_col].max()

    if pd.isna(min_date) or pd.isna(max_date):
        return df

    min_date_py = min_date.date() if hasattr(min_date, "date") else min_date
    max_date_py = max_date.date() if hasattr(max_date, "date") else max_date

    default_start = max(min_date_py, max_date_py - timedelta(days=365))

    start, end = target.date_input(
        "Date range:",
        value=(default_start, max_date_py),
        min_value=min_date_py,
        max_value=max_date_py,
        key=f"{key_prefix}_date",
    )

    return df[(df[date_col].dt.date >= start) & (df[date_col].dt.date <= end)]


def rating_range_filter(
    df: pd.DataFrame,
    rating_col: str = "rating",
    sidebar: bool = True,
    key_prefix: str = "rr",
) -> pd.DataFrame:
    """
    Render a min/max rating slider and return a filtered DataFrame.
    Expects 1–5 star ratings.
    """
    target = st.sidebar if sidebar else st

    if rating_col not in df.columns:
        return df

    min_r, max_r = target.slider(
        "Rating range (⭐):",
        min_value=1.0,
        max_value=5.0,
        value=(1.0, 5.0),
        step=0.5,
        key=f"{key_prefix}_rating",
    )
    return df[(df[rating_col] >= min_r) & (df[rating_col] <= max_r)]


def topic_multiselect_filter(
    df: pd.DataFrame,
    sidebar: bool = True,
    key_prefix: str = "tm",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Render a multi-select for complaint categories.

    Returns
    -------
    (filtered_df, selected_topics)
    """
    from src.config.settings import COMPLAINT_CATEGORIES

    target = st.sidebar if sidebar else st

    available = [c for c in COMPLAINT_CATEGORIES if f"has_{c}" in df.columns]
    if not available:
        return df, []

    selected = target.multiselect(
        "Filter by topic:",
        options=available,
        default=[],
        format_func=lambda x: x.title(),
        key=f"{key_prefix}_topics",
    )

    if not selected:
        return df, available

    mask = pd.Series(False, index=df.index)
    for t in selected:
        mask |= df.get(f"has_{t}", pd.Series(False, index=df.index))
    return df[mask], selected


def all_sidebar_filters(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Apply all three filters (date range, rating, topic) from the sidebar
    and return the filtered DataFrame.
    """
    st.sidebar.markdown("### Filters")
    df = date_range_filter(df, date_col=date_col, sidebar=True, key_prefix="asf_dr")
    df = rating_range_filter(df, sidebar=True, key_prefix="asf_rr")
    df, _ = topic_multiselect_filter(df, sidebar=True, key_prefix="asf_tm")
    return df
