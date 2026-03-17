"""
HotelInsight - Reusable Table Components
==========================================
Renders formatted DataFrames inside Streamlit with consistent styling.
"""

import streamlit as st
import pandas as pd


def impact_table(impact_df: pd.DataFrame) -> None:
    """Render the impact analysis table with visual formatting."""
    if impact_df is None or impact_df.empty:
        st.info("No impact data available.")
        return

    display_df = impact_df.copy()
    if "impact" in display_df.columns:
        display_df["impact"] = display_df["impact"].apply(lambda x: f"{x:.3f}")
    if "complaint_rate" in display_df.columns:
        display_df["complaint_rate"] = display_df["complaint_rate"].apply(lambda x: f"{x:.1%}")
    if "avg_rating_with" in display_df.columns:
        display_df["avg_rating_with"] = display_df["avg_rating_with"].apply(lambda x: f"{x:.2f}")
    if "avg_rating_without" in display_df.columns:
        display_df["avg_rating_without"] = display_df["avg_rating_without"].apply(lambda x: f"{x:.2f}")

    st.dataframe(display_df, use_container_width=True, hide_index=True)


def root_cause_table(root_causes: list[dict]) -> None:
    """Render root cause inference results as a styled table."""
    if not root_causes:
        st.info("No root causes identified.")
        return

    rows = [
        {
            "Root Cause":   rc["cause"].replace("_", " ").title(),
            "Confidence":   f"{rc['confidence']:.0f}%",
            "Description":  rc.get("description", ""),
        }
        for rc in root_causes
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def reviews_sample_table(reviews: pd.DataFrame, max_rows: int = 10) -> None:
    """Render a sample of reviews with rating and date."""
    if reviews is None or reviews.empty:
        st.info("No reviews to display.")
        return

    display_cols = [c for c in ["date", "rating", "review_text"] if c in reviews.columns]
    sample = reviews[display_cols].head(max_rows).copy()
    if "rating" in sample.columns:
        sample["rating"] = sample["rating"].apply(lambda x: "⭐" * round(x))
    if "date" in sample.columns:
        sample["date"] = pd.to_datetime(sample["date"]).dt.strftime("%b %Y")
    st.dataframe(sample, use_container_width=True, hide_index=True)


def action_cost_summary(total_cost: dict) -> None:
    """Render a 3-column cost summary for an action plan."""
    c1, c2, c3 = st.columns(3)
    c1.metric("One-Time Investment",  f"£{total_cost.get('one_time', 0):,.0f}")
    c2.metric("Monthly Ongoing Cost", f"£{total_cost.get('monthly', 0):,.0f}")
    c3.metric("3-Month Total",        f"£{total_cost.get('three_month_total', 0):,.0f}")
