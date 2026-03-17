"""
HotelInsight - Progress Tracker Page
=======================================
Compares complaint trends over time, showing rating trajectories and
month-over-month category improvements for the selected hotel.
"""

import streamlit as st
import pandas as pd


@st.cache_data(show_spinner="Loading hotel data…")
def _load_hotel(hotel_name: str):
    from src.data_processing.data_loader import load_hotel_by_name
    from src.data_processing.data_cleaner import clean_reviews
    from src.analysis.pattern_detector import _ensure_topic_columns
    raw = load_hotel_by_name(hotel_name)
    cleaned = clean_reviews(raw)
    return _ensure_topic_columns(cleaned)


@st.cache_data(show_spinner="Calculating trend data…")
def _get_trends(hotel_name: str):
    from src.analysis.pattern_detector import analyze_trends
    return analyze_trends(hotel_name, months=12)


def _trend_arrow(direction: str) -> str:
    return {"INCREASING": "🔴 Worsening", "DECREASING": "🟢 Improving", "STABLE": "🔵 Stable"}.get(
        direction, "—"
    )


def _monthly_ratings(hotel_df: pd.DataFrame) -> pd.DataFrame:
    """Return monthly average rating grouped by month."""
    df = hotel_df.copy()
    df["month"] = df["date"].dt.to_period("M")
    monthly = (
        df.groupby("month")["rating"]
        .mean()
        .reset_index()
    )
    monthly["month_dt"] = monthly["month"].dt.to_timestamp()
    return monthly.sort_values("month_dt")


def _monthly_complaints(hotel_df: pd.DataFrame, topic: str) -> pd.DataFrame:
    """Return monthly complaint rate for a given topic."""
    col = f"has_{topic}"
    if col not in hotel_df.columns:
        return pd.DataFrame()
    df = hotel_df.copy()
    df["month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("month")[col].mean().reset_index()
    monthly.columns = ["month", "complaint_rate"]
    monthly["month_dt"] = monthly["month"].dt.to_timestamp()
    monthly["complaint_rate"] = (monthly["complaint_rate"] * 100).round(1)
    return monthly.sort_values("month_dt")


def show() -> None:
    """Render the progress tracker page."""

    if "selected_hotel" not in st.session_state:
        st.warning("⚠️ Please select a hotel first.")
        return

    hotel_name = st.session_state["selected_hotel"]
    st.header("📈 Progress Tracker")
    st.caption(f"Hotel: **{hotel_name}** — 12-month trend analysis")
    st.markdown("---")

    # ── Load data ──────────────────────────────────────────────────────
    try:
        hotel_df = _load_hotel(hotel_name)
        trends   = _get_trends(hotel_name)
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        return

    # ── Rating trajectory chart ────────────────────────────────────────
    st.subheader("⭐ Rating Over Time")
    monthly = _monthly_ratings(hotel_df)

    if not monthly.empty:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=monthly["month_dt"],
                y=monthly["rating"],
                mode="lines+markers",
                name="Avg Rating",
                line=dict(color="#4CAF50", width=2),
            )
        )
        avg = monthly["rating"].mean()
        fig.add_hline(y=avg, line_dash="dash", line_color="red", annotation_text=f"Avg {avg:.2f}")
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Avg Rating (1–5)",
            yaxis=dict(range=[1, 5]),
            height=280,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Trend summary table ────────────────────────────────────────────
    st.subheader("📊 Complaint Trend by Category")
    st.caption("Comparing most-recent 3 months vs the 3 months before that.")

    trend_rows = []
    for topic, info in trends.items():
        direction = info.get("trend", "STABLE") if isinstance(info, dict) else info
        recent    = info.get("recent_rate", 0) * 100 if isinstance(info, dict) else 0
        previous  = info.get("previous_rate", 0) * 100 if isinstance(info, dict) else 0
        delta     = recent - previous
        trend_rows.append(
            {
                "Category":       topic.title(),
                "Recent Rate (%)":  f"{recent:.1f}%",
                "Previous Rate (%)": f"{previous:.1f}%",
                "Change":           f"{'+' if delta >= 0 else ''}{delta:.1f}%",
                "Trend":            _trend_arrow(direction),
            }
        )

    if trend_rows:
        st.dataframe(
            pd.DataFrame(trend_rows),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")

    # ── Drill-down chart for selected topic ────────────────────────────
    st.subheader("🔍 Monthly Complaint Rate Drill-Down")

    from src.config.settings import COMPLAINT_CATEGORIES

    selected_topic = st.selectbox("Select a category to inspect:", COMPLAINT_CATEGORIES)
    monthly_comp = _monthly_complaints(hotel_df, selected_topic)

    if not monthly_comp.empty:
        import plotly.graph_objects as go

        fig2 = go.Figure()
        fig2.add_trace(
            go.Bar(
                x=monthly_comp["month_dt"],
                y=monthly_comp["complaint_rate"],
                name="Complaint %",
                marker_color="#E57373",
            )
        )
        fig2.update_layout(
            xaxis_title="Month",
            yaxis_title=f"{selected_topic.title()} Complaints (%)",
            height=280,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info(f"No monthly data available for **{selected_topic}**.")

    # ── Recent metrics snapshot ────────────────────────────────────────
    st.markdown("---")
    st.subheader("📌 Last-30-Days Snapshot")

    from src.utils.date_utils import filter_recent
    recent_df = filter_recent(hotel_df, "date", months=1)

    if recent_df.empty:
        st.info("No reviews in the last 30 days.")
    else:
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Reviews",      len(recent_df))
        r2.metric("Avg Rating",   f"{recent_df['rating'].mean():.2f}")
        sentiments = recent_df.get("sentiment_score", pd.Series(dtype=float))
        r3.metric("Avg Sentiment", f"{sentiments.mean():.2f}" if not sentiments.empty else "N/A")
        r4.metric(
            "Positive %",
            f"{(sentiments > 0.3).mean() * 100:.0f}%"
            if not sentiments.empty
            else "N/A",
        )
