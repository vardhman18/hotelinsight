"""
HotelInsight - Reusable Metric Card Components
================================================
Thin wrappers around st.metric with colour-coded delta logic.
"""

import streamlit as st


def rating_metric(label: str, value: float, previous: float | None = None) -> None:
    """Display a 1–5 rating metric with optional delta vs previous value."""
    delta = None
    if previous is not None:
        diff = round(value - previous, 2)
        delta = f"{'+' if diff >= 0 else ''}{diff}"
    st.metric(label, f"{value:.2f} ⭐", delta=delta)


def percentage_metric(label: str, value: float, inverted: bool = False) -> None:
    """Display a percentage metric. inverted=True means higher is worse."""
    colour_dir = "inverse" if inverted else "normal"
    st.metric(label, f"{value:.1f}%", delta_color=colour_dir)


def count_metric(label: str, count: int, icon: str = "") -> None:
    """Display a count metric."""
    st.metric(label, f"{icon}{count:,}")


def currency_metric(label: str, amount: float) -> None:
    """Display a currency (GBP) metric."""
    if amount >= 1_000_000:
        formatted = f"£{amount / 1_000_000:.1f}M"
    elif amount >= 1_000:
        formatted = f"£{amount / 1_000:.1f}K"
    else:
        formatted = f"£{amount:.0f}"
    st.metric(label, formatted)


def priority_metric(label: str, score: float) -> None:
    """Display a priority score with badge colour."""
    from src.planning.priority_scorer import priority_badge
    badge = priority_badge(score)
    st.metric(label, f"{score:.0f} / 100", delta=badge, delta_color="off")


def kpi_row(metrics: list[dict]) -> None:
    """
    Render a row of KPI metrics from a list of dicts.

    Each dict must have 'label' and 'value', and optionally 'delta', 'help'.
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            st.metric(
                label=m["label"],
                value=str(m["value"]),
                delta=m.get("delta"),
                help=m.get("help"),
            )
