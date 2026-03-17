"""
HotelInsight - Chart Generation
==================================
Reusable Plotly chart factory functions.  Every function returns a
``plotly.graph_objects.Figure`` that can be rendered in Streamlit with
``st.plotly_chart(fig, use_container_width=True)``.
"""

from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config.settings import COMPLAINT_CATEGORIES

# ---------------------------------------------------------------------------
# Colour palette (consistent across all charts)
# ---------------------------------------------------------------------------

_BRAND_BLUE = "#1f77b4"
_BRAND_RED = "#d62728"
_BRAND_GREEN = "#2ca02c"
_BRAND_ORANGE = "#ff7f0e"

_PRIORITY_COLOURS = {
    "CRITICAL": "#ff4444",
    "HIGH":     "#ff8c00",
    "MEDIUM":   "#ffd700",
    "LOW":      "#4caf50",
}


def rating_trend_chart(
    hotel_df: pd.DataFrame,
    date_col: str = "date",
    rating_col: str = "rating",
    title: str = "Monthly Average Rating",
) -> go.Figure:
    """Line chart of monthly average rating over time.

    Args:
        hotel_df: Reviews DataFrame with date and rating columns.
        date_col: Name of the date column.
        rating_col: Name of the rating column.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    df = hotel_df.copy()
    df["year_month"] = df[date_col].dt.to_period("M").astype(str)
    monthly = df.groupby("year_month")[rating_col].mean().reset_index()
    monthly.columns = ["Month", "Avg Rating"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly["Month"],
            y=monthly["Avg Rating"],
            mode="lines+markers",
            line=dict(color=_BRAND_BLUE, width=3),
            marker=dict(size=7),
            hovertemplate="<b>%{x}</b><br>Avg Rating: %{y:.2f} ⭐<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title="Average Rating",
        yaxis=dict(range=[1, 5.2]),
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def complaint_frequency_bar(
    topic_stats: Dict[str, Dict],
    title: str = "Complaint Frequency by Category",
) -> go.Figure:
    """Horizontal bar chart of complaint frequencies.

    Args:
        topic_stats: Dict as returned by ``analyze_hotel()['topic_stats']``.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    categories = list(topic_stats.keys())
    percentages = [topic_stats[c].get("percentage", 0) for c in categories]

    # Sort by frequency
    sorted_pairs = sorted(zip(categories, percentages), key=lambda p: p[1])
    categories, percentages = zip(*sorted_pairs) if sorted_pairs else ([], [])

    colours = [
        _BRAND_RED if p >= 30 else _BRAND_ORANGE if p >= 15 else _BRAND_BLUE
        for p in percentages
    ]

    fig = go.Figure(
        go.Bar(
            x=percentages,
            y=categories,
            orientation="h",
            marker_color=colours,
            text=[f"{p:.1f}%" for p in percentages],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="% of Reviews",
        yaxis_title="",
        height=350,
        margin=dict(l=100, r=60, t=50, b=40),
    )
    return fig


def sentiment_distribution_pie(
    labels: List[str],
    title: str = "Sentiment Distribution",
) -> go.Figure:
    """Pie chart showing positive / neutral / negative proportions.

    Args:
        labels: List of sentiment label strings (``"positive"``,
            ``"neutral"``, ``"negative"``).
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    from collections import Counter

    counts = Counter(labels)
    names = ["positive", "neutral", "negative"]
    values = [counts.get(n, 0) for n in names]
    colours = [_BRAND_GREEN, "#aaaaaa", _BRAND_RED]

    fig = go.Figure(
        go.Pie(
            labels=names,
            values=values,
            marker=dict(colors=colours),
            hole=0.4,
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
        )
    )
    fig.update_layout(title=title, height=320)
    return fig


def impact_heatmap(impact_table: pd.DataFrame) -> go.Figure:
    """Bar chart showing rating impact per topic, colour-coded by severity.

    Args:
        impact_table: DataFrame from
            :func:`~src.analysis.impact_calculator.build_impact_table`.

    Returns:
        Plotly Figure.
    """
    df = impact_table.sort_values("impact", ascending=False)

    colours = [
        _BRAND_RED if v >= 1.0 else _BRAND_ORANGE if v >= 0.5 else _BRAND_BLUE
        for v in df["impact"]
    ]

    fig = go.Figure(
        go.Bar(
            x=df["topic"],
            y=df["impact"],
            marker_color=colours,
            text=[f"-{v:.2f}★" for v in df["impact"]],
            textposition="outside",
            hovertemplate="%{x}: -%{y:.2f} stars<extra></extra>",
        )
    )
    fig.update_layout(
        title="Rating Impact by Complaint Category",
        xaxis_title="Category",
        yaxis_title="Star Rating Reduction",
        height=350,
        margin=dict(l=40, r=20, t=50, b=80),
    )
    return fig


def trend_line_chart(
    trends: Dict,
    selected_topics: Optional[List[str]] = None,
) -> go.Figure:
    """Multi-line chart showing complaint rate trends over time.

    Args:
        trends: Output of ``analyze_trends()`` – dict of topic → trend data.
        selected_topics: Topics to include.  Defaults to all available.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()
    topics = selected_topics or list(trends.keys())

    colour_cycle = px.colors.qualitative.Set2

    for i, topic in enumerate(topics):
        data = trends.get(topic, {})
        monthly = data.get("monthly_data", {})
        if not monthly:
            continue

        months = sorted(monthly.keys())
        rates = [monthly[m] for m in months]

        fig.add_trace(
            go.Scatter(
                x=months,
                y=rates,
                mode="lines+markers",
                name=topic.capitalize(),
                line=dict(color=colour_cycle[i % len(colour_cycle)], width=2),
                hovertemplate=f"<b>{topic}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>",
            )
        )

    fig.update_layout(
        title="Complaint Rate Trends",
        xaxis_title="Month",
        yaxis_title="Complaint Rate (%)",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def roi_waterfall_chart(roi_data: Dict) -> go.Figure:
    """Waterfall chart visualising the financial ROI breakdown.

    Args:
        roi_data: Dictionary returned by ``calculate_roi()``.

    Returns:
        Plotly Figure.
    """

    def _parse_gbp(s: str) -> float:
        """Strip '£' and commas from a formatted currency string."""
        return float(s.replace("£", "").replace(",", "").replace("-", ""))

    investment = _parse_gbp(roi_data.get("total_investment", "£0"))
    returns = _parse_gbp(roi_data.get("returns_3_months", "£0"))
    net = _parse_gbp(roi_data.get("net_profit_3_months", "£0"))

    fig = go.Figure(
        go.Waterfall(
            name="ROI",
            orientation="v",
            measure=["absolute", "relative", "total"],
            x=["Investment", "Revenue Uplift (3M)", "Net Profit"],
            y=[-investment, returns, net],
            connector={"line": {"color": "grey"}},
            decreasing={"marker": {"color": _BRAND_RED}},
            increasing={"marker": {"color": _BRAND_GREEN}},
            totals={"marker": {"color": _BRAND_BLUE}},
            text=[
                roi_data.get("total_investment", ""),
                roi_data.get("returns_3_months", ""),
                roi_data.get("net_profit_3_months", ""),
            ],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="3-Month Financial ROI Breakdown",
        yaxis_title="Amount (£)",
        height=380,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig
