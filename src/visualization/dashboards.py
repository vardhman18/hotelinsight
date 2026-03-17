"""
HotelInsight - Dashboard Component Builders
=============================================
Higher-level functions that combine multiple charts into cohesive dashboard
sections.  These wrap the low-level chart factories in
:mod:`src.visualization.charts` with layout and annotation logic.
"""

from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.visualization.charts import (
    complaint_frequency_bar,
    impact_heatmap,
    rating_trend_chart,
    sentiment_distribution_pie,
    trend_line_chart,
)


def overview_dashboard(
    hotel_df: pd.DataFrame,
    topic_stats: Dict,
    trends: Dict,
    impact_table: pd.DataFrame,
) -> Dict[str, go.Figure]:
    """Generate all figures needed for the main Overview dashboard.

    Args:
        hotel_df: Cleaned reviews DataFrame.
        topic_stats: Topic statistics dict from ``analyze_hotel()``.
        trends: Trend data dict from ``analyze_trends()``.
        impact_table: Impact table DataFrame from ``build_impact_table()``.

    Returns:
        Dictionary mapping figure names to Plotly Figure objects:

        - ``"rating_trend"``
        - ``"complaint_freq"``
        - ``"impact"``
        - ``"trend_lines"``
    """
    return {
        "rating_trend": rating_trend_chart(hotel_df),
        "complaint_freq": complaint_frequency_bar(topic_stats),
        "impact": impact_heatmap(impact_table),
        "trend_lines": trend_line_chart(trends),
    }


def issue_detail_figure(
    hotel_df: pd.DataFrame,
    topic: str,
    trends: Dict,
) -> go.Figure:
    """Two-panel figure for a single issue: rating distribution + trend.

    Args:
        hotel_df: Reviews DataFrame.
        topic: Issue topic name.
        trends: Trend data from ``analyze_trends()``.

    Returns:
        Plotly Figure with two side-by-side sub-plots.
    """
    col = f"has_{topic}"

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Rating Distribution — Mentions '{topic}'",
            f"'{topic}' Complaint Rate over Time",
        ),
    )

    # Left: histogram of ratings when topic is mentioned vs not
    if col in hotel_df.columns and "rating" in hotel_df.columns:
        with_topic = hotel_df[hotel_df[col] == True]["rating"]
        without_topic = hotel_df[hotel_df[col] == False]["rating"]

        fig.add_trace(
            go.Histogram(
                x=with_topic,
                name="With complaint",
                marker_color="#d62728",
                opacity=0.75,
                nbinsx=10,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Histogram(
                x=without_topic,
                name="Without complaint",
                marker_color="#1f77b4",
                opacity=0.75,
                nbinsx=10,
            ),
            row=1, col=1,
        )

    # Right: monthly trend line
    topic_trend = trends.get(topic, {})
    monthly = topic_trend.get("monthly_data", {})
    if monthly:
        months = sorted(monthly.keys())
        rates = [monthly[m] for m in months]
        fig.add_trace(
            go.Scatter(
                x=months,
                y=rates,
                mode="lines+markers",
                name="Complaint rate %",
                line=dict(color="#ff7f0e", width=2),
            ),
            row=1, col=2,
        )

    fig.update_layout(height=360, margin=dict(l=40, r=20, t=60, b=40))
    return fig
