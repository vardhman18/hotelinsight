"""
HotelInsight - App-Level Chart Wrappers
=========================================
Thin wrappers that call src.visualization.charts and render to Streamlit.
"""

import streamlit as st


def rating_trend(hotel_df, key: str = "rt") -> None:
    from src.visualization.charts import rating_trend_chart
    st.plotly_chart(rating_trend_chart(hotel_df), use_container_width=True, key=key)


def complaint_frequency(topic_stats: dict, key: str = "cf") -> None:
    from src.visualization.charts import complaint_frequency_bar
    st.plotly_chart(complaint_frequency_bar(topic_stats), use_container_width=True, key=key)


def sentiment_pie(labels: list, key: str = "sp") -> None:
    from src.visualization.charts import sentiment_distribution_pie
    st.plotly_chart(sentiment_distribution_pie(labels), use_container_width=True, key=key)


def impact_heat(impact_table, key: str = "ih") -> None:
    from src.visualization.charts import impact_heatmap
    st.plotly_chart(impact_heatmap(impact_table), use_container_width=True, key=key)


def trend_lines(trends: dict, key: str = "tl") -> None:
    from src.visualization.charts import trend_line_chart
    st.plotly_chart(trend_line_chart(trends), use_container_width=True, key=key)


def roi_waterfall(roi_data: dict, key: str = "rw") -> None:
    from src.visualization.charts import roi_waterfall_chart
    st.plotly_chart(roi_waterfall_chart(roi_data), use_container_width=True, key=key)
