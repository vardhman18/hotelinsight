"""
HotelInsight - Detailed Analysis Page
========================================
Deep-dive view for a single complaint topic: root cause breakdown,
evidence quotes, rating distribution, and trend chart.
"""

import streamlit as st


@st.cache_data(show_spinner="Loading reviews…")
def _load_hotel(hotel_name: str):
    from src.data_processing.data_loader import load_hotel_by_name
    from src.data_processing.data_cleaner import clean_reviews
    from src.analysis.pattern_detector import _ensure_topic_columns
    raw = load_hotel_by_name(hotel_name)
    cleaned = clean_reviews(raw)
    return _ensure_topic_columns(cleaned)


@st.cache_data(show_spinner="Inferring root causes…")
def _get_root_causes(hotel_name: str, topic: str):
    from src.analysis.root_cause_analyzer import infer_root_causes
    hotel_df = _load_hotel(hotel_name)
    return infer_root_causes(hotel_df, topic)


@st.cache_data(show_spinner="Analysing trends…")
def _get_trends(hotel_name: str):
    from src.analysis.pattern_detector import analyze_trends
    return analyze_trends(hotel_name)


def show() -> None:
    """Render the detailed analysis page."""

    if "selected_hotel" not in st.session_state:
        st.warning("⚠️ Please select a hotel first.")
        return

    hotel_name = st.session_state["selected_hotel"]
    st.header("🔍 Detailed Analysis")
    st.caption(f"Hotel: **{hotel_name}**")
    st.markdown("---")

    # Topic selector
    from src.config.settings import COMPLAINT_CATEGORIES
    if (
        "selected_issue" not in st.session_state
        or st.session_state["selected_issue"] not in COMPLAINT_CATEGORIES
    ):
        st.session_state["selected_issue"] = COMPLAINT_CATEGORIES[0]

    st.selectbox(
        "Select complaint topic to analyse:",
        COMPLAINT_CATEGORIES,
        key="selected_issue",
        format_func=lambda t: f"{t.capitalize()}",
    )
    topic = st.session_state["selected_issue"]

    st.markdown("---")

    # ------------------------------------------------------------------
    # Load data and compute analysis
    # ------------------------------------------------------------------
    try:
        hotel_df = _load_hotel(hotel_name)
        root_causes = _get_root_causes(hotel_name, topic)
        trends = _get_trends(hotel_name)
    except FileNotFoundError:
        st.error("⚠️ Dataset not found.")
        return
    except Exception as exc:
        st.error(f"Analysis error: {exc}")
        return

    col_tag = f"has_{topic}"

    # Topic frequency
    total = len(hotel_df)
    topic_count = int(hotel_df[col_tag].sum()) if col_tag in hotel_df.columns else 0
    topic_pct = topic_count / total * 100 if total > 0 else 0.0

    # ------------------------------------------------------------------
    # KPI row for this topic
    # ------------------------------------------------------------------
    c1, c2, c3 = st.columns(3)
    c1.metric("Complaint Frequency", f"{topic_pct:.1f}%", f"{topic_count:,} / {total:,} reviews")

    topic_trend = trends.get(topic, {})
    trend_label = topic_trend.get("trend", "N/A")
    trend_icon = topic_trend.get("trend_icon", "→")
    c2.metric("Trend", f"{trend_icon} {trend_label}")

    from src.analysis.pattern_detector import calculate_impact
    impact = calculate_impact(hotel_df, topic)
    c3.metric("Rating Impact", f"-{impact:.2f} ⭐", "when this topic is mentioned")

    st.markdown("---")

    # ------------------------------------------------------------------
    # Issue detail figure
    # ------------------------------------------------------------------
    from src.visualization.dashboards import issue_detail_figure
    fig = issue_detail_figure(hotel_df, topic, trends)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------------------------
    # Root cause breakdown
    # ------------------------------------------------------------------
    st.subheader(f"🔬 Root Cause Analysis — {topic.capitalize()}")

    if not root_causes:
        st.info(
            f"Not enough keyword evidence in the reviews to infer root causes for **{topic}**. "
            "Try a topic with more complaints."
        )
    else:
        for cause_info in root_causes:
            cause = cause_info["cause"]
            conf = cause_info["confidence"]
            desc = cause_info.get("description", "")
            evidence = cause_info.get("evidence", [])

            # Colour-code by confidence
            if conf >= 60:
                colour = "🔴"
            elif conf >= 30:
                colour = "🟠"
            else:
                colour = "🟡"

            with st.expander(
                f"{colour} **{cause.replace('_', ' ')}** — {conf:.1f}% confidence",
                expanded=(conf >= 40),
            ):
                st.markdown(f"*{desc}*")
                st.progress(int(conf))

                if evidence:
                    st.markdown("**Evidence:**")
                    for point in evidence:
                        st.markdown(f"- {point}")

                st.markdown(
                    f"➡️ To see recommended actions for this root cause, go to **Action Plans**."
                )

    st.markdown("---")

    # ------------------------------------------------------------------
    # Sample reviews for this topic
    # ------------------------------------------------------------------
    st.subheader(f"📝 Sample Reviews Mentioning {topic.capitalize()}")

    if col_tag in hotel_df.columns:
        samples = hotel_df[hotel_df[col_tag] == True].copy()

        if samples.empty:
            st.info("No reviews found for this topic.")
        else:
            # Filter controls
            n_samples = st.slider("Number of reviews to show:", 3, 20, 5)
            low_rated = st.checkbox("Show only low-rated reviews (≤ 3 stars)", value=True)

            if low_rated and "rating" in samples.columns:
                display = samples[samples["rating"] <= 3].head(n_samples)
            else:
                display = samples.head(n_samples)

            for _, row in display.iterrows():
                rating_val = row.get("rating", 0)
                stars = "⭐" * max(1, min(5, int(round(rating_val))))
                text = str(row.get("review_text", ""))[:350]
                date_val = row.get("date", "")
                date_str = str(date_val)[:10] if date_val else ""
                st.markdown(
                    f'<div class="review-quote">{stars} ({date_str}) "{text}…"</div>',
                    unsafe_allow_html=True,
                )
