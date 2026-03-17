"""
HotelInsight - Main Dashboard Page
=====================================
Displays the top-level KPIs, rating trend, topic frequency chart, and the
ranked list of top issues for the selected hotel.
"""

import streamlit as st


# ---------------------------------------------------------------------------
# Cached analysis helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Running hotel analysis…")
def _run_analysis(hotel_name: str):
    from src.analysis.pattern_detector import analyze_hotel
    return analyze_hotel(hotel_name)


@st.cache_data(show_spinner="Analysing trends…")
def _run_trends(hotel_name: str):
    from src.analysis.pattern_detector import analyze_trends
    return analyze_trends(hotel_name)


@st.cache_data(show_spinner="Calculating impact scores…")
def _run_impacts(hotel_name: str):
    from src.analysis.pattern_detector import get_all_impacts
    return get_all_impacts(hotel_name)


@st.cache_data(show_spinner="Loading reviews…")
def _load_hotel(hotel_name: str):
    from src.data_processing.data_loader import load_hotel_by_name
    from src.data_processing.data_cleaner import clean_reviews
    from src.analysis.pattern_detector import _ensure_topic_columns
    raw = load_hotel_by_name(hotel_name)
    cleaned = clean_reviews(raw)
    return _ensure_topic_columns(cleaned)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def show() -> None:
    """Render the main dashboard."""

    if "selected_hotel" not in st.session_state:
        st.warning("⚠️ No hotel selected.  Please go to **Hotel Selection** first.")
        return

    hotel_name = st.session_state["selected_hotel"]
    st.title(f"📊 Dashboard: {hotel_name}")
    st.markdown("---")

    # ------------------------------------------------------------------
    # Run analyses (cached after first call)
    # ------------------------------------------------------------------
    try:
        analysis = _run_analysis(hotel_name)
        trends   = _run_trends(hotel_name)
        impacts  = _run_impacts(hotel_name)
        hotel_df = _load_hotel(hotel_name)
    except FileNotFoundError:
        st.error("⚠️ Dataset file not found.  Please check `data/raw/Hotel_Reviews.csv`.")
        return
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        return

    total = analysis.get("total_reviews", 0)
    if total == 0:
        st.warning("No reviews found for this hotel.")
        return

    # ------------------------------------------------------------------
    # Top KPI row
    # ------------------------------------------------------------------
    avg_rating = analysis.get("avg_rating", 0.0)
    # Convert Kaggle 1-10 scale to 1-5 if needed (data_cleaner already does this)
    health_score = int(min(avg_rating / 5 * 100, 100))
    avg_sentiment = analysis.get("avg_sentiment", 0.0)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Health Score", f"{health_score}/100")
        if health_score >= 80:
            st.success("Excellent")
        elif health_score >= 60:
            st.warning("Needs Improvement")
        else:
            st.error("Critical")

    with col2:
        st.metric("⭐ Average Rating", f"{avg_rating:.2f} / 5")

    with col3:
        st.metric("📝 Total Reviews", f"{total:,}")

    with col4:
        sentiment_label = "😊 Positive" if avg_sentiment > 0.3 else "😐 Neutral" if avg_sentiment > -0.3 else "😞 Negative"
        st.metric("Avg Sentiment", f"{avg_sentiment:.2f}", delta=sentiment_label)

    st.markdown("---")

    # ------------------------------------------------------------------
    # Rating trend chart
    # ------------------------------------------------------------------
    st.subheader("📈 Rating Trend")

    if not hotel_df.empty and "date" in hotel_df.columns and "rating" in hotel_df.columns:
        from src.visualization.charts import rating_trend_chart
        fig = rating_trend_chart(hotel_df)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough date/rating data for trend chart.")

    st.markdown("---")

    # ------------------------------------------------------------------
    # Complaint frequency chart
    # ------------------------------------------------------------------
    col_left, col_right = st.columns(2)

    topic_stats = analysis.get("topic_stats", {})

    with col_left:
        st.subheader("📊 Complaint Frequency")
        if topic_stats:
            from src.visualization.charts import complaint_frequency_bar
            fig2 = complaint_frequency_bar(topic_stats)
            st.plotly_chart(fig2, use_container_width=True)

    with col_right:
        st.subheader("⚡ Rating Impact")
        if impacts:
            from src.analysis.impact_calculator import build_impact_table
            impact_table = build_impact_table(hotel_df)
            if not impact_table.empty:
                from src.visualization.charts import impact_heatmap
                fig3 = impact_heatmap(impact_table)
                st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------------------------
    # Top issues table (sorted by priority score)
    # ------------------------------------------------------------------
    st.subheader("🚨 Top Issues — Ranked by Priority")

    from src.analysis.pattern_detector import calculate_priority_score
    from src.planning.priority_scorer import priority_badge

    priorities = []
    for topic, stats in topic_stats.items():
        if stats.get("percentage", 0) < 5:
            continue
        trend_info = trends.get(topic, {})
        impact_val = impacts.get(topic, 0.0)
        priority = calculate_priority_score(
            stats["percentage"],
            impact_val,
            trend_info.get("trend", "STABLE"),
        )
        priorities.append(
            {
                "topic": topic,
                "frequency": stats["percentage"],
                "impact": impact_val,
                "trend": trend_info.get("trend", "STABLE"),
                "trend_icon": trend_info.get("trend_icon", "→"),
                "priority_score": priority["score"],
                "priority_category": priority["category"],
            }
        )

    priorities.sort(key=lambda x: x["priority_score"], reverse=True)

    if not priorities:
        st.info("No issues above 5% frequency threshold.")
    else:
        for idx, issue in enumerate(priorities[:6], 1):
            cat = issue["priority_category"]
            badge = priority_badge(cat)

            with st.expander(
                f"{idx}. {badge} **{issue['topic'].upper()}** — {cat}  "
                f"(Score: {issue['priority_score']:.0f})",
                expanded=(idx == 1),
            ):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Frequency", f"{issue['frequency']:.1f}%")
                c2.metric("Rating Impact", f"-{issue['impact']:.2f} ⭐")
                c3.metric("Trend", f"{issue['trend_icon']} {issue['trend']}")
                c4.metric("Priority Score", f"{issue['priority_score']:.0f}/100")

                # Sample reviews for this topic
                col_tag = f"has_{issue['topic']}"
                if col_tag in hotel_df.columns:
                    samples = hotel_df[hotel_df[col_tag] == True].head(3)
                    if not samples.empty:
                        st.markdown("**Sample Reviews:**")
                        for _, row in samples.iterrows():
                            review_text = str(row.get("review_text", ""))[:200]
                            rating_val = row.get("rating", 0)
                            stars = "⭐" * max(1, min(5, int(round(rating_val))))
                            st.markdown(
                                f'<div class="review-quote">{stars} "{review_text}…"</div>',
                                unsafe_allow_html=True,
                            )

                # Navigation buttons
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button(
                        "🔍 Deep Dive",
                        key=f"detail_{issue['topic']}",
                        help="Go to Detailed Analysis page for this issue",
                    ):
                        st.session_state["selected_issue"] = issue["topic"]
                        st.info("Navigate to **Detailed Analysis** in the sidebar.")
                with btn_col2:
                    if st.button(
                        "📋 Action Plan",
                        key=f"action_{issue['topic']}",
                        help="Go to Action Plans page for this issue",
                    ):
                        st.session_state["selected_issue"] = issue["topic"]
                        st.info("Navigate to **Action Plans** in the sidebar.")
