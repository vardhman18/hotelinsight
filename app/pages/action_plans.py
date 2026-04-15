"""
HotelInsight - Action Plans Page
===================================
Generates and displays structured action plans with costs for the selected
hotel and complaint topic, and computes the predicted financial ROI.
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


def show() -> None:
    """Render the action plans page."""

    if "selected_hotel" not in st.session_state:
        st.warning("⚠️ Please select a hotel first.")
        return

    hotel_name = st.session_state["selected_hotel"]
    st.header("📋 Action Plans")
    st.caption(f"Hotel: **{hotel_name}**")
    st.markdown("---")

    # ── Topic + room count selectors ──────────────────────────────────
    from src.config.settings import COMPLAINT_CATEGORIES

    if (
        "selected_issue" not in st.session_state
        or st.session_state["selected_issue"] not in COMPLAINT_CATEGORIES
    ):
        st.session_state["selected_issue"] = COMPLAINT_CATEGORIES[0]

    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.selectbox(
            "Complaint topic:",
            COMPLAINT_CATEGORIES,
            key="selected_issue",
            format_func=lambda t: f"{t.capitalize()}",
        )
        topic = st.session_state["selected_issue"]
    with col_right:
        num_rooms = st.number_input(
            "Number of rooms:", min_value=10, max_value=2000, value=100, step=10
        )

    st.markdown("---")

    # ── Load data & root causes ────────────────────────────────────────
    try:
        root_causes = _get_root_causes(hotel_name, topic)
    except Exception as exc:
        st.error(f"Root cause analysis failed: {exc}")
        return

    if any(bool(rc.get("fallback", False)) for rc in root_causes):
        st.info(
            "Limited complaint evidence detected. Generated a conservative action plan from fallback root-cause assumptions."
        )

        # In sparse-evidence mode, keep the plan focused by using only the
        # strongest fallback cause for the selected complaint topic.
        root_causes = sorted(
            root_causes,
            key=lambda rc: rc.get("confidence", 0),
            reverse=True,
        )[:1]

    if not root_causes:
        st.warning(
            f"Not enough review evidence to generate a plan for **{topic}**.  "
            "Try a category with more reviews."
        )
        return

    # ── Generate action plan ───────────────────────────────────────────
    from src.planning.action_generator import ActionPlanGenerator

    try:
        generator = ActionPlanGenerator()
        plan = generator.generate_plan(
            hotel_name, topic, root_causes, num_rooms=int(num_rooms)
        )
    except Exception as exc:
        st.error(f"Plan generation failed: {exc}")
        return

    # ── Cost summary banner ────────────────────────────────────────────
    total_cost = plan.get("total_cost", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("One-Time Investment",    f"£{total_cost.get('one_time', 0):,.0f}")
    c2.metric("Ongoing Monthly Cost",   f"£{total_cost.get('monthly', 0):,.0f}")
    c3.metric("3-Month Total",          f"£{total_cost.get('three_month_total', 0):,.0f}")
    st.markdown("---")

    # ── Actions by timeframe ───────────────────────────────────────────
    def _render_actions(actions: list, css_class: str, icon: str, label: str) -> None:
        st.subheader(f"{icon} {label}")
        if not actions:
            st.info(f"No {label.lower()} generated.")
            return
        for action in actions:
            conf  = action.get("confidence", 0)
            cause = action.get("root_cause", "").replace("_", " ").title()
            st.markdown(
                f'<div class="{css_class}">'
                f'<strong>{action["description"]}</strong><br>'
                f'<small>💰 Cost: {action["cost"]} &nbsp;|&nbsp; ⏱️ {action["timeline"]} '
                f'&nbsp;|&nbsp; 📌 {cause} ({conf:.0f}% confidence)</small><br>'
                f'<em>Expected impact: {action["expected_impact"]}</em>'
                f'</div>',
                unsafe_allow_html=True,
            )

    _render_actions(plan["immediate_actions"],  "action-immediate",  "🚨", "Immediate Actions (This Week)")
    st.markdown("")
    _render_actions(plan["short_term_actions"], "action-short-term", "⚡", "Short-Term Actions (2–6 Weeks)")
    st.markdown("")
    _render_actions(plan["long_term_actions"],  "action-long-term",  "🌱", "Long-Term Actions (1–6 Months)")

    st.markdown("---")

    # ── ROI prediction section ─────────────────────────────────────────
    st.subheader("💰 Predicted Financial ROI")

    roi_c1, roi_c2 = st.columns(2)
    with roi_c1:
        avg_rate = st.number_input(
            "Average Nightly Room Rate (£):", min_value=20, max_value=1000, value=150, step=10
        )
    with roi_c2:
        current_occ = st.slider(
            "Current Occupancy Rate (%):", min_value=20, max_value=95, value=65
        )

    if st.button("📈 Calculate ROI", type="primary"):
        try:
            from src.planning.roi_predictor import calculate_roi

            roi = calculate_roi(
                hotel_name,
                topic,
                plan,
                num_rooms=int(num_rooms),
                avg_room_rate=float(avg_rate),
                current_occupancy=current_occ / 100,
            )

            r1, r2, r3 = st.columns(3)
            r1.metric(
                "Rating Improvement",
                f"+{roi['rating_improvement']} ⭐",
                f"{roi['current_rating']} → {roi['expected_rating']}",
            )
            r2.metric(
                "Occupancy Uplift",
                roi["expected_occupancy"],
                f"from {roi['current_occupancy']}",
            )
            r3.metric("Monthly Revenue Increase", roi["monthly_revenue_increase"])

            r4, r5, r6 = st.columns(3)
            r4.metric("Total Investment",    roi["total_investment"])
            r5.metric("3-Month Net Profit",  roi["net_profit_3_months"])
            r6.metric(
                "ROI",
                roi["roi_percentage"],
                f"Payback: {roi['payback_months']} months",
            )

            from src.visualization.charts import roi_waterfall_chart
            st.plotly_chart(roi_waterfall_chart(roi), use_container_width=True)

        except Exception as exc:
            st.error(f"ROI calculation failed: {exc}")

    # ── Export ─────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("📥 Export Report to Excel"):
        try:
            from src.analysis.impact_calculator import build_impact_table
            from src.analysis.pattern_detector import analyze_hotel
            from src.visualization.report_generator import export_analysis_excel

            analysis    = analyze_hotel(hotel_name)
            hotel_df    = _load_hotel(hotel_name)
            impact_tbl  = build_impact_table(hotel_df)
            path = export_analysis_excel(hotel_name, analysis, impact_tbl, action_plan=plan)
            st.success(f"Report saved to: `{path}`")
        except Exception as exc:
            st.error(f"Export failed: {exc}")
