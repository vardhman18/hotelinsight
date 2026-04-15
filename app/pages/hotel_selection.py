"""
HotelInsight - Hotel Selection Page
======================================
Allows users to search for and select a hotel to analyse.
Stores the selection in ``st.session_state['selected_hotel']``.
"""

import time

import streamlit as st
import pandas as pd


@st.cache_data(show_spinner="Loading hotel list…")
def _cached_hotel_list():
    """Load hotel list once and cache for the session."""
    from src.data_processing.data_loader import get_hotel_list

    return get_hotel_list()


@st.cache_data(show_spinner="Fetching hotel stats…")
def _cached_hotel_stats(hotel_name: str):
    """Cache hotel stats to avoid repeated CSV scans."""
    from src.data_processing.data_loader import get_hotel_stats

    return get_hotel_stats(hotel_name)


def show() -> None:
    """Render the hotel selection page."""

    st.header("🏨 Select Hotel to Analyse")
    st.markdown(
        "Search for a hotel, review its summary statistics, then click **Analyse** to begin."
    )

    st.subheader("📤 Upload Your Own Review File")
    st.caption(
        "Accepted formats: CSV, XLSX, XLS. Use columns like hotel name, review text/positive/negative review, score/rating, and date."
    )

    prev_source_active = bool(st.session_state.get("uploaded_dataset_active", False))

    uploaded = st.file_uploader(
        "Upload hotel reviews",
        type=["csv", "xlsx", "xls"],
        help="When activated, all pages analyse your uploaded file instead of the default Kaggle dataset.",
    )

    up_col1, up_col2 = st.columns([2, 1])
    with up_col1:
        use_uploaded = st.checkbox(
            "Use uploaded file for analysis",
            value=bool(st.session_state.get("uploaded_dataset_active", False)),
            disabled=uploaded is None and "uploaded_dataset_df" not in st.session_state,
        )
    with up_col2:
        if st.button("Reset to Default Dataset"):
            for key in ["uploaded_dataset_df", "uploaded_dataset_name", "uploaded_dataset_active"]:
                st.session_state.pop(key, None)
            for key in ["selected_hotel", "selected_issue", "analysis_cache", "trends_cache", "impacts_cache"]:
                st.session_state.pop(key, None)
            st.cache_data.clear()
            st.success("Using default Kaggle dataset.")
            st.rerun()

    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df_uploaded = pd.read_csv(uploaded, low_memory=False)
            else:
                df_uploaded = pd.read_excel(uploaded)

            if df_uploaded.empty:
                st.error("Uploaded file is empty.")
            else:
                st.session_state["uploaded_dataset_df"] = df_uploaded
                st.session_state["uploaded_dataset_name"] = uploaded.name
                st.caption(
                    f"Loaded **{uploaded.name}** with {len(df_uploaded):,} rows and {len(df_uploaded.columns):,} columns."
                )
        except Exception as exc:
            st.error(f"Could not read uploaded file: {exc}")

    if use_uploaded and "uploaded_dataset_df" in st.session_state:
        st.session_state["uploaded_dataset_active"] = True
        st.info(f"Using uploaded dataset: **{st.session_state.get('uploaded_dataset_name', 'Uploaded dataset')}**")
    else:
        st.session_state["uploaded_dataset_active"] = False

    if prev_source_active != st.session_state.get("uploaded_dataset_active", False):
        for key in ["selected_hotel", "selected_issue", "analysis_cache", "trends_cache", "impacts_cache"]:
            st.session_state.pop(key, None)
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")

    # ------------------------------------------------------------------
    # Load hotel list (with graceful error handling for missing data)
    # ------------------------------------------------------------------
    try:
        hotels = _cached_hotel_list()
    except FileNotFoundError:
        st.error(
            "⚠️ Dataset file not found.  "
            "Please download `Hotel_Reviews.csv` from Kaggle and place it in `data/raw/`."
        )
        st.info(
            "See the [Setup Guide](docs/setup_guide.md) for download instructions."
        )
        return
    except Exception as exc:
        st.error(f"Failed to load hotel list: {exc}")
        return

    if not hotels:
        st.warning("No hotels found in the dataset.")
        return

    # ------------------------------------------------------------------
    # Search and select
    # ------------------------------------------------------------------
    search = st.text_input("🔍 Search hotels:", placeholder="e.g. Park Inn, London…")

    if search.strip():
        filtered = [h for h in hotels if search.lower() in h.lower()]
        if not filtered:
            st.warning("No hotels match your search. Try fewer keywords.")
            return
    else:
        filtered = hotels[:200]  # Show first 200 by default

    st.caption(f"Showing {len(filtered):,} of {len(hotels):,} hotels")

    selected_hotel = st.selectbox(
        "Choose a hotel:",
        filtered,
        help="Start typing in the search box above to narrow down the list.",
    )

    # ------------------------------------------------------------------
    # Hotel preview
    # ------------------------------------------------------------------
    if selected_hotel:
        st.markdown("---")
        st.subheader(f"📌 {selected_hotel}")

        try:
            stats = _cached_hotel_stats(selected_hotel)
        except Exception as exc:
            st.error(f"Could not load stats: {exc}")
            return

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Reviews", f"{stats['total_reviews']:,}")

        with col2:
            rating = stats["avg_rating"]
            delta_label = "⭐⭐⭐ Good" if rating >= 8 else "⭐⭐ Average" if rating >= 6 else "⭐ Poor"
            st.metric("Average Score (1–10)", f"{rating:.1f}", delta=delta_label)

        with col3:
            start, end = stats["date_range"]
            st.metric("Date Range", f"{start} → {end}")

        st.markdown("---")

        # ------------------------------------------------------------------
        # Analyse button
        # ------------------------------------------------------------------
        if st.button(
            "🔍 Analyse This Hotel",
            type="primary",
            help="Run full NLP analysis on all reviews for this hotel",
        ):
            st.session_state["selected_hotel"] = selected_hotel

            # Clear any cached analysis from a previous hotel
            for key in ["analysis_cache", "trends_cache", "impacts_cache"]:
                st.session_state.pop(key, None)

            progress = st.progress(0)
            status = st.empty()

            steps = [
                "Loading reviews…",
                "Cleaning text…",
                "Classifying topics…",
                "Detecting patterns…",
                "Computing sentiment…",
                "Ready!",
            ]

            for i, step in enumerate(steps):
                status.info(f"⚙️ {step}")
                progress.progress(int((i + 1) / len(steps) * 100))
                time.sleep(0.35)

            status.empty()
            progress.empty()
            st.success(
                f"✅ Analysis ready for **{selected_hotel}**!  "
                "Navigate to **Dashboard** in the sidebar."
            )
