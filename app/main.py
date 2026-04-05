"""
HotelInsight - Streamlit Entry Point
======================================
Run with::

    streamlit run app/main.py

Navigation is sidebar-based; each page is a module in ``app/pages/`` that
exposes a ``show()`` function.
"""

import os
import sys

# Ensure the project root is on sys.path so ``src.*`` imports work when
# Streamlit is launched from any working directory.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st

from src.data_processing.data_loader import (
    clear_runtime_dataset,
    set_runtime_dataset,
)

# ---------------------------------------------------------------------------
# Page configuration  (must be the first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="HotelInsight – AI Hotel Analytics",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS injection
# ---------------------------------------------------------------------------

_CSS_PATH = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
if os.path.exists(_CSS_PATH):
    with open(_CSS_PATH, "r", encoding="utf-8") as _f:
        st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

# Inline base styles (minimal fallback if CSS file is missing)
st.markdown(
    """
    <style>
        .big-metric   { font-size: 48px; font-weight: bold; }
        .badge-critical { background-color:#ff4444;color:white;padding:4px 10px;
                          border-radius:12px;font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("🏨 HotelInsight")
st.sidebar.markdown("*AI-Powered Hotel Analytics*")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🏨 Hotel Selection",
        "📊 Dashboard",
        "🔍 Detailed Analysis",
        "📋 Action Plans",
        "📈 Progress Tracker",
    ],
)

st.sidebar.markdown("---")

# Show selected hotel in sidebar (if any)
if "selected_hotel" in st.session_state:
    st.sidebar.success(f"**Selected Hotel:**\n{st.session_state['selected_hotel']}")
    if st.sidebar.button("🔄 Change Hotel"):
        # Clear selection so user goes back to hotel selection
        for key in ["selected_hotel", "selected_issue", "analysis_cache"]:
            st.session_state.pop(key, None)
        st.rerun()

st.sidebar.markdown("---")

# Keep data-loader runtime state aligned with this Streamlit session.
if (
    st.session_state.get("uploaded_dataset_active", False)
    and "uploaded_dataset_df" in st.session_state
    and st.session_state["uploaded_dataset_df"] is not None
):
    set_runtime_dataset(
        st.session_state["uploaded_dataset_df"],
        dataset_name=st.session_state.get("uploaded_dataset_name", "Uploaded dataset"),
    )
else:
    clear_runtime_dataset()

# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------

# Lazy imports keep startup fast; each page module is loaded only when needed

if page == "🏠 Home":
    from app.pages import home
    home.show()

elif page == "🏨 Hotel Selection":
    from app.pages import hotel_selection
    hotel_selection.show()

elif page == "📊 Dashboard":
    from app.pages import dashboard
    dashboard.show()

elif page == "🔍 Detailed Analysis":
    from app.pages import detailed_analysis
    detailed_analysis.show()

elif page == "📋 Action Plans":
    from app.pages import action_plans
    action_plans.show()

elif page == "📈 Progress Tracker":
    from app.pages import progress_tracker
    progress_tracker.show()
