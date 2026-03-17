"""
HotelInsight - Home Page
==========================
Landing page that introduces the system and guides users to begin.
"""

import streamlit as st


def show() -> None:
    """Render the HotelInsight home / landing page."""

    st.title("🏨 HotelInsight")
    st.subheader("AI-Powered Hotel Operations Analytics")
    st.markdown("---")

    # Hero description
    st.markdown(
        """
        **HotelInsight** turns thousands of guest reviews into actionable intelligence.  
        Upload your hotel's review data and get:

        | Feature | What you get |
        |---|---|
        | 🧠 **Sentiment Analysis** | Understand how guests *feel* about their stay |
        | 🏷️ **Topic Classification** | Automatically identify complaint categories |
        | 🔍 **Root Cause Detection** | Discover *why* problems keep occurring |
        | 📋 **Action Plans** | Concrete steps with timeline and cost estimates |
        | 💰 **ROI Prediction** | See the financial return before you invest |
        | 📈 **Trend Tracking** | Monitor whether issues are improving over time |
        """
    )

    st.markdown("---")

    # Quick-start guide
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 🚀 Getting Started")
        st.markdown(
            """
            1. **Place your data** in `data/raw/Hotel_Reviews.csv`
            2. **Select a hotel** using the Hotel Selection page
            3. **View the Dashboard** to see the top issues ranked by priority
            4. **Deep-dive** into any issue with the Detailed Analysis page
            5. **Generate action plans** with costed recommendations
            6. **Track progress** over time on the Progress Tracker
            """
        )

    with col2:
        st.markdown("### 📊 Performance Targets")
        st.markdown(
            """
            | Model | Metric | Target |
            |---|---|---|
            | Sentiment Analysis | Accuracy | ≥ 87 % |
            | Topic Classifier | F1-score | ≥ 78 % |
            | Root Cause Inference | Accuracy | ≥ 72 % |
            | Dashboard Load | Speed | < 5 s |
            | Full Analysis | Speed | < 20 s |
            """
        )

    st.markdown("---")

    # Call to action
    st.info(
        "👉 Start by going to **Hotel Selection** in the sidebar to choose a hotel to analyse."
    )

    # Dataset info
    with st.expander("📂 About the Dataset"):
        st.markdown(
            """
            **Primary Dataset**: [515K Hotel Reviews – Kaggle](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe)

            - **515,738** guest reviews
            - **1,493** European hotels
            - Reviews span **Aug 2015 – Aug 2017**
            - Columns include: hotel name, review date, reviewer nationality, positive/negative text, score (1–10)

            The dataset is **not included** in this repository (file size ~230 MB).  
            Download it from Kaggle and place it at `data/raw/Hotel_Reviews.csv`.
            """
        )

    with st.expander("🛠️ Tech Stack"):
        st.markdown(
            """
            | Layer | Technology |
            |---|---|
            | NLP / Sentiment | BERT (`nlptown`), VADER |
            | Topic Classification | TF-IDF + RandomForest |
            | Data Processing | pandas, numpy |
            | Visualisation | Plotly, Matplotlib |
            | Web Interface | Streamlit |
            | Storage | CSV files, joblib |
            """
        )
