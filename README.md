# HotelInsight

> **AI-powered hotel review analytics** — turn 515 K guest reviews into prioritised action plans with predicted financial ROI.

---

## What it does

HotelInsight ingests the Kaggle 515 K European Hotel Reviews dataset and gives hotel managers a five-page interactive dashboard to:

| Capability | Description |
|------------|-------------|
| **Sentiment Analysis** | BERT (multilingual) + VADER scores every review −1 → +1 |
| **Topic Classification** | 8 complaint categories: cleanliness, staff, maintenance, noise, wifi, breakfast, value, location |
| **Root Cause Inference** | Maps complaint patterns to 7 operational root causes |
| **Action Plan Generator** | Immediate / short-term / long-term actions with £ cost estimates |
| **ROI Prediction** | Projects rating uplift, occupancy increase and 3-month net profit |
| **Progress Tracker** | Month-over-month trend analysis per complaint category |
| **Excel Export** | One-click multi-sheet report for any hotel |

---

## Quick Start

```bash
# 1. Clone and create venv
git clone <repo-url> HotelInsight && cd HotelInsight
python -m venv .venv && .venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate                           # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place dataset
#    data/raw/Hotel_Reviews.csv  (see docs/setup_guide.md for download instructions)

# 4. Launch app
streamlit run app/main.py
```

App opens at **http://localhost:8501**

---

## Repository structure

```
HotelInsight/
├── app/
│   ├── main.py                 Streamlit entry point
│   ├── assets/styles.css       Custom CSS
│   ├── pages/                  home, hotel_selection, dashboard,
│   │                           detailed_analysis, action_plans, progress_tracker
│   └── components/             metric_cards, charts, tables, filters
├── data/
│   ├── raw/                    Hotel_Reviews.csv (add manually)
│   └── results/                Generated Excel reports
├── docs/
│   ├── setup_guide.md
│   ├── user_manual.md
│   └── api_documentation.md
├── models/                     Saved ML artefacts
├── notebooks/                  01–04 Jupyter analysis notebooks
├── scripts/
│   ├── download_data.py        Dataset verification
│   ├── train_models.py         ML training
│   ├── evaluate_models.py      Model evaluation
│   └── export_results.py       Batch Excel export
├── src/
│   ├── analysis/               sentiment, topic, pattern, root cause, impact
│   ├── config/                 settings.py, action_templates.json
│   ├── data_processing/        loader, cleaner, feature extractor
│   ├── planning/               cost, priority, action generator, ROI predictor
│   ├── utils/                  logger, text_processing, date_utils, metrics
│   └── visualization/          charts, dashboards, report_generator
├── tests/                      pytest suite (5 test modules)
├── requirements.txt
└── README.md
```

---

## Dataset

| Field | Value |
|-------|-------|
| Source | [Kaggle — 515K Hotel Reviews](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe) |
| Reviews | 515,738 |
| Hotels | 1,493 |
| Date range | 2015–2017 |
| Rating scale | 1–10 (converted internally to 1–5) |

---

## Technology

| Component | Library |
|-----------|---------|
| Web app | Streamlit |
| Sentiment (primary) | `nlptown/bert-base-multilingual-uncased-sentiment` (HuggingFace) |
| Sentiment (fallback) | VADER (`nltk`) |
| Topic classification | TF-IDF + `OneVsRestClassifier(RandomForestClassifier)` |
| Data processing | pandas, scikit-learn |
| Visualisation | Plotly |
| Excel export | openpyxl |
| Testing | pytest |

---

## Scripts

```bash
# Verify dataset
python scripts/download_data.py

# Train ML model (20 000 sample reviews, ~5 min)
python scripts/train_models.py --sample 20000

# Evaluate accuracy
python scripts/evaluate_models.py --method vader

# Export Excel report for a hotel
python scripts/export_results.py --hotel "Hotel Arena"

# Export top-10 hotels
python scripts/export_results.py --top 10
```

---

## Tests

```bash
pytest tests/ -v
```

---

## Performance targets

| Metric | Target |
|--------|--------|
| Complaint category detection (F1) | ≥ 0.75 |
| Sentiment accuracy vs star rating | ≥ 0.70 |
| Dashboard first load | < 30 s |
| Per-hotel analysis | < 5 s |

---

## Docs

- [Setup Guide](docs/setup_guide.md)
- [User Manual](docs/user_manual.md)
- [API Documentation](docs/api_documentation.md)

---

## License

MIT
