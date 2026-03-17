# HotelInsight — Setup Guide

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python      | 3.8 +   |
| pip         | 23 +    |
| RAM         | 8 GB minimum (16 GB recommended for BERT) |
| Disk space  | ~2 GB (dataset + model weights) |

---

## 1. Clone the repository

```bash
git clone <your-repo-url> HotelInsight
cd HotelInsight
```

---

## 2. Create a virtual environment

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run will download the `nlptown/bert-base-multilingual-uncased-sentiment`
> model (~670 MB). An internet connection is required. Subsequent runs use the local cache.

---

## 4. Obtain the dataset

HotelInsight uses the **515K Hotel Reviews** dataset from Kaggle.

### Option A — Kaggle CLI (recommended)

```bash
pip install kaggle
# Set up ~/.kaggle/kaggle.json with your API token
# https://www.kaggle.com/docs/api#authentication

kaggle datasets download -d jiashenliu/515k-hotel-reviews-data-in-europe
# Unzip and place Hotel_Reviews.csv in:  data/raw/Hotel_Reviews.csv
```

### Option B — Manual download

1. Visit: <https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe>
2. Download and extract.
3. Place `Hotel_Reviews.csv` in `data/raw/`.

### Verify the dataset

```bash
python scripts/download_data.py
```

---

## 5. (Optional) Pre-train the ML models

The app works out-of-the-box using keyword classification.
To enable the RandomForest model for improved accuracy:

```bash
# ~5 minutes on a modern CPU with 20 000 sample reviews
python scripts/train_models.py --sample 20000

# Train on the full dataset (slow, ~30 minutes):
python scripts/train_models.py --sample 0
```

---

## 6. Launch the Streamlit application

```bash
streamlit run app/main.py
```

Your browser will open at `http://localhost:8501`.

---

## 7. Directory layout after setup

```
HotelInsight/
├── app/                  Streamlit application
│   ├── pages/            Individual pages
│   └── components/       Reusable UI components
├── data/
│   ├── raw/              Hotel_Reviews.csv goes here
│   └── results/          Generated Excel reports
├── models/               Saved ML model artefacts
├── notebooks/            Jupyter analysis notebooks
├── scripts/              CLI utilities
├── src/                  Core Python library
│   ├── analysis/
│   ├── config/
│   ├── data_processing/
│   ├── planning/
│   ├── utils/
│   └── visualization/
└── tests/                pytest test suite
```

---

## 8. Running the test suite

```bash
pytest tests/ -v
```

To include BERT tests (slow):

```bash
set HOTELINSIGHT_BERT_TESTS=1   # Windows
pytest tests/test_sentiment_analysis.py -v
```

---

## 9. Environment variables

| Variable                   | Default | Description |
|---------------------------|---------|-------------|
| `HOTELINSIGHT_BERT_TESTS` | `0`     | Set to `1` to run BERT model tests |
| `HOTELINSIGHT_LOG_DIR`    | `logs/` | Override log file directory |

---

## 10. Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: streamlit` | Activate venv and run `pip install -r requirements.txt` |
| BERT model download fails | Check internet connection; model is cached in `~/.cache/huggingface` |
| `FileNotFoundError: Hotel_Reviews.csv` | Follow Section 4 above |
| Slow dashboard on first visit | BERT model loads on first request; subsequent requests are faster |
| `OOMError` / memory crash | Use `--method vader` or reduce `--sample` in train scripts |
