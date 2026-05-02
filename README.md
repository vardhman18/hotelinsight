# HotelInsight

HotelInsight is an AI-powered hotel review analytics platform that converts large-scale guest feedback into sentiment insights, complaint classification, root-cause analysis, and prioritised action plans with estimated ROI.

## Overview

The application analyses the Kaggle 515K European Hotel Reviews dataset and presents the results in a Streamlit dashboard. It supports sentiment scoring, complaint topic detection, operational root-cause inference, action recommendations, progress tracking, and exportable reports.

## Key Features

- Sentiment analysis with BERT and VADER
- Multi-label topic classification across eight complaint categories
- Root-cause inference for operational issues
- Action plan generation with cost estimates
- ROI prediction for proposed improvements
- Progress tracking and Excel export

## Quick Start

```bash
git clone <repo-url> HotelInsight
cd HotelInsight
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/main.py
```

Place the dataset at data/raw/Hotel_Reviews.csv before launching the app.

## Project Structure

- app/ - Streamlit application, pages, and UI components
- data/ - raw data and generated results
- docs/ - setup, user, and API documentation
- notebooks/ - exploratory analysis and model evaluation notebooks
- scripts/ - data, training, evaluation, and export utilities
- src/ - analysis, processing, planning, utilities, and visualisation code
- tests/ - automated test suite

## Main Commands

```bash
python scripts/download_data.py
python scripts/train_models.py --sample 20000
python scripts/evaluate_models.py --method vader
python scripts/export_results.py --hotel "Hotel Arena"
pytest tests/ -v
```

## Documentation

- [Setup Guide](docs/setup_guide.md)
- [User Manual](docs/user_manual.md)
- [API Documentation](docs/api_documentation.md)

## License

MIT
