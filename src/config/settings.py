"""
HotelInsight - Global Configuration
====================================
Central constants and configuration values used throughout the application.
All environment-specific values should be adjusted here rather than scattered
across individual modules.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Directory Paths
# ---------------------------------------------------------------------------

# Root of the project (two levels up from this file: src/config/ -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Top-level data directory
DATA_DIR = str(PROJECT_ROOT / "data")

# Raw (unprocessed) datasets; gitignored for size reasons
RAW_DATA_DIR = str(PROJECT_ROOT / "data" / "raw")

# Cleaned and processed datasets ready for model training/inference
PROCESSED_DATA_DIR = str(PROJECT_ROOT / "data" / "processed")

# Persisted ML model artefacts (.pkl, .json); gitignored
MODELS_DIR = str(PROJECT_ROOT / "models")

# Analysis output files (CSV exports, JSON summaries, charts)
RESULTS_DIR = str(PROJECT_ROOT / "data" / "results")

# ---------------------------------------------------------------------------
# Dataset File Names
# ---------------------------------------------------------------------------

# Primary Kaggle dataset: "515K Hotel Reviews Data in Europe"
MAIN_DATASET_FILENAME = "Hotel_Reviews.csv"

# Full path helpers
MAIN_DATASET_PATH = os.path.join(RAW_DATA_DIR, MAIN_DATASET_FILENAME)

# ---------------------------------------------------------------------------
# Sentiment Analysis - BERT Model
# ---------------------------------------------------------------------------

# HuggingFace model identifier for multilingual 1-5 star sentiment classification.
# Produces star ratings (1–5) which are converted to a -1 to +1 scale internally.
SENTIMENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"

# Maximum number of sub-word tokens fed to BERT.
# Texts longer than this are truncated; 512 is the BERT architectural limit.
MAX_SEQUENCE_LENGTH = 512

# Number of texts processed in a single forward pass during batch inference.
# Smaller values use less RAM; larger values are faster on GPU/multi-core CPU.
BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Topic / Complaint Categories
# ---------------------------------------------------------------------------

# Eight canonical complaint categories used for multi-label classification
# and all downstream analysis (root-cause, action plans, ROI).
COMPLAINT_CATEGORIES = [
    "cleanliness",   # Dirty rooms, stains, odours, mould
    "staff",         # Rude, unhelpful, slow service
    "maintenance",   # Broken appliances, HVAC, plumbing
    "noise",         # Thin walls, street noise, loud neighbours
    "wifi",          # Slow, absent or unreliable internet
    "breakfast",     # Poor quality, limited choice, service issues
    "value",         # Price-to-quality perception
    "location",      # Accessibility, distance from attractions
]

# ---------------------------------------------------------------------------
# Root Cause Inference Thresholds
# ---------------------------------------------------------------------------

# Minimum confidence (0–100) required to surface a root cause to the user.
# Causes inferred with less than this level are silently discarded.
ROOT_CAUSE_CONFIDENCE_THRESHOLD = 0.60  # 60 %

# ---------------------------------------------------------------------------
# Priority Scoring Thresholds
# ---------------------------------------------------------------------------
# Issues with a priority score above these thresholds receive the corresponding
# severity label used across the dashboard and action-plan modules.

# Score ≥ 60 → CRITICAL: Requires immediate management attention
PRIORITY_CRITICAL_THRESHOLD = 60

# Score ≥ 40 → HIGH: Address within the current planning cycle
PRIORITY_HIGH_THRESHOLD = 40

# Score ≥ 20 → MEDIUM: Schedule for upcoming quarter
PRIORITY_MEDIUM_THRESHOLD = 20

# Score <  20 → LOW: Monitor; no urgent action required

# ---------------------------------------------------------------------------
# Cost Estimation Constants (GBP)
# ---------------------------------------------------------------------------

# Estimated monthly labour cost per room for an additional housekeeping/service
# staff member.  Used as a baseline in cost formula evaluations.
STAFF_COST_PER_ROOM = 30  # £ per room per month

# Base cost for a structured staff-training programme (materials + facilitator).
# Often a one-time or annual expense.
TRAINING_COST_BASE = 1_200  # £

# Typical per-room cost for replacing a single category of in-room equipment
# (e.g., mattresses, HVAC units, kettles).
EQUIPMENT_COST_PER_ROOM = 100  # £ per room

# ---------------------------------------------------------------------------
# ROI / Revenue Projection Constants
# ---------------------------------------------------------------------------

# Industry benchmark: a +0.1 star improvement in average review rating leads to
# approximately +5 % occupancy uplift.
# Source: Cornell Hospitality Research (adapted for European mid-scale hotels).
RATING_TO_OCCUPANCY_FACTOR = 0.05  # +0.1 star → +5 % occupancy

# Realistic complaint-reduction rate achievable through sustained operational
# improvements (not a 100 % elimination — complaints never reach zero).
COMPLAINT_REDUCTION_RATE = 0.70  # 70 % reduction is achievable

# Maximum realistic occupancy cap used in revenue projections.
# Even highly rated hotels rarely sustain >95 % long term due to maintenance
# downtime, group-booking patterns, and seasonal gaps.
MAX_OCCUPANCY_CAP = 0.95

# ---------------------------------------------------------------------------
# Model Persistence File Names
# ---------------------------------------------------------------------------

SENTIMENT_MODEL_FILE = os.path.join(MODELS_DIR, "sentiment_model.pkl")
TOPIC_CLASSIFIER_FILE = os.path.join(MODELS_DIR, "topic_classifier.pkl")
MODEL_METRICS_FILE = os.path.join(MODELS_DIR, "model_metrics.json")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR = str(PROJECT_ROOT / "logs")
LOG_FILE = os.path.join(LOG_DIR, "hotelinsight.log")
LOG_LEVEL = "INFO"
