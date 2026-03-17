#!/usr/bin/env python
"""
scripts/download_data.py
=========================
Helper script to verify the dataset is present and optionally give
instructions for manually downloading the Kaggle dataset.

Usage:
    python scripts/download_data.py [--check-only]

Flags:
    --check-only   Just verify the file exists; do not print download guide.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXPECTED_PATH = ROOT / "data" / "raw" / "Hotel_Reviews.csv"
KAGGLE_URL    = "https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe"
MIN_ROWS      = 100_000


def check_dataset() -> bool:
    """Return True if the dataset file exists and appears valid."""
    if not EXPECTED_PATH.exists():
        print(f"[ERROR] Dataset not found at: {EXPECTED_PATH}")
        return False

    size_mb = EXPECTED_PATH.stat().st_size / (1024 * 1024)
    print(f"[OK] Dataset found: {EXPECTED_PATH}  ({size_mb:.1f} MB)")

    try:
        import pandas as pd
        df = pd.read_csv(EXPECTED_PATH, nrows=5)
        required_cols = {"Hotel_Name", "Negative_Review", "Positive_Review", "Reviewer_Score"}
        missing = required_cols - set(df.columns)
        if missing:
            print(f"[WARN] Missing expected columns: {missing}")
            return False
        print(f"[OK] Required columns present.")
        return True
    except Exception as exc:
        print(f"[ERROR] Failed to read CSV: {exc}")
        return False


def print_download_guide() -> None:
    print()
    print("=" * 60)
    print("  HOW TO DOWNLOAD THE DATASET")
    print("=" * 60)
    print()
    print("1. Install the Kaggle CLI:")
    print("      pip install kaggle")
    print()
    print("2. Set up your Kaggle API token:")
    print("      https://www.kaggle.com/docs/api#authentication")
    print()
    print("3. Download the dataset:")
    print("      kaggle datasets download -d jiashenliu/515k-hotel-reviews-data-in-europe")
    print(f"      Unzip to: {EXPECTED_PATH.parent}")
    print()
    print(f"   Or download manually from:")
    print(f"      {KAGGLE_URL}")
    print()
    print(f"4. Ensure the file is named:  Hotel_Reviews.csv")
    print(f"   and placed in:             {EXPECTED_PATH.parent}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify HotelInsight dataset")
    parser.add_argument("--check-only", action="store_true",
                        help="Only check; do not print download guide on failure")
    args = parser.parse_args()

    ok = check_dataset()
    if not ok and not args.check_only:
        print_download_guide()

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
