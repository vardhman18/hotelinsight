#!/usr/bin/env python
"""
scripts/evaluate_models.py
============================
Evaluates the trained TopicClassifier and SentimentAnalyzer on held-out
data and prints a detailed classification report.

Usage:
    python scripts/evaluate_models.py [--sample SAMPLE] [--method METHOD]

Flags:
    --sample   Reviews to use for evaluation (default: 5000; 0 = all)
    --method   Sentiment method: vader or bert (default: vader)
    --save     Save evaluation report as JSON to data/results/
"""

import argparse
import json
import sys
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate HotelInsight models")
    parser.add_argument("--sample", type=int, default=5000)
    parser.add_argument("--method", choices=["vader", "bert"], default="vader")
    parser.add_argument("--save",   action="store_true")
    args = parser.parse_args()

    from src.utils.logger import get_logger
    log = get_logger("evaluate_models")

    print("Loading data…")
    from src.data_processing.data_loader import load_hotel_reviews
    from src.data_processing.data_cleaner import clean_reviews, split_train_test
    from src.data_processing.feature_extractor import add_topic_indicator_columns

    df = load_hotel_reviews()
    df = clean_reviews(df)

    if args.sample > 0 and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=99).reset_index(drop=True)

    df = add_topic_indicator_columns(df)
    _, test_df = split_train_test(df)
    print(f"Evaluating on {len(test_df):,} test reviews.")

    # ── Topic classification ──────────────────────────────────────────
    print("\n── Topic Classification ──")
    from src.analysis.topic_classifier import TopicClassifier
    from src.config.settings import COMPLAINT_CATEGORIES
    from src.utils.metrics import multilabel_f1

    clf = TopicClassifier()
    label_cols = [f"has_{c}" for c in COMPLAINT_CATEGORIES if f"has_{c}" in test_df.columns]
    y_true_arr = test_df[label_cols].values.astype(int)

    t0 = time.time()
    preds = clf.predict_batch(test_df["review_text"].tolist())

    import numpy as np
    y_pred_arr = np.zeros_like(y_true_arr)
    for i, pred_topics in enumerate(preds):
        for t in pred_topics:
            col_name = f"has_{t}"
            if col_name in label_cols:
                idx = label_cols.index(col_name)
                y_pred_arr[i, idx] = 1

    f1 = multilabel_f1(y_true_arr, y_pred_arr)
    elapsed = time.time() - t0
    print(f"  Macro-F1: {f1:.3f}  ({elapsed:.1f}s for {len(test_df)} reviews)")

    from sklearn.metrics import classification_report
    report_str = classification_report(
        y_true_arr, y_pred_arr, target_names=[c.title() for c in COMPLAINT_CATEGORIES if f"has_{c}" in label_cols],
        zero_division=0
    )
    print(report_str)

    # ── Sentiment evaluation ──────────────────────────────────────────
    print(f"\n── Sentiment Analysis ({args.method}) ──")
    from src.analysis.sentiment_analyzer import SentimentAnalyzer

    analyser   = SentimentAnalyzer(method=args.method)
    sample_reviews = test_df.head(500)

    t1 = time.time()
    scores = analyser.analyze_batch(sample_reviews["review_text"].tolist())
    elapsed2 = time.time() - t1

    labels = [analyser.get_sentiment_label(s) for s in scores]
    actual_labels = [
        "positive" if r >= 3.5 else ("negative" if r < 2.5 else "neutral")
        for r in sample_reviews["rating"]
    ]

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(actual_labels, labels)
    print(f"  Sentiment accuracy (vs star rating): {acc:.3f}  ({elapsed2:.1f}s for 500 reviews)")

    results = {
        "topic_macro_f1": round(f1, 4),
        "sentiment_accuracy": round(acc, 4),
        "method": args.method,
        "n_test": len(test_df),
    }

    if args.save:
        from src.config.settings import RESULTS_DIR
        out_path = Path(RESULTS_DIR) / "evaluation_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nReport saved to: {out_path}")

    log.info("Evaluation complete: %s", results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
