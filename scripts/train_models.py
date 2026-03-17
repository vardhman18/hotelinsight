#!/usr/bin/env python
"""
scripts/train_models.py
========================
Trains the TopicClassifier ML model on a random sample of the dataset,
then saves the model artefacts to models/.

Usage:
    python scripts/train_models.py [--sample SAMPLE] [--max-iter MAX_ITER]

Flags:
    --sample     Number of reviews to sample for training (default: 20000).
                 Use 0 to train on the full dataset (slow).
    --max-iter   Maximum number of estimators for RandomForest (default: 200).
    --force      Re-train even if a saved model already exists.
"""

import argparse
import sys
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train HotelInsight ML models")
    parser.add_argument("--sample",   type=int, default=20_000,
                        help="Number of reviews to sample (0 = full dataset)")
    parser.add_argument("--max-iter", type=int, default=200,
                        help="RandomForest n_estimators")
    parser.add_argument("--force",    action="store_true",
                        help="Re-train even if saved model exists")
    args = parser.parse_args()

    from src.utils.logger import get_logger
    log = get_logger("train_models")

    from src.config.settings import MODELS_DIR
    model_path = Path(MODELS_DIR) / "topic_classifier.pkl"

    if model_path.exists() and not args.force:
        print(f"[SKIP] Model already exists at: {model_path}")
        print("       Use --force to re-train.")
        return 0

    print("Loading and cleaning dataset…")
    t0 = time.time()

    from src.data_processing.data_loader import load_hotel_reviews
    from src.data_processing.data_cleaner import clean_reviews, split_train_test
    from src.data_processing.feature_extractor import FeatureExtractor, add_topic_indicator_columns
    from src.analysis.topic_classifier import TopicClassifier

    df = load_hotel_reviews()
    df = clean_reviews(df)

    if args.sample > 0 and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)
        print(f"Sampled {args.sample:,} reviews for training.")

    print("Adding topic indicator columns (keyword pass)…")
    df = add_topic_indicator_columns(df)

    from src.config.settings import COMPLAINT_CATEGORIES
    label_cols = [f"has_{c}" for c in COMPLAINT_CATEGORIES]
    available  = [c for c in label_cols if c in df.columns]

    if not available:
        print("[ERROR] No topic columns found after keyword labelling.")
        return 1

    train_df, test_df = split_train_test(df)
    X_train = train_df["review_text"].fillna("").tolist()
    X_test  = test_df["review_text"].fillna("").tolist()
    y_test  = test_df[available].values.astype(int)

    # Convert binary indicator columns to label lists expected by TopicClassifier.train.
    y_train_labels = []
    for _, row in train_df.iterrows():
        labels = [col.replace("has_", "") for col in available if int(row[col]) == 1]
        y_train_labels.append(labels)

    print(f"Training on {len(X_train):,} reviews … (n_estimators={args.max_iter})")
    clf = TopicClassifier(n_estimators=args.max_iter)
    clf.train(X_train, y_train_labels)

    from src.utils.metrics import multilabel_f1
    X_te_vec = clf.extractor.transform(X_test)
    y_pred = clf.classifier.predict(X_te_vec)
    f1 = multilabel_f1(y_test, y_pred)
    print(f"Validation macro-F1: {f1:.3f}")

    clf.save()
    fe_path = Path(MODELS_DIR) / "feature_extractor.pkl"
    import joblib
    joblib.dump(clf.extractor, fe_path)

    elapsed = time.time() - t0
    print(f"Training complete in {elapsed:.1f}s.")
    print(f"Model saved to: {model_path}")
    log.info("Training complete. Macro-F1=%.3f  elapsed=%.1fs", f1, elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
