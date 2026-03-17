"""
HotelInsight - Topic Classifier
==================================
Multi-label topic classification for hotel complaint categories.

Two complementary approaches are combined:

1. **Keyword matching** – deterministic, works without any training data.
2. **RandomForest classifier** – trained on TF-IDF features when labelled
   data is available.

The final ``predict`` method returns the *union* of both result sets so
that neither precision nor recall is sacrificed.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from src.config.settings import COMPLAINT_CATEGORIES, MODELS_DIR
from src.data_processing.feature_extractor import FeatureExtractor
from src.utils.logger import get_logger

logger = get_logger(__name__)

CLASSIFIER_PATH = os.path.join(MODELS_DIR, "topic_classifier.pkl")
BINARIZER_PATH = os.path.join(MODELS_DIR, "topic_binarizer.pkl")


class TopicClassifier:
    """Multi-label topic classifier for hotel complaint detection.

    Attributes:
        keywords: Dictionary mapping each category to a list of trigger words.
        classifier: Trained ``OneVsRestClassifier`` or ``None`` if untrained.
        extractor: :class:`~src.data_processing.feature_extractor.FeatureExtractor`
            used to build TF-IDF features.
        binarizer: ``MultiLabelBinarizer`` fitted on the category list.
    """

    def __init__(self, n_estimators: int = 200) -> None:
        """Initialise classifier with keyword dictionaries."""
        self.n_estimators = int(n_estimators)
        self.keywords: Dict[str, List[str]] = {
            "cleanliness": [
                "dirty", "filthy", "stain", "stained", "smell", "smelly", "unclean",
                "hair", "dust", "dusty", "mold", "mould", "garbage", "grime", "grimy",
                "cockroach", "insects", "bugs", "bathroom dirty", "toilet dirty",
                "not cleaned", "poorly cleaned", "disgusting",
            ],
            "staff": [
                "rude", "unhelpful", "unfriendly", "staff", "receptionist",
                "service", "attitude", "ignored", "waited", "slow", "arrogant",
                "dismissive", "impolite", "indifferent", "unprofessional",
                "front desk", "check in", "check out", "reception",
            ],
            "maintenance": [
                "broken", "not working", "ac", "air conditioning", "heating",
                "tv", "television", "elevator", "lift", "plumbing", "leak",
                "leaking", "shower", "toilet flush", "light bulb", "bulb",
                "socket", "plug", "outlet", "repair", "maintenance",
                "didn't work", "stopped working",
            ],
            "noise": [
                "noisy", "noise", "loud", "thin walls", "heard everything",
                "next door", "street noise", "traffic noise", "bar noise",
                "music", "party", "neighbours", "neighbors", "kept awake",
                "couldn't sleep", "could not sleep", "disruptive",
            ],
            "wifi": [
                "wifi", "wi-fi", "internet", "connection", "slow internet",
                "no wifi", "no internet", "weak signal", "disconnected",
                "offline", "bandwidth", "broadband",
            ],
            "breakfast": [
                "breakfast", "brunch", "buffet", "food", "meal", "eggs",
                "toast", "cereal", "juice", "coffee", "tea", "tasteless",
                "cold food", "poor food", "limited choice", "expensive breakfast",
                "not included", "dining",
            ],
            "value": [
                "overpriced", "expensive", "not worth", "poor value",
                "too much", "price", "cost", "cheap", "ripoff", "rip off",
                "money", "rate", "fee", "charge", "extortionate",
            ],
            "location": [
                "location", "far from", "distance", "remote", "inconvenient",
                "transport", "bus stop", "tube", "metro", "subway", "underground",
                "taxi", "walk", "centre", "center", "attractions",
            ],
        }

        self.classifier: Optional[OneVsRestClassifier] = None
        self.extractor: FeatureExtractor = FeatureExtractor()
        self.binarizer: MultiLabelBinarizer = MultiLabelBinarizer(
            classes=COMPLAINT_CATEGORIES
        )
        self.binarizer.fit([COMPLAINT_CATEGORIES])  # pre-fit with known classes

        logger.info(
            "TopicClassifier initialised with %d categories (n_estimators=%d).",
            len(self.keywords),
            self.n_estimators,
        )

    # ------------------------------------------------------------------
    # Keyword-based prediction
    # ------------------------------------------------------------------

    def extract_topics_keyword(self, text: str) -> List[str]:
        """Detect topics in *text* using keyword matching.

        Case-insensitive substring matching is used; a category is flagged
        if *any* of its keywords appear in the text.

        Args:
            text: Review text (should be lowercased).

        Returns:
            List of detected category names.
        """
        if not text:
            return []

        text_lower = text.lower()
        detected = []
        for category, words in self.keywords.items():
            if any(kw in text_lower for kw in words):
                detected.append(category)
        return detected

    # ------------------------------------------------------------------
    # ML-based training and prediction
    # ------------------------------------------------------------------

    def train(
        self, X_train: List[str], y_train: List[List[str]]
    ) -> None:
        """Train a multi-label RandomForest classifier.

        Args:
            X_train: List of cleaned review strings.
            y_train: List of lists of topic labels for each review.
        """
        logger.info("Training TopicClassifier on %d examples …", len(X_train))

        # Build TF-IDF features
        X_features = self.extractor.fit_transform(X_train)

        # Binarise labels
        Y = self.binarizer.transform(y_train)

        # Train one-vs-rest RandomForest
        base_rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=None,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        self.classifier = OneVsRestClassifier(base_rf)
        self.classifier.fit(X_features, Y)

        logger.info("TopicClassifier training complete.")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, text: str) -> List[str]:
        """Predict complaint topics for a single review.

        Returns the *union* of keyword-matching and ML-classifier results
        when the classifier has been trained.  Falls back to keyword-only
        if no classifier is available.

        Args:
            text: Review text.

        Returns:
            Deduplicated list of predicted topic labels.
        """
        keyword_topics = set(self.extract_topics_keyword(text))

        if self.classifier is not None and self.extractor.is_fitted:
            try:
                X = self.extractor.transform([text])
                Y_pred = self.classifier.predict(X)
                ml_topics = set(self.binarizer.inverse_transform(Y_pred)[0])
                return list(keyword_topics | ml_topics)
            except Exception as exc:
                logger.debug("ML classifier prediction failed: %s", exc)

        return list(keyword_topics)

    def predict_batch(self, texts: List[str]) -> List[List[str]]:
        """Predict complaint topics for a list of reviews.

        Args:
            texts: List of review strings.

        Returns:
            List of topic-label lists, one per input text.
        """
        if not texts:
            return []

        # Always compute keyword results
        keyword_results = [set(self.extract_topics_keyword(t)) for t in texts]

        if self.classifier is not None and self.extractor.is_fitted:
            try:
                X = self.extractor.transform(texts)
                Y_pred = self.classifier.predict(X)
                ml_all = [
                    set(labels)
                    for labels in self.binarizer.inverse_transform(Y_pred)
                ]
                return [list(kw | ml) for kw, ml in zip(keyword_results, ml_all)]
            except Exception as exc:
                logger.warning(
                    "Batch ML prediction failed (%s); using keyword-only.", exc
                )

        return [list(s) for s in keyword_results]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist the trained classifier and binarizer to disk."""
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(
            {"classifier": self.classifier, "binarizer": self.binarizer},
            CLASSIFIER_PATH,
        )
        self.extractor.save()
        logger.info("TopicClassifier saved to: %s", CLASSIFIER_PATH)

    @classmethod
    def load(cls) -> "TopicClassifier":
        """Load a previously trained TopicClassifier from disk.

        Returns:
            ``TopicClassifier`` instance with trained models loaded.
        """
        instance = cls()
        data = joblib.load(CLASSIFIER_PATH)
        instance.classifier = data["classifier"]
        instance.binarizer = data["binarizer"]
        instance.extractor = FeatureExtractor.load()
        logger.info("TopicClassifier loaded from: %s", CLASSIFIER_PATH)
        return instance
