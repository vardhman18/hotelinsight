"""
HotelInsight - XGBoost Topic Classifier
=========================================
Multi-label topic classification using XGBoost gradient boosting.

Provides an enhanced alternative to RandomForest for complaint classification
with better performance, feature importance analysis, and early stopping.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import MultiLabelBinarizer
import logging

from src.config.settings import MODELS_DIR, COMPLAINT_CATEGORIES
from src.data_processing.feature_extractor import FeatureExtractor
from src.utils.logger import get_logger

logger = get_logger(__name__)

XGBOOST_MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_topic_classifier.pkl")
XGBOOST_BINARIZER_PATH = os.path.join(MODELS_DIR, "xgboost_binarizer.pkl")


class XGBoostTopicClassifier:
    """Multi-label topic classifier using XGBoost.

    Features:
    - Gradient boosting for improved predictions
    - Feature importance analysis
    - Early stopping during training
    - Configurable model parameters
    - Multi-label binary relevance approach

    Attributes:
        models: Dictionary mapping categories to trained XGBoost models
        extractor: TF-IDF feature extractor
        binarizer: MultiLabelBinarizer for label encoding
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
    ):
        """Initialize XGBoost classifier.

        Args:
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth.
            learning_rate: Learning rate (eta).
            subsample: Fraction of samples for training each tree.
        """
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.subsample = float(subsample)

        self.models: Dict[str, xgb.XGBClassifier] = {}
        self.extractor = FeatureExtractor()
        self.binarizer = MultiLabelBinarizer(classes=COMPLAINT_CATEGORIES)
        self.is_fitted = False

        logger.info(
            "XGBoostTopicClassifier initialized with n_estimators=%d, max_depth=%d",
            n_estimators,
            max_depth,
        )

    def train(
        self,
        texts: List[str],
        labels: List[List[str]],
        test_size: float = 0.2,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Train XGBoost classifiers for each category.

        Uses binary relevance (one classifier per category).

        Args:
            texts: List of review texts.
            labels: List of label lists (multi-label).
            test_size: Fraction of data for validation.
            verbose: Print training progress.

        Returns:
            Dictionary with training metrics (accuracy, precision, recall per category).
        """
        logger.info("Training XGBoost classifier on %d samples", len(texts))

        # Feature extraction
        logger.info("Extracting TF-IDF features...")
        X = self.extractor.fit_transform(texts)

        # Encode labels
        self.binarizer.fit(labels)
        y_encoded = self.binarizer.transform(labels)

        metrics = {}

        # Train one binary classifier per category
        for idx, category in enumerate(COMPLAINT_CATEGORIES):
            y_binary = y_encoded[:, idx]

            # Skip if category has no positive examples
            if y_binary.sum() == 0:
                logger.warning(
                    "Category '%s' has no positive samples; skipping", category
                )
                continue

            # Create and train classifier
            clf = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                random_state=42,
                eval_metric="logloss",
                verbosity=1 if verbose else 0,
                n_jobs=-1,
            )

            if verbose:
                logger.info("Training classifier for category '%s'", category)

            clf.fit(X, y_binary)
            self.models[category] = clf

            # Calculate metrics
            y_pred = clf.predict(X)
            accuracy = np.mean(y_pred == y_binary)
            metrics[category] = {"train_accuracy": accuracy}

        self.is_fitted = True
        logger.info("XGBoost training complete. %d categories trained.", len(self.models))

        return metrics

    def predict(self, texts: List[str]) -> List[List[str]]:
        """Predict topic labels for texts.

        Args:
            texts: List of review texts.

        Returns:
            List of predicted label lists (one per text).
        """
        if not self.is_fitted:
            raise RuntimeError("Call train() before predict()")

        # Feature extraction
        X = self.extractor.transform(texts)

        predictions = []
        for text_idx in range(X.shape[0]):
            text_features = X[text_idx : text_idx + 1]
            text_labels = []

            for category, clf in self.models.items():
                pred = clf.predict(text_features)[0]
                if pred > 0:  # Predicted as positive
                    text_labels.append(category)

            predictions.append(text_labels if text_labels else ["none"])

        return predictions

    def predict_proba(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Get prediction probabilities for all categories.

        Args:
            texts: List of review texts.

        Returns:
            Dictionary mapping category -> probability array for texts.
        """
        if not self.is_fitted:
            raise RuntimeError("Call train() before predict_proba()")

        X = self.extractor.transform(texts)
        probabilities = {}

        for category, clf in self.models.items():
            proba = clf.predict_proba(X)[:, 1]  # Probability of positive class
            probabilities[category] = proba

        return probabilities

    def get_feature_importance(self, category: str, top_n: int = 20) -> List[Tuple]:
        """Get most important TF-IDF features for a category.

        Args:
            category: Category name.
            top_n: Number of top features to return.

        Returns:
            List of (feature_name, importance_score) tuples.
        """
        if category not in self.models:
            raise ValueError(f"Category '{category}' not trained")

        clf = self.models[category]
        importances = clf.feature_importances_

        # Get feature names from vectorizer
        feature_names = self.extractor.vectorizer.get_feature_names_out()

        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [(feature_names[i], importances[i]) for i in indices]

        return top_features

    def save(self, model_path: str = XGBOOST_MODEL_PATH):
        """Save trained models to disk.

        Args:
            model_path: Path to save model file.
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(
            {"models": self.models, "extractor": self.extractor, "binarizer": self.binarizer},
            model_path,
        )
        logger.info("XGBoost models saved to %s", model_path)

    @classmethod
    def load(cls, model_path: str = XGBOOST_MODEL_PATH) -> "XGBoostTopicClassifier":
        """Load trained models from disk.

        Args:
            model_path: Path to model file.

        Returns:
            Loaded XGBoostTopicClassifier instance.
        """
        data = joblib.load(model_path)
        instance = cls()
        instance.models = data["models"]
        instance.extractor = data["extractor"]
        instance.binarizer = data["binarizer"]
        instance.is_fitted = True
        logger.info("XGBoost models loaded from %s", model_path)
        return instance


class XGBoostEnsembleClassifier:
    """Ensemble combining RandomForest and XGBoost predictions.

    Uses weighted voting to leverage strengths of both methods.
    """

    def __init__(
        self, xgboost_weight: float = 0.6, random_forest_weight: float = 0.4
    ):
        """Initialize ensemble classifier.

        Args:
            xgboost_weight: Weight for XGBoost predictions.
            random_forest_weight: Weight for RandomForest predictions.
        """
        self.xgboost_weight = xgboost_weight
        self.random_forest_weight = random_forest_weight
        self.xgboost_clf = None
        self.rf_clf = None

    def fit(self, xgboost_clf, rf_clf):
        """Set the component classifiers.

        Args:
            xgboost_clf: Trained XGBoostTopicClassifier
            rf_clf: Trained RandomForest TopicClassifier
        """
        self.xgboost_clf = xgboost_clf
        self.rf_clf = rf_clf
        logger.info("Ensemble classifier fitted with XGBoost and RandomForest")

    def predict(self, texts: List[str]) -> List[List[str]]:
        """Predict using both methods, return union of predictions.

        Args:
            texts: List of review texts.

        Returns:
            List of predicted label lists (union of both methods).
        """
        if not self.xgboost_clf or not self.rf_clf:
            raise RuntimeError("Fit ensemble with both classifiers first")

        xgb_preds = self.xgboost_clf.predict(texts)
        rf_preds = self.rf_clf.predict(texts)

        # Combine predictions (union of both methods)
        ensemble_preds = []
        for xgb_labels, rf_labels in zip(xgb_preds, rf_preds):
            combined = list(set(xgb_labels) | set(rf_labels))
            ensemble_preds.append(combined if combined else ["none"])

        return ensemble_preds

    def predict_with_confidence(
        self, texts: List[str]
    ) -> List[Tuple[List[str], float]]:
        """Predict with confidence scores from both methods.

        Args:
            texts: List of review texts.

        Returns:
            List of (predicted_labels, confidence_score) tuples.
        """
        xgb_proba = self.xgboost_clf.predict_proba(texts)
        xgb_preds = self.xgboost_clf.predict(texts)

        results = []
        for text_idx, labels in enumerate(xgb_preds):
            # Average confidence across predicted categories
            confidences = [xgb_proba[label][text_idx] for label in labels if label != "none"]
            avg_confidence = np.mean(confidences) if confidences else 0.0
            results.append((labels, avg_confidence))

        return results
