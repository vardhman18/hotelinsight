"""
HotelInsight - Evaluation Metrics
====================================
Wrappers around scikit-learn metric functions with project-specific defaults
and logging.  Used by the model-training and evaluation scripts, as well as
the ``app/pages/progress_tracker.py`` page.
"""

from typing import Dict, List, Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def classification_report_dict(
    y_true: List,
    y_pred: List,
    labels: Optional[List[str]] = None,
) -> Dict:
    """Generate a classification report as a dictionary.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Optional ordered list of label names for the report.

    Returns:
        Dictionary produced by ``sklearn.metrics.classification_report``
        with ``output_dict=True``.
    """
    from sklearn.metrics import classification_report  # lazy import

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    logger.info(
        "Classification report – accuracy: %.3f, macro F1: %.3f",
        report.get("accuracy", 0.0),
        report.get("macro avg", {}).get("f1-score", 0.0),
    )
    return report


def multilabel_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> float:
    """F1-score for multi-label classification problems.

    Args:
        y_true: Binary indicator matrix (n_samples × n_classes).
        y_pred: Predicted binary indicator matrix.
        average: Averaging strategy – ``"macro"``, ``"micro"``, or
            ``"weighted"``.

    Returns:
        F1-score as a float in ``[0, 1]``.
    """
    from sklearn.metrics import f1_score  # lazy import

    score = f1_score(y_true, y_pred, average=average, zero_division=0)
    logger.info("Multi-label F1 (%s): %.3f", average, score)
    return score


def accuracy_score(y_true: List, y_pred: List) -> float:
    """Compute simple accuracy.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Accuracy as a float in ``[0, 1]``.
    """
    from sklearn.metrics import accuracy_score as _acc  # lazy import

    score = _acc(y_true, y_pred)
    logger.info("Accuracy: %.3f", score)
    return score


def sentiment_metrics(
    y_true: List[str],
    y_pred: List[str],
) -> Dict[str, float]:
    """Compute precision, recall, and F1 for sentiment classification.

    Args:
        y_true: Ground-truth sentiment labels (``"positive"``,
            ``"neutral"``, ``"negative"``).
        y_pred: Predicted sentiment labels.

    Returns:
        Dictionary with keys ``accuracy``, ``precision``, ``recall``,
        ``f1`` (all macro-averaged).
    """
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score as _acc

    acc = _acc(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    result = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
    }
    logger.info("Sentiment metrics: %s", result)
    return result


def mean_absolute_error(y_true: List[float], y_pred: List[float]) -> float:
    """Compute Mean Absolute Error for regression tasks (e.g. rating prediction).

    Args:
        y_true: Ground-truth numeric values.
        y_pred: Predicted numeric values.

    Returns:
        MAE as a float.
    """
    from sklearn.metrics import mean_absolute_error as _mae

    score = _mae(y_true, y_pred)
    logger.info("MAE: %.4f", score)
    return float(score)
