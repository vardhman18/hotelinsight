"""
HotelInsight - Feature Extractor
===================================
Transforms cleaned review text into numerical feature matrices suitable for
scikit-learn classifiers.  The primary feature is a TF-IDF vector with
optional n-gram support.

Also provides convenience functions to add binary topic-indicator columns
to a DataFrame (used by the pattern detector and ROI predictor).
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from src.config.settings import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Path where the fitted vectoriser is persisted
import os
TFIDF_VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")


class FeatureExtractor:
    """Fit and transform review texts into TF-IDF feature matrices.

    Attributes:
        vectorizer: scikit-learn ``TfidfVectorizer`` instance.
        is_fitted: Whether the vectoriser has been fitted to training data.
    """

    def __init__(
        self,
        max_features: int = 10_000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
    ) -> None:
        """Initialise the feature extractor.

        Args:
            max_features: Maximum vocabulary size (most frequent *n* tokens).
            ngram_range: Minimum and maximum n-gram lengths.
            min_df: Ignore terms that appear in fewer than *min_df* documents.
            max_df: Ignore terms that appear in more than *max_df* fraction of
                documents (removes corpus-wide stopwords automatically).
            sublinear_tf: Apply ``log(1 + tf)`` scaling to term frequencies.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\b[a-zA-Z][a-zA-Z]+\b",  # words ≥ 2 letters
        )
        self.is_fitted: bool = False

    def fit(self, texts: List[str]) -> "FeatureExtractor":
        """Fit the TF-IDF vectoriser on *texts*.

        Args:
            texts: List of cleaned review strings.

        Returns:
            ``self`` (for method chaining).
        """
        logger.info("Fitting TF-IDF vectoriser on %d texts …", len(texts))
        self.vectorizer.fit(texts)
        self.is_fitted = True
        vocab_size = len(self.vectorizer.vocabulary_)
        logger.info("Vectoriser fitted.  Vocabulary size: %d", vocab_size)
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts into a TF-IDF sparse matrix.

        Args:
            texts: List of review strings.

        Returns:
            Sparse matrix of shape ``(n_samples, n_features)``.

        Raises:
            RuntimeError: If ``fit`` has not been called first.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before transform().")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit on *texts* then transform them in one step.

        Args:
            texts: List of review strings.

        Returns:
            Sparse matrix.
        """
        self.fit(texts)
        return self.transform(texts)

    def save(self, path: Optional[str] = None) -> None:
        """Persist the fitted vectoriser to disk.

        Args:
            path: File path for the pickle file.  Defaults to
                ``TFIDF_VECTORIZER_PATH`` from settings.
        """
        save_path = path or TFIDF_VECTORIZER_PATH
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(self.vectorizer, save_path)
        logger.info("TF-IDF vectoriser saved to: %s", save_path)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "FeatureExtractor":
        """Load a previously saved vectoriser.

        Args:
            path: File path of the pickled vectoriser.

        Returns:
            A new ``FeatureExtractor`` instance with the loaded vectoriser.
        """
        load_path = path or TFIDF_VECTORIZER_PATH
        instance = cls.__new__(cls)
        instance.vectorizer = joblib.load(load_path)
        instance.is_fitted = True
        logger.info("TF-IDF vectoriser loaded from: %s", load_path)
        return instance


# ---------------------------------------------------------------------------
# DataFrame-level helpers
# ---------------------------------------------------------------------------

def add_topic_indicator_columns(
    df: pd.DataFrame,
    topic_col: str = "topics",
    categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Add binary ``has_<topic>`` columns to *df* from a topics list column.

    The :mod:`topic_classifier` stores predicted topics as a Python list
    in each row.  This function explodes that into individual boolean flags,
    which are required by the pattern detector and ROI predictor.

    Args:
        df: DataFrame containing a column of topic lists.
        topic_col: Name of the column holding ``list[str]`` of topics.
        categories: Topic names to create columns for.  Defaults to
            :data:`~src.config.settings.COMPLAINT_CATEGORIES`.

    Returns:
        DataFrame with new boolean columns ``has_cleanliness``,
        ``has_staff``, … etc.
    """
    from src.config.settings import COMPLAINT_CATEGORIES

    cats = categories or COMPLAINT_CATEGORIES

    if topic_col not in df.columns:
        logger.warning(
            "Column '%s' not found; attempting keyword topic extraction from 'review_text'.",
            topic_col,
        )
        if "review_text" in df.columns:
            from src.analysis.topic_classifier import TopicClassifier

            classifier = TopicClassifier()
            df = df.copy()
            df[topic_col] = df["review_text"].fillna("").astype(str).apply(
                classifier.extract_topics_keyword
            )
        else:
            for cat in cats:
                df[f"has_{cat}"] = False
            return df

    for cat in cats:
        df[f"has_{cat}"] = df[topic_col].apply(
            lambda topics: cat in topics if isinstance(topics, list) else False
        )

    return df
