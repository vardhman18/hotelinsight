"""
HotelInsight - Sentiment Analyser
====================================
Provides a unified ``SentimentAnalyzer`` class that supports two backends:

- **BERT** (default): Uses the ``nlptown/bert-base-multilingual-uncased-sentiment``
  pipeline from Hugging Face Transformers.  Produces 1–5 star predictions
  which are mapped to the ``[-1, +1]`` scale.
- **VADER**: Lightweight rule-based analyser from ``vaderSentiment``.
  Uses the compound score directly (already on ``[-1, +1]``).

Both backends expose the same public interface so callers are agnostic to the
underlying method.
"""

import time
from functools import lru_cache
from typing import List, Optional

from src.config.settings import BATCH_SIZE, MAX_SEQUENCE_LENGTH, SENTIMENT_MODEL
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Mapping from BERT label strings to star integers
_BERT_LABEL_TO_STARS = {
    "1 star": 1,
    "2 stars": 2,
    "3 stars": 3,
    "4 stars": 4,
    "5 stars": 5,
}


def _stars_to_score(stars: int) -> float:
    """Convert a 1–5 star rating to a -1 to +1 sentiment score.

    Mapping:
    - 1 star  → -1.0
    - 2 stars → -0.5
    - 3 stars →  0.0
    - 4 stars → +0.5
    - 5 stars → +1.0

    Args:
        stars: Integer star rating (1–5).

    Returns:
        Float sentiment score in ``[-1.0, 1.0]``.
    """
    return (stars - 3) / 2.0


class SentimentAnalyzer:
    """Sentiment analysis using BERT or VADER.

    Attributes:
        method: ``"bert"`` or ``"vader"``.
    """

    def __init__(self, method: str = "bert") -> None:
        """Initialise the sentiment analyser.

        Args:
            method: ``"bert"`` (accurate, slower) or ``"vader"`` (fast,
                rule-based).  Defaults to ``"bert"``.

        Raises:
            ValueError: For unrecognised *method* values.
        """
        if method not in ("bert", "vader"):
            raise ValueError(f"method must be 'bert' or 'vader', got '{method}'.")

        self.method = method
        self._model = None  # Lazy-loaded on first use

        logger.info("SentimentAnalyzer initialised (method='%s').", method)

    # ------------------------------------------------------------------
    # Internal model access
    # ------------------------------------------------------------------

    def _get_model(self):
        """Return the loaded model, loading it on first access."""
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self) -> None:
        """Load the backing model into ``self._model``."""
        if self.method == "bert":
            self._load_bert()
        else:
            self._load_vader()

    def _load_bert(self) -> None:
        """Load the HuggingFace sentiment pipeline."""
        logger.info("Loading BERT sentiment model: %s …", SENTIMENT_MODEL)
        start = time.time()
        try:
            from transformers import pipeline

            self._model = pipeline(
                "sentiment-analysis",
                model=SENTIMENT_MODEL,
                tokenizer=SENTIMENT_MODEL,
                truncation=True,
                max_length=MAX_SEQUENCE_LENGTH,
            )
            logger.info(
                "BERT model loaded in %.1f s.", time.time() - start
            )
        except Exception as exc:
            logger.error("Failed to load BERT model: %s.  Falling back to VADER.", exc)
            self.method = "vader"
            self._load_vader()

    def _load_vader(self) -> None:
        """Load the VADER SentimentIntensityAnalyzer."""
        logger.info("Loading VADER sentiment analyser …")
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self._model = SentimentIntensityAnalyzer()
        logger.info("VADER analyser loaded.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> float:
        """Analyse sentiment of a single text.

        Args:
            text: Review text.  Long texts are truncated to
                ``MAX_SEQUENCE_LENGTH`` tokens before being sent to BERT.

        Returns:
            Sentiment score in ``[-1.0, +1.0]``.  Returns ``0.0`` for
            empty inputs.
        """
        if not text or not text.strip():
            return 0.0

        model = self._get_model()

        if self.method == "bert":
            return self._bert_score(text, model)
        return self._vader_score(text, model)

    def analyze_batch(self, texts: List[str]) -> List[float]:
        """Analyse sentiment for a list of texts.

        Processes inputs in batches of ``BATCH_SIZE`` for memory efficiency.
        An inline progress log is printed every 1 000 texts.

        Args:
            texts: List of review strings.

        Returns:
            List of sentiment scores aligned with *texts*.
        """
        if not texts:
            return []

        model = self._get_model()
        scores: List[float] = []
        total = len(texts)

        for start in range(0, total, BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]

            if self.method == "bert":
                batch_scores = [self._bert_score(t, model) for t in batch]
            else:
                batch_scores = [self._vader_score(t, model) for t in batch]

            scores.extend(batch_scores)

            if (start + BATCH_SIZE) % 1_000 < BATCH_SIZE:
                logger.info(
                    "Sentiment analysis: %d / %d processed.", min(start + BATCH_SIZE, total), total
                )

        logger.info("Batch sentiment analysis complete: %d texts.", total)
        return scores

    def get_sentiment_label(self, score: float) -> str:
        """Convert a numerical sentiment score to a human-readable label.

        Thresholds:
        - score > +0.3  → ``"positive"``
        - score < -0.3  → ``"negative"``
        - otherwise     → ``"neutral"``

        Args:
            score: Sentiment score in ``[-1.0, +1.0]``.

        Returns:
            ``"positive"``, ``"neutral"``, or ``"negative"``.
        """
        if score > 0.3:
            return "positive"
        if score < -0.3:
            return "negative"
        return "neutral"

    # ------------------------------------------------------------------
    # Private scoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bert_score(text: str, model) -> float:
        """Run BERT inference and convert the result to [-1, +1].

        The pipeline returns a label like ``"4 stars"`` and a confidence
        score; we use only the label to derive the sentiment score.

        Args:
            text: Input text (will be truncated by the pipeline).
            model: Loaded HuggingFace pipeline.

        Returns:
            Float score in ``[-1.0, +1.0]``.
        """
        try:
            result = model(text[:512])[0]  # extra runtime guard
            label_str = result["label"].lower()
            stars = _BERT_LABEL_TO_STARS.get(label_str, 3)
            return _stars_to_score(stars)
        except Exception as exc:
            logger.debug("BERT scoring failed for text snippet: %s", exc)
            return 0.0

    @staticmethod
    def _vader_score(text: str, model) -> float:
        """Return the VADER compound score for *text*.

        Args:
            text: Input text.
            model: Loaded ``SentimentIntensityAnalyzer``.

        Returns:
            Compound score in ``[-1.0, +1.0]``.
        """
        try:
            scores = model.polarity_scores(text)
            return float(scores["compound"])
        except Exception as exc:
            logger.debug("VADER scoring failed: %s", exc)
            return 0.0
