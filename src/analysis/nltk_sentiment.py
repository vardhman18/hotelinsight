"""
HotelInsight - Enhanced Sentiment Analysis with NLTK
======================================================
Extends sentiment analysis with NLTK capabilities:
- TextBlob sentiment scoring (alternative to VADER/BERT)
- Tokenization and word analysis
- Subjectivity scoring
- Multi-method ensemble analysis
"""

import nltk
from nltk.sentiment import TextBlob
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from typing import Dict, List, Tuple
import logging

from src.utils.logger import get_logger

logger = get_logger(__name__)


class NLTKSentimentAnalyzer:
    """NLTK-based sentiment analysis with TextBlob and word analysis.

    Features:
    - TextBlob sentiment (polarity & subjectivity)
    - Sentence-level sentiment
    - Subjectivity analysis
    - Stop word filtering
    - Word frequency analysis
    """

    def __init__(self):
        """Initialize NLTK resources."""
        self._download_nltk_resources()
        self.stop_words = set(stopwords.words("english"))

    @staticmethod
    def _download_nltk_resources():
        """Download required NLTK data (punkt, stopwords, etc)."""
        required_resources = [
            ("tokenizers/punkt", "punkt"),
            ("corpora/stopwords", "stopwords"),
        ]

        for resource_path, resource_name in required_resources:
            try:
                nltk.data.find(resource_path)
                logger.debug("NLTK resource '%s' already present.", resource_name)
            except LookupError:
                logger.info("Downloading NLTK resource '%s'...", resource_name)
                nltk.download(resource_name, quiet=True)
                logger.info("Downloaded NLTK resource '%s'.", resource_name)

    def analyze_textblob(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using TextBlob.

        Returns polarity (sentiment) and subjectivity scores.

        Args:
            text: Review text.

        Returns:
            Dictionary with:
            - polarity: [-1, 1] sentiment score
            - subjectivity: [0, 1] subjectivity score (0=objective, 1=subjective)
        """
        if not text or not text.strip():
            return {"polarity": 0.0, "subjectivity": 0.0}

        try:
            blob = TextBlob(text)
            return {
                "polarity": float(blob.sentiment.polarity),
                "subjectivity": float(blob.sentiment.subjectivity),
            }
        except Exception as exc:
            logger.debug("TextBlob analysis failed: %s", exc)
            return {"polarity": 0.0, "subjectivity": 0.0}

    def analyze_sentence_level(self, text: str) -> Dict:
        """Analyze sentiment at sentence level.

        Args:
            text: Review text (can be multi-sentence).

        Returns:
            Dictionary with:
            - sentence_count: Number of sentences
            - polarities: List of polarity scores per sentence
            - avg_polarity: Average polarity
            - neg_sentences: Count of negative sentences
            - pos_sentences: Count of positive sentences
        """
        try:
            sentences = sent_tokenize(text)
            polarities = []

            for sent in sentences:
                blob = TextBlob(sent)
                polarities.append(blob.sentiment.polarity)

            neg_count = sum(1 for p in polarities if p < -0.1)
            pos_count = sum(1 for p in polarities if p > 0.1)

            return {
                "sentence_count": len(sentences),
                "polarities": polarities,
                "avg_polarity": (
                    sum(polarities) / len(polarities) if polarities else 0.0
                ),
                "negative_sentences": neg_count,
                "positive_sentences": pos_count,
                "neutral_sentences": len(sentences) - neg_count - pos_count,
            }
        except Exception as exc:
            logger.debug("Sentence-level analysis failed: %s", exc)
            return {
                "sentence_count": 0,
                "polarities": [],
                "avg_polarity": 0.0,
                "negative_sentences": 0,
                "positive_sentences": 0,
                "neutral_sentences": 0,
            }

    def extract_subjective_words(self, text: str) -> List[str]:
        """Extract subjective opinion words from text.

        Higher TextBlob subjectivity indicates opinion-bearing words.

        Args:
            text: Review text.

        Returns:
            List of tokens identified as subjective.
        """
        try:
            tokens = word_tokenize(text.lower())
            subjective_tokens = []

            for token in tokens:
                if token.isalpha():
                    blob = TextBlob(token)
                    # Tokens with high subjectivity (>0.5) are opinion words
                    if blob.sentiment.subjectivity > 0.5:
                        subjective_tokens.append(token)

            return subjective_tokens
        except Exception as exc:
            logger.debug("Subjective word extraction failed: %s", exc)
            return []

    def word_frequency(self, text: str, top_n: int = 10) -> List[Tuple[str, int]]:
        """Analyze word frequency (excluding stop words).

        Args:
            text: Review text.
            top_n: Number of top words to return.

        Returns:
            List of (word, frequency) tuples, sorted by frequency.
        """
        try:
            tokens = word_tokenize(text.lower())
            filtered_tokens = [
                t for t in tokens if t.isalpha() and t not in self.stop_words
            ]

            freq_dist = FreqDist(filtered_tokens)
            return freq_dist.most_common(top_n)
        except Exception as exc:
            logger.debug("Word frequency analysis failed: %s", exc)
            return []

    def extract_keywords(
        self, text: str, top_n: int = 15
    ) -> Tuple[List[str], Dict[str, float]]:
        """Extract keywords using frequency and TF-IDF-like scoring.

        Args:
            text: Review text.
            top_n: Number of keywords to extract.

        Returns:
            Tuple of (keyword_list, keyword_scores_dict).
        """
        try:
            freq_list = self.word_frequency(text, top_n=top_n)
            keywords = [word for word, _ in freq_list]
            scores = {word: float(freq) for word, freq in freq_list}
            return keywords, scores
        except Exception as exc:
            logger.debug("Keyword extraction failed: %s", exc)
            return [], {}

    def tokenize_and_analyze(self, text: str) -> Dict:
        """Comprehensive token-based analysis.

        Args:
            text: Review text.

        Returns:
            Dictionary with:
            - token_count: Total tokens
            - word_count: Alphabetic tokens (excluding stop words)
            - avg_word_length: Average word length
            - vocabulary_richness: Unique words / total words
        """
        try:
            tokens = word_tokenize(text.lower())
            words = [t for t in tokens if t.isalpha()]
            non_stop_words = [w for w in words if w not in self.stop_words]

            avg_word_length = (
                sum(len(w) for w in words) / len(words) if words else 0
            )
            vocab_richness = (
                len(set(non_stop_words)) / len(non_stop_words)
                if non_stop_words
                else 0
            )

            return {
                "token_count": len(tokens),
                "word_count": len(words),
                "non_stop_word_count": len(non_stop_words),
                "avg_word_length": avg_word_length,
                "vocabulary_richness": vocab_richness,
                "unique_words": len(set(non_stop_words)),
            }
        except Exception as exc:
            logger.debug("Tokenization analysis failed: %s", exc)
            return {
                "token_count": 0,
                "word_count": 0,
                "non_stop_word_count": 0,
                "avg_word_length": 0.0,
                "vocabulary_richness": 0.0,
                "unique_words": 0,
            }

    def comprehensive_nltk_analysis(self, text: str) -> Dict:
        """Perform comprehensive NLTK-based analysis.

        Args:
            text: Review text.

        Returns:
            Dictionary containing all NLTK analyses.
        """
        return {
            "textblob": self.analyze_textblob(text),
            "sentence_sentiment": self.analyze_sentence_level(text),
            "tokenization": self.tokenize_and_analyze(text),
            "subjective_words": self.extract_subjective_words(text),
            "keywords": self.extract_keywords(text)[0],
        }


class EnsembleSentimentAnalyzer:
    """Combine VADER, BERT, TextBlob for robust sentiment prediction."""

    def __init__(self, vader_analyzer, bert_analyzer):
        """Initialize with existing analyzers.

        Args:
            vader_analyzer: VADER SentimentAnalyzer instance
            bert_analyzer: BERT SentimentAnalyzer instance
        """
        self.vader = vader_analyzer
        self.bert = bert_analyzer
        self.textblob_analyzer = NLTKSentimentAnalyzer()

    def ensemble_score(
        self, text: str, weights: Dict[str, float] = None
    ) -> Dict[str, float]:
        """Compute ensemble sentiment score from multiple methods.

        Args:
            text: Review text.
            weights: Optional weights for each method
                (default: equal weights {vader:0.33, bert:0.33, textblob:0.34}).

        Returns:
            Dictionary with:
            - ensemble_score: Weighted average [-1, 1]
            - vader_score: VADER score
            - bert_score: BERT score
            - textblob_score: TextBlob polarity
            - agreement: How much methods agree (std dev of scores)
        """
        if weights is None:
            weights = {"vader": 0.33, "bert": 0.33, "textblob": 0.34}

        try:
            # Get individual scores
            vader_score = self.vader.analyze(text)
            bert_score = self.bert.analyze(text)
            textblob_result = self.textblob_analyzer.analyze_textblob(text)
            textblob_score = textblob_result["polarity"]

            # Compute weighted ensemble
            ensemble = (
                weights.get("vader", 0.33) * vader_score
                + weights.get("bert", 0.33) * bert_score
                + weights.get("textblob", 0.34) * textblob_score
            )

            # Calculate agreement (std deviation of normalized scores)
            scores = [vader_score, bert_score, textblob_score]
            mean_score = sum(scores) / len(scores)
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            std_dev = variance ** 0.5

            return {
                "ensemble_score": ensemble,
                "vader_score": vader_score,
                "bert_score": bert_score,
                "textblob_score": textblob_score,
                "agreement": 1.0 - min(std_dev, 1.0),  # Normalized [0, 1]
                "consensus_label": self._score_to_label(ensemble),
            }
        except Exception as exc:
            logger.debug("Ensemble analysis failed: %s", exc)
            return {
                "ensemble_score": 0.0,
                "vader_score": 0.0,
                "bert_score": 0.0,
                "textblob_score": 0.0,
                "agreement": 0.0,
                "consensus_label": "neutral",
            }

    @staticmethod
    def _score_to_label(score: float) -> str:
        """Convert score to label."""
        if score > 0.3:
            return "positive"
        if score < -0.3:
            return "negative"
        return "neutral"
