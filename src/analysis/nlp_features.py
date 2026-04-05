"""
HotelInsight - Advanced NLP Features Extractor
================================================
Uses spaCy for advanced NLP tasks: POS tagging, entity recognition,
dependency parsing, noun phrase extraction, and named entity extraction.

Complements VADER/BERT sentiment analysis with deeper linguistic insights.
"""

import os
from typing import Dict, List, Tuple, Optional
import spacy
from spacy.tokens import Doc
import logging

from src.utils.logger import get_logger

logger = get_logger(__name__)

# spaCy model identifier
SPACY_MODEL = "en_core_web_sm"


class NLPFeaturesExtractor:
    """Extract advanced NLP features from reviews using spaCy.

    Features extracted:
    - POS tags (part-of-speech): nouns, verbs, adjectives, etc.
    - Named entities: locations, organizations, persons
    - Noun phrases: multi-word noun phrases from reviews
    - Dependency parsing: syntactic structures
    - Lemmatization: normalized word forms
    - Stop words: filtered for keyword analysis
    """

    def __init__(self):
        """Initialize spaCy NLP model."""
        self.nlp = self._load_spacy_model()

    @staticmethod
    def _load_spacy_model():
        """Load spaCy model, download if necessary."""
        try:
            nlp = spacy.load(SPACY_MODEL)
            logger.info("spaCy model '%s' loaded successfully.", SPACY_MODEL)
        except OSError:
            logger.warning(
                "spaCy model '%s' not found. Downloading...", SPACY_MODEL
            )
            import subprocess
            import sys

            subprocess.check_call(
                [sys.executable, "-m", "spacy", "download", SPACY_MODEL]
            )
            nlp = spacy.load(SPACY_MODEL)
            logger.info("spaCy model '%s' downloaded and loaded.", SPACY_MODEL)

        return nlp

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text using spaCy.

        Args:
            text: Review text.

        Returns:
            Dictionary mapping entity labels to lists of entities found:
            - PERSON: Person names
            - ORG: Organizations
            - GPE: Geopolitical entities (countries, cities)
            - PRODUCT: Products/services
            - FACILITY: Buildings/facilities
        """
        doc = self.nlp(text)
        entities: Dict[str, List[str]] = {}

        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)

        return entities

    def extract_noun_phrases(self, text: str) -> List[str]:
        """Extract noun chunks (multi-word noun phrases).

        Useful for identifying specific complaint topics like
        'bathroom cleanliness', 'front desk staff', etc.

        Args:
            text: Review text.

        Returns:
            List of noun phrases found in the text.
        """
        doc = self.nlp(text)
        return [chunk.text for chunk in doc.noun_chunks]

    def extract_adjectives(self, text: str) -> List[Tuple[str, str]]:
        """Extract adjectives and the nouns they modify.

        Identifies sentiment-bearing words and their targets.

        Args:
            text: Review text.

        Returns:
            List of (adjective, noun) tuples.
        """
        doc = self.nlp(text)
        adj_noun_pairs = []

        for token in doc:
            if token.pos_ == "ADJ":
                # Find the noun this adjective modifies
                for child in token.children:
                    if child.pos_ == "NOUN":
                        adj_noun_pairs.append((token.text.lower(), child.text.lower()))
                        break
                # If no child noun, look at head
                if token.head.pos_ == "NOUN":
                    adj_noun_pairs.append((token.text.lower(), token.head.text.lower()))

        return adj_noun_pairs

    def extract_pos_tags(self, text: str) -> Dict[str, List[str]]:
        """Extract POS (Part-of-Speech) tags and group by tag type.

        Returns structured information about word types in the text.

        Args:
            text: Review text.

        Returns:
            Dictionary mapping POS tags to lists of tokens:
            - NOUN: Nouns
            - VERB: Verbs
            - ADJ: Adjectives
            - ADV: Adverbs
            - PROPN: Proper nouns
        """
        doc = self.nlp(text)
        pos_tags: Dict[str, List[str]] = {}

        for token in doc:
            if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV", "PROPN"]:
                if token.pos_ not in pos_tags:
                    pos_tags[token.pos_] = []
                pos_tags[token.pos_].append(token.text.lower())

        return pos_tags

    def extract_lemmas(self, text: str) -> List[str]:
        """Extract lemmatized (base) forms of words.

        Useful for normalizing text for analysis.

        Args:
            text: Review text.

        Returns:
            List of lemmatized tokens (excluding stop words).
        """
        doc = self.nlp(text)
        lemmas = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and token.is_alpha
        ]
        return lemmas

    def analyze_syntactic_complexity(self, text: str) -> Dict[str, float]:
        """Analyze syntactic complexity metrics.

        Args:
            text: Review text.

        Returns:
            Dictionary with metrics:
            - avg_dependency_depth: Average depth in dependency parse tree
            - sentence_count: Number of sentences
            - avg_sentence_length: Average tokens per sentence
        """
        doc = self.nlp(text)

        # Count sentences
        sentences = list(doc.sents)
        sentence_count = len(sentences)

        # Calculate average sentence length
        avg_sentence_length = (
            len(doc) / sentence_count if sentence_count > 0 else 0
        )

        # Calculate average dependency depth
        def get_depth(token, depth=0):
            if not list(token.children):
                return depth
            return max(get_depth(child, depth + 1) for child in token.children)

        depths = [get_depth(token) for token in doc]
        avg_depth = sum(depths) / len(depths) if depths else 0

        return {
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_length,
            "avg_dependency_depth": avg_depth,
        }

    def comprehensive_analysis(self, text: str) -> Dict:
        """Perform comprehensive NLP analysis on a review.

        Combines all extraction methods into a single analysis.

        Args:
            text: Review text.

        Returns:
            Dictionary containing:
            - entities: Named entities
            - noun_phrases: Extracted noun phrases
            - adjectives: Adjective-noun pairs
            - pos_tags: POS tag groupings
            - lemmas: Lemmatized tokens
            - complexity: Syntactic complexity metrics
        """
        return {
            "entities": self.extract_entities(text),
            "noun_phrases": self.extract_noun_phrases(text),
            "adjectives": self.extract_adjectives(text),
            "pos_tags": self.extract_pos_tags(text),
            "lemmas": self.extract_lemmas(text),
            "complexity": self.analyze_syntactic_complexity(text),
        }

    def batch_analysis(self, texts: List[str]) -> List[Dict]:
        """Analyze multiple texts efficiently.

        Args:
            texts: List of review texts.

        Returns:
            List of analysis dictionaries (one per text).
        """
        results = []
        for i, text in enumerate(texts):
            if i % 100 == 0 and i > 0:
                logger.info("Processed %d / %d texts", i, len(texts))
            results.append(self.comprehensive_analysis(text))
        logger.info("Batch analysis complete: %d texts processed.", len(texts))
        return results
