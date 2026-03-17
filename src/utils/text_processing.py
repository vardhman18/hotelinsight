"""
HotelInsight - Text Processing Utilities
==========================================
Low-level text helpers used by the data-cleaning pipeline and NLP analysis
modules.  All functions are stateless and operate on plain strings.
"""

import re
import string
from typing import List

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Compiled regex patterns (compiled once at import time for performance)
# ---------------------------------------------------------------------------

_URL_PATTERN = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$\-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
    re.IGNORECASE,
)
_EXTRA_SPACE_PATTERN = re.compile(r"\s+")
_SPECIAL_CHARS_PATTERN = re.compile(r"[^a-zA-Z0-9\s']")
_NEGATION_PATTERN = re.compile(r"\b(not|no|never|neither|nor|don't|doesn't|didn't|"
                                r"won't|wouldn't|can't|cannot|isn't|aren't|wasn't|"
                                r"weren't|haven't|hasn't|hadn't|shouldn't|couldn't)\b",
                                re.IGNORECASE)


def clean_text(text: str) -> str:
    """Preprocess a single review text for NLP tasks.

    Steps applied in order:

    1. Lowercase conversion.
    2. URL removal.
    3. Special character removal (keeps apostrophes for negation terms).
    4. Extra whitespace normalisation.
    5. Leading/trailing whitespace stripping.

    Negations (``don't``, ``won't``, ``not``, etc.) are deliberately
    preserved so that sentiment models receive accurate negation signals.

    Args:
        text: Raw review string.

    Returns:
        Cleaned text string.  Returns an empty string for ``None`` or
        non-string inputs rather than raising.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Step 1: Lowercase
    text = text.lower()

    # Step 2: Remove URLs
    text = _URL_PATTERN.sub(" ", text)

    # Step 3: Remove special characters (keep sentiment-relevant punctuation)
    text = _SPECIAL_CHARS_PATTERN.sub(" ", text)

    # Step 4: Collapse multiple spaces/tabs/newlines into a single space
    text = _EXTRA_SPACE_PATTERN.sub(" ", text)

    # Step 5: Strip
    return text.strip()


def tokenize(text: str) -> List[str]:
    """Split text into a list of lowercase word tokens.

    Simple whitespace tokenisation.  For more advanced tokenisation
    (sub-words, punctuation handling) use the transformer tokeniser directly.

    Args:
        text: Input text (should be pre-cleaned with :func:`clean_text`).

    Returns:
        List of token strings.
    """
    if not text:
        return []
    return text.lower().split()


def remove_stopwords(tokens: List[str], stopwords: set) -> List[str]:
    """Remove stopword tokens from a token list.

    Args:
        tokens: List of lowercase word tokens.
        stopwords: Set of words to remove (e.g. NLTK ``stopwords.words("english")``).

    Returns:
        Filtered token list with stopwords removed.
    """
    return [t for t in tokens if t not in stopwords]


def contains_negation(text: str) -> bool:
    """Detect whether a text contains a negation word.

    Useful for flagging sentences like "room was *not* clean" where
    simple keyword matching might otherwise assign a positive label.

    Args:
        text: Review text (raw or cleaned).

    Returns:
        ``True`` if any negation word is found, ``False`` otherwise.
    """
    return bool(_NEGATION_PATTERN.search(text or ""))


def truncate_text(text: str, max_words: int = 100) -> str:
    """Truncate text to a maximum number of words.

    Used for display purposes in the Streamlit UI (review excerpts).

    Args:
        text: Input text.
        max_words: Maximum word count before truncation.

    Returns:
        Truncated text with an ellipsis appended if truncation occurred.
    """
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def word_count(text: str) -> int:
    """Return the number of whitespace-separated words in *text*.

    Args:
        text: Input text.

    Returns:
        Integer word count (0 for empty/null inputs).
    """
    if not text or not text.strip():
        return 0
    return len(text.split())


def combine_review_parts(negative: str, positive: str) -> str:
    """Merge separate negative and positive review fields into one text.

    The Kaggle dataset stores reviews split across ``Negative_Review`` and
    ``Positive_Review`` columns.  This function joins them with a separator
    while ignoring placeholder values like *"No Negative"* or *"Nothing"*.

    Args:
        negative: Text from the ``Negative_Review`` column.
        positive: Text from the ``Positive_Review`` column.

    Returns:
        Combined review string.  Returns an empty string if both parts are
        empty or placeholder values.
    """
    _placeholders = {
        "no negative", "nothing", "none", "n/a", "no positive",
        "nothing to note", "no comment", "no complaints",
    }

    def _valid(part: str) -> str:
        if not isinstance(part, str):
            return ""
        stripped = part.strip().lower()
        if stripped in _placeholders or not stripped:
            return ""
        return part.strip()

    neg = _valid(negative)
    pos = _valid(positive)

    if neg and pos:
        return f"{neg} {pos}"
    return neg or pos
