"""Feature extraction from question text."""

import logging
import re
import math
from typing import List, Dict
import numpy as np

logger = logging.getLogger(__name__)

# Common English words (for readability calculation)
FLESCH_KINCAID_CONSONANTS = {
    "b",
    "c",
    "d",
    "f",
    "g",
    "h",
    "j",
    "k",
    "l",
    "m",
    "n",
    "p",
    "q",
    "r",
    "s",
    "t",
    "v",
    "w",
    "x",
    "y",
    "z",
}

DIFFICULT_WORDS = {
    "algorithm",
    "recursion",
    "derivative",
    "polynomial",
    "hypothesis",
    "paradigm",
    "methodology",
    "synthesis",
    "complexity",
    "abstraction",
    "optimization",
    "simulation",
    "theorem",
    "constant",
    "variable",
    "coefficient",
    "exponential",
    "logarithmic",
    "trigonometric",
}


def count_sentences(text: str) -> int:
    """Count sentences in text."""
    sentences = re.split(r"[.!?]+", text)
    return max(1, len([s for s in sentences if s.strip()]))


def count_words(text: str) -> int:
    """Count words in text."""
    words = text.split()
    return len(words)


def count_syllables(text: str) -> int:
    """Estimate syllable count in text."""
    words = text.lower().split()
    syllable_count = 0

    for word in words:
        word = re.sub(r"[^a-z]", "", word)
        if len(word) > 0:
            # Simple heuristic: vowel groups = approximate syllables
            vowels = "aeiouy"
            syllables = 0
            previous_was_vowel = False

            for char in word:
                is_vowel = char in vowels
                if is_vowel and not previous_was_vowel:
                    syllables += 1
                previous_was_vowel = is_vowel

            # Adjust for silent e
            if word.endswith("e"):
                syllables -= 1

            # Ensure at least 1 syllable
            syllables = max(1, syllables)
            syllable_count += syllables

    return max(1, syllable_count)


def extract_text_features(text: str) -> Dict[str, float]:
    """
    Extract text-based features from a question.

    Args:
        text: Question text

    Returns:
        Dictionary with features:
        - text_length: Length in characters
        - word_count: Number of words
        - avg_word_length: Average characters per word
        - sentence_count: Number of sentences
        - avg_sentence_length: Average words per sentence
        - flesch_kincaid_grade: Readability score (0-18+)
        - difficult_word_ratio: Percentage of complex words
    """
    text_length = len(text)
    word_count = count_words(text)
    sentence_count = count_sentences(text)
    syllable_count = count_syllables(text)

    # Avoid division by zero
    avg_word_length = text_length / word_count if word_count > 0 else 0
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

    # Flesch-Kincaid Grade Level
    # Formula: 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
    if word_count > 0 and sentence_count > 0:
        flesch_kincaid = (
            0.39 * (word_count / sentence_count)
            + 11.8 * (syllable_count / word_count)
            - 15.59
        )
        flesch_kincaid = max(0, min(18, flesch_kincaid))  # Clamp 0-18
    else:
        flesch_kincaid = 0

    # Count difficult words
    words_lower = text.lower().split()
    difficult_count = sum(
        1 for w in words_lower if any(dw in w for dw in DIFFICULT_WORDS)
    )
    difficult_word_ratio = difficult_count / word_count if word_count > 0 else 0

    return {
        "text_length": text_length,
        "word_count": word_count,
        "avg_word_length": avg_word_length,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "flesch_kincaid_grade": flesch_kincaid,
        "difficult_word_ratio": difficult_word_ratio,
    }


def extract_statistical_features(
    avg_time: float, correct_percent: float
) -> Dict[str, float]:
    """
    Extract statistical features from student response data.

    Args:
        avg_time: Average time to answer (seconds)
        correct_percent: Percentage of correct answers (0-100)

    Returns:
        Dictionary with features:
        - avg_time: Time to answer (normalized)
        - correct_percent: Correctness (normalized)
        - difficulty_score: Composite difficulty (0-1)
    """
    # Normalize average time (0-1 scale, assuming 0-300 seconds is normal)
    normalized_time = min(1.0, avg_time / 300.0)

    # Normalize correct percent (0-1 scale)
    normalized_correct = correct_percent / 100.0

    # Composite difficulty: high time + low correctness = high difficulty
    difficulty_score = normalized_time * 0.5 + (1 - normalized_correct) * 0.5

    return {
        "avg_time_normalized": normalized_time,
        "correct_percent_normalized": normalized_correct,
        "difficulty_score": difficulty_score,
    }


def extract_all_features(
    text: str, avg_time: float, correct_percent: float
) -> Dict[str, float]:
    """
    Extract all features for a question.

    Args:
        text: Question text
        avg_time: Average time to answer
        correct_percent: Percentage of correct answers

    Returns:
        Combined feature dictionary
    """
    text_features = extract_text_features(text)
    stat_features = extract_statistical_features(avg_time, correct_percent)

    combined = {**text_features, **stat_features}
    return combined


def get_feature_names() -> List[str]:
    """Get list of all feature names in order."""
    return [
        "text_length",
        "word_count",
        "avg_word_length",
        "sentence_count",
        "avg_sentence_length",
        "flesch_kincaid_grade",
        "difficult_word_ratio",
        "avg_time_normalized",
        "correct_percent_normalized",
        "difficulty_score",
    ]
