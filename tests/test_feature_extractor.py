"""Tests for feature extraction module."""

import pytest
from src.feature_extractor import (
    extract_text_features,
    extract_statistical_features,
    extract_all_features,
    count_words,
    count_sentences,
    get_feature_names,
)


def test_count_words():
    """Test word counting."""
    assert count_words("Hello world") == 2
    assert count_words("What is machine learning?") == 4
    assert count_words("") == 0


def test_count_sentences():
    """Test sentence counting."""
    assert count_sentences("Hello world.") == 1
    assert count_sentences("First. Second.") == 2
    assert count_sentences("Question? Answer!") == 2
    assert count_sentences("No punctuation") == 1


def test_extract_text_features():
    """Test text feature extraction."""
    text = "What is machine learning? It is a subset of artificial intelligence."
    features = extract_text_features(text)

    # Check that all features are present
    expected_keys = [
        "text_length",
        "word_count",
        "avg_word_length",
        "sentence_count",
        "avg_sentence_length",
        "flesch_kincaid_grade",
        "difficult_word_ratio",
    ]

    for key in expected_keys:
        assert key in features
        assert isinstance(features[key], (int, float))

    # Check reasonable values
    assert features["word_count"] > 0
    assert features["text_length"] > 0
    assert 0 <= features["flesch_kincaid_grade"] <= 18


def test_extract_statistical_features():
    """Test statistical feature extraction."""
    features = extract_statistical_features(avg_time=120, correct_percent=75)

    expected_keys = [
        "avg_time_normalized",
        "correct_percent_normalized",
        "difficulty_score",
    ]

    for key in expected_keys:
        assert key in features
        assert 0 <= features[key] <= 1  # Should be normalized


def test_extract_all_features():
    """Test combined feature extraction."""
    text = "Explain the concept of recursion in programming."
    features = extract_all_features(text, avg_time=180, correct_percent=45)

    # Should have 10 features total
    assert len(features) == 10

    # Check all features are numeric
    for key, value in features.items():
        assert isinstance(value, (int, float))


def test_get_feature_names():
    """Test feature names retrieval."""
    names = get_feature_names()

    assert len(names) == 10
    assert isinstance(names, list)
    assert all(isinstance(name, str) for name in names)

    # Check key features are present
    assert "flesch_kincaid_grade" in names
    assert "difficulty_score" in names
    assert "avg_time_normalized" in names


def test_easy_question_features():
    """Test features of an easy question."""
    text = "What is 2+2?"
    features = extract_all_features(text, avg_time=30, correct_percent=95)

    # Easy questions should have low difficulty score
    assert features["difficulty_score"] < 0.5


def test_hard_question_features():
    """Test features of a hard question."""
    text = "Derive the solution for the wave equation using separation of variables method."
    features = extract_all_features(text, avg_time=300, correct_percent=25)

    # Hard questions should have high difficulty score
    assert features["difficulty_score"] > 0.5
