"""Tests for classifier module."""

import pytest
import pandas as pd
import numpy as np
from src.classifier import QuestionDifficultyClassifier


@pytest.fixture
def sample_questions():
    """Create sample question data for testing."""
    return pd.DataFrame({
        'text': [
            'What is 2+2?',
            'Explain recursion',
            'What is the capital of France?',
            'Derive the quadratic formula',
            'What is photosynthesis?',
            'Define a variable',
            'What does OOP stand for?',
            'Implement a sorting algorithm',
            'Name the planets',
            'Discuss the theory of evolution',
        ],
        'avg_time': [30, 180, 20, 240, 60, 25, 90, 200, 40, 150],
        'correct_percent': [95, 45, 98, 30, 75, 92, 65, 35, 88, 50],
        'difficulty': ['easy', 'hard', 'easy', 'hard', 'medium', 'easy', 'medium', 'hard', 'easy', 'medium'],
    })


def test_classifier_initialization():
    """Test classifier initialization."""
    classifier = QuestionDifficultyClassifier(model_type='random_forest')
    
    assert classifier.model_type == 'random_forest'
    assert classifier.difficulty_levels == ['easy', 'medium', 'hard']
    assert len(classifier.feature_names) == 10


def test_classifier_invalid_model_type():
    """Test error on invalid model type."""
    with pytest.raises(ValueError):
        QuestionDifficultyClassifier(model_type='invalid_model')


def test_classifier_training(sample_questions):
    """Test classifier training."""
    classifier = QuestionDifficultyClassifier(model_type='random_forest')
    classifier.fit(sample_questions)
    
    # Check that model is trained
    assert classifier.model is not None


def test_classifier_prediction(sample_questions):
    """Test making predictions."""
    classifier = QuestionDifficultyClassifier(model_type='random_forest')
    classifier.fit(sample_questions)
    
    # Make predictions on same data
    predictions = classifier.predict(sample_questions[['text', 'avg_time', 'correct_percent']])
    
    assert len(predictions) == len(sample_questions)
    assert all(pred in ['easy', 'medium', 'hard'] for pred in predictions)


def test_classifier_predict_proba(sample_questions):
    """Test probability predictions."""
    classifier = QuestionDifficultyClassifier(model_type='random_forest')
    classifier.fit(sample_questions)
    
    probas = classifier.predict_proba(sample_questions[['text', 'avg_time', 'correct_percent']])
    
    assert probas.shape == (len(sample_questions), 3)  # 3 difficulty levels
    assert np.allclose(probas.sum(axis=1), 1)  # Probabilities sum to 1


def test_naive_bayes_classifier(sample_questions):
    """Test Naive Bayes classifier."""
    classifier = QuestionDifficultyClassifier(model_type='naive_bayes')
    classifier.fit(sample_questions)
    
    predictions = classifier.predict(sample_questions[['text', 'avg_time', 'correct_percent']])
    assert len(predictions) == len(sample_questions)


def test_svm_classifier(sample_questions):
    """Test SVM classifier."""
    classifier = QuestionDifficultyClassifier(model_type='svm')
    classifier.fit(sample_questions)
    
    predictions = classifier.predict(sample_questions[['text', 'avg_time', 'correct_percent']])
    assert len(predictions) == len(sample_questions)


def test_classifier_save_load(sample_questions, tmp_path):
    """Test saving and loading classifier."""
    filepath = str(tmp_path / 'test_classifier.pkl')
    
    # Train and save
    classifier = QuestionDifficultyClassifier(model_type='random_forest')
    classifier.fit(sample_questions)
    classifier.save(filepath)
    
    # Load
    loaded_classifier = QuestionDifficultyClassifier.load(filepath)
    
    # Test that loaded model works
    predictions = loaded_classifier.predict(sample_questions[['text', 'avg_time', 'correct_percent']])
    assert len(predictions) == len(sample_questions)


def test_classifier_consistency(sample_questions):
    """Test that predictions are consistent."""
    classifier = QuestionDifficultyClassifier(model_type='random_forest')
    classifier.fit(sample_questions)
    
    X = sample_questions[['text', 'avg_time', 'correct_percent']]
    
    # Make predictions twice
    pred1 = classifier.predict(X)
    pred2 = classifier.predict(X)
    
    # Should be identical
    assert pred1 == pred2
