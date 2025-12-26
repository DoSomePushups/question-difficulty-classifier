"""Tests for evaluation module."""

import pytest
import pandas as pd
import numpy as np
from src.evaluation import (
    evaluate_model,
    get_confusion_matrix,
    get_classification_report,
    generate_report
)
from src.classifier import QuestionDifficultyClassifier


@pytest.fixture
def trained_classifier_and_data():
    """Create a trained classifier and test data."""
    df = pd.DataFrame({
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
    
    classifier = QuestionDifficultyClassifier(model_type='random_forest')
    classifier.fit(df)
    
    return classifier, df


def test_evaluate_model_returns_dict(trained_classifier_and_data):
    """Test that evaluate_model returns a dictionary."""
    classifier, df = trained_classifier_and_data
    
    metrics = evaluate_model(classifier, df)
    
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics


def test_evaluate_model_metrics_in_range(trained_classifier_and_data):
    """Test that all metrics are in valid range."""
    classifier, df = trained_classifier_and_data
    
    metrics = evaluate_model(classifier, df)
    
    # All metrics should be between 0 and 1
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1'] <= 1


def test_get_confusion_matrix_shape(trained_classifier_and_data):
    """Test that confusion matrix has correct shape."""
    classifier, df = trained_classifier_and_data
    X = df[['text', 'avg_time', 'correct_percent']]
    y = df['difficulty'].values
    
    cm = get_confusion_matrix(classifier, X, y)
    
    # Should be 3x3 (3 classes)
    assert cm.shape == (3, 3)


def test_get_confusion_matrix_values_positive(trained_classifier_and_data):
    """Test that confusion matrix values are non-negative."""
    classifier, df = trained_classifier_and_data
    X = df[['text', 'avg_time', 'correct_percent']]
    y = df['difficulty'].values
    
    cm = get_confusion_matrix(classifier, X, y)
    
    # All values should be non-negative
    assert (cm >= 0).all()


def test_get_classification_report_returns_string(trained_classifier_and_data):
    """Test that classification report is a string."""
    classifier, df = trained_classifier_and_data
    X = df[['text', 'avg_time', 'correct_percent']]
    y = df['difficulty'].values
    
    report = get_classification_report(classifier, X, y)
    
    assert isinstance(report, str)
    assert len(report) > 0


def test_get_classification_report_contains_classes(trained_classifier_and_data):
    """Test that classification report contains class names."""
    classifier, df = trained_classifier_and_data
    X = df[['text', 'avg_time', 'correct_percent']]
    y = df['difficulty'].values
    
    report = get_classification_report(classifier, X, y)
    
    assert 'easy' in report
    assert 'medium' in report
    assert 'hard' in report


def test_generate_report_returns_string(trained_classifier_and_data):
    """Test that generate_report returns a comprehensive string."""
    classifier, df = trained_classifier_and_data
    metrics = {
        'accuracy': 0.85,
        'precision': 0.84,
        'recall': 0.83,
        'f1': 0.82
    }
    predictions = classifier.predict(df[['text', 'avg_time', 'correct_percent']])
    
    report = generate_report(classifier, metrics, df, predictions)
    
    assert isinstance(report, str)
    assert len(report) > 0


def test_generate_report_contains_metrics(trained_classifier_and_data):
    """Test that generated report contains all metrics."""
    classifier, df = trained_classifier_and_data
    metrics = {
        'accuracy': 0.85,
        'precision': 0.84,
        'recall': 0.83,
        'f1': 0.82
    }
    predictions = classifier.predict(df[['text', 'avg_time', 'correct_percent']])
    
    report = generate_report(classifier, metrics, df, predictions)
    
    # Check that metrics are in report
    assert '0.8500' in report or '0.85' in report
    assert 'Accuracy' in report
    assert 'Precision' in report


def test_generate_report_contains_model_info(trained_classifier_and_data):
    """Test that report contains model type information."""
    classifier, df = trained_classifier_and_data
    metrics = {
        'accuracy': 0.85,
        'precision': 0.84,
        'recall': 0.83,
        'f1': 0.82
    }
    predictions = classifier.predict(df[['text', 'avg_time', 'correct_percent']])
    
    report = generate_report(classifier, metrics, df, predictions)
    
    # Should mention the model type
    assert 'Random' in report or 'random' in report or 'RANDOM' in report


def test_generate_report_contains_confusion_matrix(trained_classifier_and_data):
    """Test that report contains confusion matrix."""
    classifier, df = trained_classifier_and_data
    metrics = {
        'accuracy': 0.85,
        'precision': 0.84,
        'recall': 0.83,
        'f1': 0.82
    }
    predictions = classifier.predict(df[['text', 'avg_time', 'correct_percent']])
    
    report = generate_report(classifier, metrics, df, predictions)
    
    # Should contain confusion matrix header
    assert 'CONFUSION MATRIX' in report or 'Confusion Matrix' in report
