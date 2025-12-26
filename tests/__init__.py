"""
Assessment Question Difficulty Classifier

A machine learning system for automatically classifying the difficulty level
of educational test questions based on text analysis and student response data.

Author: Student
Date: 2025-12-26
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Student"

from src.classifier import QuestionDifficultyClassifier
from src.data_loader import load_questions_csv, split_data
from src.feature_extractor import extract_all_features

__all__ = [
    "QuestionDifficultyClassifier",
    "load_questions_csv",
    "split_data",
    "extract_all_features",
]
