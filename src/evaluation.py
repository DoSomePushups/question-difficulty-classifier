"""Model evaluation utilities."""

import logging
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)


def evaluate_model(classifier, df: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate classifier on a dataset.

    Args:
        classifier: Trained classifier
        df: DataFrame with 'text', 'avg_time', 'correct_percent', 'difficulty'

    Returns:
        Dictionary of metrics
    """
    # Predict
    X = df[["text", "avg_time", "correct_percent"]]
    y_true = df["difficulty"].values
    y_pred = classifier.predict(X)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    logger.info(f"Evaluation metrics: {metrics}")
    return metrics


def get_confusion_matrix(classifier, X: pd.DataFrame, y_true) -> np.ndarray:
    """Get confusion matrix."""
    y_pred = classifier.predict(X)

    # Encode labels for confusion matrix
    labels = classifier.difficulty_levels
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return cm


def get_classification_report(classifier, X: pd.DataFrame, y_true: np.ndarray) -> str:
    """Get detailed classification report."""
    y_pred = classifier.predict(X)

    report = classification_report(
        y_true,
        y_pred,
        labels=classifier.difficulty_levels,
        target_names=classifier.difficulty_levels,
    )

    return report


def generate_report(
    classifier, metrics: Dict, df: pd.DataFrame, predictions: list
) -> str:
    """
    Generate a comprehensive report.

    Args:
        classifier: Trained classifier
        metrics: Evaluation metrics
        df: Original dataset
        predictions: Model predictions

    Returns:
        Formatted report string
    """
    X = df[["text", "avg_time", "correct_percent"]]
    y_true = df["difficulty"].values

    report_lines = [
        "=" * 60,
        "ASSESSMENT QUESTION DIFFICULTY CLASSIFICATION REPORT",
        "=" * 60,
        "",
        f"Model Type: {classifier.model_type.upper()}",
        f"Total Samples: {len(df)}",
        "",
        "PERFORMANCE METRICS",
        "-" * 60,
        f"Accuracy:  {metrics['accuracy']:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall:    {metrics['recall']:.4f}",
        f"F1-Score:  {metrics['f1']:.4f}",
        "",
        "CLASS DISTRIBUTION",
        "-" * 60,
    ]

    # Add class distribution
    for label in classifier.difficulty_levels:
        count = (y_true == label).sum()
        percentage = (count / len(y_true)) * 100
        report_lines.append(
            f"{label.upper():10} {count:3d} samples ({percentage:5.1f}%)"
        )

    report_lines.extend(
        [
            "",
            "CONFUSION MATRIX",
            "-" * 60,
        ]
    )

    # Get confusion matrix
    cm = get_confusion_matrix(classifier, X, y_true)

    # Format confusion matrix
    header = "        " + " ".join(
        f"{label:>8}" for label in classifier.difficulty_levels
    )
    report_lines.append(header)

    for i, label in enumerate(classifier.difficulty_levels):
        row = f"{label:>8}" + " ".join(
            f"{cm[i, j]:>8}" for j in range(len(classifier.difficulty_levels))
        )
        report_lines.append(row)

    report_lines.extend(
        [
            "",
            "DETAILED CLASSIFICATION REPORT",
            "-" * 60,
            get_classification_report(classifier, X, y_true),
        ]
    )

    return "\n".join(report_lines)
