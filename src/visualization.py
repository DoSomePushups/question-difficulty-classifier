"""Visualization utilities."""

import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    classifier, X, y_true, filepath: str = "confusion_matrix.png"
) -> None:
    """
    Plot and save confusion matrix.

    Args:
        classifier: Trained classifier
        X: Features
        y_true: True labels
        filepath: Where to save the figure
    """
    # Get predictions
    y_pred = classifier.predict(X)

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classifier.difficulty_levels)

    # Create figure
    plt.figure(figsize=(8, 6))

    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classifier.difficulty_levels,
        yticklabels=classifier.difficulty_levels,
        cbar_kws={"label": "Count"},
    )

    plt.title(
        f"Confusion Matrix - {classifier.model_type.upper()}",
        fontsize=14,
        fontweight="bold",
    )
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()

    # Save
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    logger.info(f"Confusion matrix saved to {filepath}")
    plt.close()


def plot_feature_importance(
    classifier, filepath: str = "feature_importance.png"
) -> None:
    """
    Plot and save feature importance (for tree-based models).

    Args:
        classifier: Trained classifier
        filepath: Where to save the figure
    """
    if not hasattr(classifier.model, "feature_importances_"):
        logger.warning("Model does not support feature importance")
        return

    # Get feature importance
    importances = classifier.model.feature_importances_
    features = classifier.feature_names

    # Sort by importance
    indices = np.argsort(importances)[::-1][:10]  # Top 10

    # Create figure
    plt.figure(figsize=(10, 6))

    plt.bar(range(len(indices)), importances[indices], align="center")
    plt.xticks(
        range(len(indices)), [features[i] for i in indices], rotation=45, ha="right"
    )

    plt.title("Top 10 Feature Importance", fontsize=14, fontweight="bold")
    plt.ylabel("Importance", fontsize=12)
    plt.xlabel("Features", fontsize=12)
    plt.tight_layout()

    # Save
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    logger.info(f"Feature importance plot saved to {filepath}")
    plt.close()


def plot_difficulty_distribution(
    df, filepath: str = "difficulty_distribution.png"
) -> None:
    """
    Plot and save difficulty distribution.

    Args:
        df: DataFrame with 'difficulty' column
        filepath: Where to save the figure
    """
    # Count difficulties
    counts = df["difficulty"].value_counts().sort_index()

    # Create figure
    plt.figure(figsize=(8, 6))

    colors = ["#2ecc71", "#f39c12", "#e74c3c"]  # Green, Orange, Red
    counts.plot(kind="bar", color=colors, edgecolor="black", linewidth=1.5)

    plt.title("Question Difficulty Distribution", fontsize=14, fontweight="bold")
    plt.ylabel("Number of Questions", fontsize=12)
    plt.xlabel("Difficulty Level", fontsize=12)
    plt.xticks(rotation=0)

    # Add count labels on bars
    for i, v in enumerate(counts):
        plt.text(i, v + 1, str(v), ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()

    # Save
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    logger.info(f"Distribution plot saved to {filepath}")
    plt.close()
