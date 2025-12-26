"""Tests for data loading module."""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
from src.data_loader import load_questions_csv, split_data, get_difficulty_distribution


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file for testing."""
    data = {
        "id": [1, 2, 3, 4, 5],
        "text": [
            "What is 2+2?",
            "Explain recursion",
            "What is the capital of France?",
            "Derive the quadratic formula",
            "What is photosynthesis?",
        ],
        "avg_time": [30, 180, 20, 240, 60],
        "correct_percent": [95, 45, 98, 30, 75],
        "difficulty": ["easy", "hard", "easy", "hard", "medium"],
    }

    df = pd.DataFrame(data)
    csv_path = tmp_path / "test_questions.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


def test_load_questions_csv_success(sample_csv):
    """Test successful loading of CSV file."""
    df = load_questions_csv(str(sample_csv))

    assert len(df) == 5
    assert list(df.columns) == [
        "id",
        "text",
        "avg_time",
        "correct_percent",
        "difficulty",
    ]
    assert df["difficulty"].unique().tolist() == ["easy", "hard", "medium"]


def test_load_questions_csv_missing_file():
    """Test error handling for missing file."""
    with pytest.raises(FileNotFoundError):
        load_questions_csv("nonexistent_file.csv")


def test_load_questions_csv_missing_columns(tmp_path):
    """Test error handling for missing required columns."""
    incomplete_data = {
        "id": [1, 2],
        "text": ["Question 1", "Question 2"],
        # Missing: avg_time, correct_percent, difficulty
    }

    df = pd.DataFrame(incomplete_data)
    csv_path = tmp_path / "incomplete.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        load_questions_csv(str(csv_path))


def test_split_data(sample_csv):
    """Test data splitting."""
    df = load_questions_csv(str(sample_csv))
    train_df, test_df = split_data(df, test_size=0.4)

    assert len(train_df) + len(test_df) == len(df)
    assert len(test_df) / len(df) == pytest.approx(0.4, abs=0.1)


def test_get_difficulty_distribution(sample_csv):
    """Test difficulty distribution calculation."""
    df = load_questions_csv(str(sample_csv))
    distribution = get_difficulty_distribution(df)

    assert distribution["easy"] == 2
    assert distribution["hard"] == 2
    assert distribution["medium"] == 1
