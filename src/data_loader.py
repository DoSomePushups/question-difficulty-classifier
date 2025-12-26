"""Data loading and preprocessing module."""

import logging
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def load_questions_csv(filepath: str) -> pd.DataFrame:
    """
    Load questions from CSV file.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with columns: id, text, avg_time, correct_percent, difficulty
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Validate required columns
    required_columns = ['id', 'text', 'avg_time', 'correct_percent', 'difficulty']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    logger.info(f"Loaded {len(df)} questions from {filepath}")
    
    # Basic preprocessing
    df = df.dropna(subset=['text', 'difficulty'])
    df = df[df['text'].str.len() > 0]  # Remove empty texts
    
    logger.info(f"After cleaning: {len(df)} questions remain")
    
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split data into training and test sets.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of test set (0-1)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['difficulty']
    )
    
    logger.info(f"Data split: {len(train_df)} train, {len(test_df)} test")
    
    return train_df, test_df


def get_difficulty_distribution(df: pd.DataFrame) -> dict:
    """Get the distribution of difficulty levels in the dataset."""
    return df['difficulty'].value_counts().to_dict()
