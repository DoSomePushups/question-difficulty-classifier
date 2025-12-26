"""Question difficulty classifier."""

import logging
import pickle
from typing import Union, List
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from src.feature_extractor import extract_all_features, get_feature_names

logger = logging.getLogger(__name__)


class QuestionDifficultyClassifier:
    """
    Classifier for predicting question difficulty levels.

    Supports multiple algorithms:
    - naive_bayes: Gaussian Naive Bayes
    - svm: Support Vector Machine
    - random_forest: Random Forest Classifier
    """

    DIFFICULTY_LEVELS = ["easy", "medium", "hard"]

    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize classifier.

        Args:
            model_type: Type of model ('naive_bayes', 'svm', 'random_forest')

        Raises:
            ValueError: If model_type is not supported
        """
        supported_models = {"naive_bayes", "svm", "random_forest"}
        if model_type not in supported_models:
            raise ValueError(f"Model type must be one of {supported_models}")

        self.model_type = model_type
        self.difficulty_levels = self.DIFFICULTY_LEVELS
        self.label_encoder = LabelEncoder()
        self.feature_names = get_feature_names()

        # Initialize model
        if model_type == "naive_bayes":
            self.model = GaussianNB()
        elif model_type == "svm":
            self.model = SVC(kernel="rbf", probability=True, random_state=42)
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )

        logger.info(f"Initialized {model_type} classifier")

    def fit(self, df: pd.DataFrame) -> "QuestionDifficultyClassifier":
        """
        Train the classifier on a dataset.

        Args:
            df: DataFrame with columns: text, avg_time, correct_percent, difficulty

        Returns:
            Self for method chaining
        """
        # Extract features
        X = self._extract_features(df)
        y = df["difficulty"].values

        # Encode labels
        self.label_encoder.fit(self.difficulty_levels)
        y_encoded = self.label_encoder.transform(y)

        # Train model
        self.model.fit(X, y_encoded)
        logger.info(f"Trained {self.model_type} on {len(df)} samples")

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> List[str]:
        """
        Predict difficulty levels.

        Args:
            X: Features (DataFrame or array with columns: text, avg_time, correct_percent)

        Returns:
            List of predicted difficulty levels
        """
        if isinstance(X, pd.DataFrame):
            features = self._extract_features(X)
        else:
            features = X

        y_pred_encoded = self.model.predict(features)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)

        return y_pred.tolist()

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict probability for each class.

        Args:
            X: Features

        Returns:
            Array of shape (n_samples, n_classes)
        """
        if isinstance(X, pd.DataFrame):
            features = self._extract_features(X)
        else:
            features = X

        return self.model.predict_proba(features)

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features from DataFrame."""
        features = []

        for _, row in df.iterrows():
            feature_dict = extract_all_features(
                row["text"], row["avg_time"], row["correct_percent"]
            )
            feature_vector = [feature_dict[name] for name in self.feature_names]
            features.append(feature_vector)

        return np.array(features)

    def save(self, filepath: str) -> None:
        """Save trained model to file."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {filepath}")

    @staticmethod
    def load(filepath: str) -> "QuestionDifficultyClassifier":
        """Load trained model from file."""
        with open(filepath, "rb") as f:
            classifier = pickle.load(f)
        logger.info(f"Model loaded from {filepath}")
        return classifier
