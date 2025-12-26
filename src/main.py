"""Main module for question difficulty classification."""

import logging
from pathlib import Path
from src.classifier import QuestionDifficultyClassifier
from src.data_loader import load_questions_csv
from src.evaluation import evaluate_model, generate_report
from src.visualization import plot_confusion_matrix, plot_feature_importance

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the classification system."""
    logger.info("Starting Question Difficulty Classifier...")

    # Define paths
    data_dir = Path("data")
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # Load data
    logger.info("Loading training data...")
    questions_df = load_questions_csv(data_dir / "sample_questions.csv")
    logger.info(f"Loaded {len(questions_df)} questions")

    # Train classifier
    logger.info("Training Random Forest classifier...")
    classifier = QuestionDifficultyClassifier(model_type="random_forest")
    classifier.fit(questions_df)
    logger.info("Classifier trained successfully")

    # Make predictions on test data
    logger.info("Making predictions on test data...")
    test_df = load_questions_csv(data_dir / "sample_questions.csv")
    predictions = classifier.predict(test_df[["text", "avg_time", "correct_percent"]])

    # Evaluate model
    logger.info("Evaluating model performance...")
    metrics = evaluate_model(classifier, test_df)
    logger.info(f"Model accuracy: {metrics['accuracy']:.3f}")

    # Generate report
    logger.info("Generating report...")
    report = generate_report(classifier, metrics, test_df, predictions)
    report_path = reports_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")

    # Visualize results
    logger.info("Generating visualizations...")
    plot_confusion_matrix(
        classifier,
        test_df[["text", "avg_time", "correct_percent"]],
        test_df["difficulty"],
        reports_dir / "confusion_matrix.png",
    )

    plot_feature_importance(classifier, reports_dir / "feature_importance.png")
    logger.info("Visualizations saved")

    logger.info("Classification complete!")
    print("\n" + "=" * 50)
    print("CLASSIFICATION RESULTS")
    print("=" * 50)
    print(report)


if __name__ == "__main__":
    main()
