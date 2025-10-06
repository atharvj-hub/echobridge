"""
Model evaluation script for EchoBridge
Comprehensive evaluation on test data with metrics and visualizations
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from echobridge.models.bert_classifier import BertEmotionClassifier
from echobridge.data.data_loader import EmotionDataLoader
from echobridge.utils.logger import setup_logger

logger = setup_logger("evaluate_model")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate EchoBridge emotion detection model")
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/processed/test_data.csv",
        help="Path to test data CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    
    return parser.parse_args()

def plot_confusion_matrix(cm, labels, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Emotion Detection Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Confusion matrix saved to {save_path}")
    plt.close()

def main():
    """Main evaluation pipeline"""
    
    logger.info("="*70)
    logger.info("EchoBridge Model Evaluation")
    logger.info("="*70)
    
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model
    logger.info(f"\nLoading model from {args.model_path}...")
    classifier = BertEmotionClassifier()
    classifier.load_model(args.model_path)
    
    # Load test data
    logger.info(f"\nLoading test data from {args.test_data}...")
    test_df = pd.read_csv(args.test_data)
    X_test = test_df['text']
    y_test = test_df['emotion']
    
    logger.info(f"Test set size: {len(X_test)} samples")
    
    # Make predictions
    logger.info("\nMaking predictions on test set...")
    predictions = []
    confidences = []
    
    for text in X_test:
        emotion, confidence = classifier.predict_emotion(text)
        predictions.append(emotion)
        confidences.append(confidence)
    
    # Generate classification report
    logger.info("\n" + "="*70)
    logger.info("Classification Report")
    logger.info("="*70)
    
    report = classification_report(
        y_test,
        predictions,
        target_names=classifier.emotion_labels,
        digits=4
    )
    print(report)
    
    # Save report
    report_path = output_dir / "classification_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"\nClassification report saved to {report_path}")
    
    # Generate confusion matrix
    logger.info("\nGenerating confusion matrix...")
    cm = confusion_matrix(y_test, predictions, labels=classifier.emotion_labels)
    
    # Plot and save confusion matrix
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, classifier.emotion_labels, cm_path)
    
    # Calculate per-emotion accuracy
    logger.info("\n" + "="*70)
    logger.info("Per-Emotion Accuracy")
    logger.info("="*70)
    
    per_emotion_accuracy = {}
    for i, emotion in enumerate(classifier.emotion_labels):
        mask = y_test == emotion
        if mask.sum() > 0:
            correct = (predictions[mask] == emotion).sum()
            accuracy = correct / mask.sum()
            per_emotion_accuracy[emotion] = accuracy
            logger.info(f"{emotion.title():12} : {accuracy:.2%}")
    
    # Calculate average confidence
    logger.info("\n" + "="*70)
    logger.info("Confidence Statistics")
    logger.info("="*70)
    
    avg_confidence = sum(confidences) / len(confidences)
    min_confidence = min(confidences)
    max_confidence = max(confidences)
    
    logger.info(f"Average Confidence: {avg_confidence:.2%}")
    logger.info(f"Min Confidence: {min_confidence:.2%}")
    logger.info(f"Max Confidence: {max_confidence:.2%}")
    
    # Save detailed results
    results_df = pd.DataFrame({
        'text': X_test,
        'true_emotion': y_test,
        'predicted_emotion': predictions,
        'confidence': confidences,
        'correct': y_test == predictions
    })
    
    results_path = output_dir / "detailed_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nDetailed results saved to {results_path}")
    
    # Overall accuracy
    overall_accuracy = (results_df['correct'].sum() / len(results_df))
    
    logger.info("\n" + "="*70)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*70)
    logger.info(f"\nOverall Accuracy: {overall_accuracy:.2%}")
    logger.info(f"Total Samples: {len(X_test)}")
    logger.info(f"Average Confidence: {avg_confidence:.2%}")
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("\n✅ Evaluation completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"\n❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
