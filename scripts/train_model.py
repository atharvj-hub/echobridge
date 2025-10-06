"""
Main training script for EchoBridge emotion detection model
Orchestrates data preparation, model training, and evaluation
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from echobridge.data.preprocessing import EmotionDataPreprocessor
from echobridge.models.bert_classifier import BertEmotionClassifier
from echobridge.models.spacy_pipeline import EmotionSpacyPipeline
from echobridge.utils.logger import setup_logger

# Setup logger
logger = setup_logger("train_model")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train EchoBridge emotion detection model")
    
    parser.add_argument(
        "--data-paths",
        nargs="+",
        required=True,
        help="Paths to training data CSV files"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training (default: 16)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models/saved_models/bert_emotion_latest",
        help="Directory to save trained model"
    )
    parser.add_argument(
        "--balance-classes",
        action="store_true",
        help="Balance emotion classes using oversampling"
    )
    
    return parser.parse_args()

def main():
    """Main training pipeline"""
    
    logger.info("="*70)
    logger.info("EchoBridge Emotion Detection Model Training")
    logger.info("="*70)
    
    # Parse arguments
    args = parse_args()
    
    logger.info(f"\nTraining Configuration:")
    logger.info(f"  Data paths: {args.data_paths}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Balance classes: {args.balance_classes}")
    logger.info(f"  Save directory: {args.save_dir}")
    
    # Step 1: Data Preprocessing
    logger.info("\n" + "="*70)
    logger.info("STEP 1: Data Preprocessing")
    logger.info("="*70)
    
    preprocessor = EmotionDataPreprocessor()
    
    # Load datasets
    logger.info("\nLoading datasets...")
    df = preprocessor.load_emotion_dataset(args.data_paths)
    
    # Balance classes if requested
    if args.balance_classes:
        logger.info("\nBalancing emotion classes...")
        df = preprocessor.balance_classes(df)
    
    # Create train/val/test splits
    logger.info("\nCreating train/validation/test splits...")
    X_train, X_val, X_test, y_train, y_val, y_test = \
        preprocessor.create_train_test_splits(df)
    
    # Save processed data
    preprocessor.save_processed_data(
        X_train, X_val, X_test,
        y_train, y_val, y_test
    )
    
    # Step 2: Model Training
    logger.info("\n" + "="*70)
    logger.info("STEP 2: Model Training")
    logger.info("="*70)
    
    # Initialize BERT classifier
    logger.info("\nInitializing BERT classifier...")
    classifier = BertEmotionClassifier()
    
    # Create data loaders
    logger.info("\nCreating data loaders...")
    train_loader, val_loader = classifier.create_data_loaders(
        X_train, y_train,
        X_val, y_val,
        batch_size=args.batch_size
    )
    
    # Fine-tune model
    logger.info("\nStarting model fine-tuning...")
    start_time = datetime.now()
    
    classifier.fine_tune_model(
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    training_time = datetime.now() - start_time
