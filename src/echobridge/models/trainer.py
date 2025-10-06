"""Training utilities and trainer class"""

from typing import Optional, Dict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from echobridge.models.bert_classifier import BertEmotionClassifier
from echobridge.data.preprocessing import EmotionDataPreprocessor
from echobridge.data.data_loader import EmotionDataLoader

class EmotionModelTrainer:
    """High-level trainer for emotion detection models"""
    
    def __init__(self):
        self.preprocessor = EmotionDataPreprocessor()
        self.classifier = None
        self.training_history = {
            'train_loss': [],
            'val_accuracy': []
        }
    
    def prepare_data(self, data_paths: list):
        """Load and prepare training data"""
        print("📦 Preparing data...")
        
        # Load datasets
        df = self.preprocessor.load_emotion_dataset(data_paths)
        
        # Balance classes
        df = self.preprocessor.balance_classes(df)
        
        # Create splits
        X_train, X_val, X_test, y_train, y_val, y_test = \
            self.preprocessor.create_train_test_splits(df)
        
        # Save processed data
        self.preprocessor.save_processed_data(
            X_train, X_val, X_test,
            y_train, y_val, y_test
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs: int = 3,
        batch_size: int = 16
    ):
        """Train the model"""
        print("\n🚀 Starting training pipeline...")
        
        # Initialize classifier
        self.classifier = BertEmotionClassifier()
        
        # Create data loaders
        train_loader, val_loader = self.classifier.create_data_loaders(
            X_train, y_train, X_val, y_val, batch_size
        )
        
        # Fine-tune model
        self.classifier.fine_tune_model(train_loader, val_loader, epochs)
        
        print("\n✅ Training pipeline completed!")
    
    def save(self, save_dir: str = "models/saved_models/bert_emotion_latest"):
        """Save trained model"""
        if self.classifier:
            self.classifier.save_model(save_dir)
        else:
            print("⚠️ No trained model to save")
