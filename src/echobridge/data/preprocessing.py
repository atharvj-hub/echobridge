"""
Data preprocessing module for emotion detection
Handles data cleaning, balancing, and preparation for BERT training
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
import re
from typing import Tuple, List
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from echobridge.config import BERT_MODEL_NAME, MAX_SEQUENCE_LENGTH

class EmotionDataPreprocessor:
    """Preprocessor for emotion detection datasets"""
    
    def __init__(self, model_name: str = BERT_MODEL_NAME, max_length: int = MAX_SEQUENCE_LENGTH):
        """
        Initialize preprocessor with BERT tokenizer
        
        Args:
            model_name: Pre-trained BERT model name
            max_length: Maximum sequence length for tokenization
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.label_encoder = LabelEncoder()
        self.max_length = max_length
        print(f"✓ Initialized EmotionDataPreprocessor with {model_name}")
    
    def load_emotion_dataset(self, file_paths: List[str]) -> pd.DataFrame:
        """
        Load and combine multiple emotion datasets
        
        Args:
            file_paths: List of paths to CSV files
            
        Returns:
            Combined DataFrame with text and emotion columns
        """
        datasets = []
        for file_path in file_paths:
            try:
                df = pd.read_csv(file_path)
                datasets.append(df)
                print(f"✓ Loaded {len(df)} records from {file_path}")
            except Exception as e:
                print(f"✗ Error loading {file_path}: {e}")
        
        if not datasets:
            raise ValueError("No datasets were successfully loaded")
        
        combined_data = pd.concat(datasets, ignore_index=True)
        print(f"\n✓ Total records loaded: {len(combined_data)}")
        
        return combined_data
    
    def clean_text(self, text: str) -> str:
        """
        Comprehensive text cleaning
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\'-]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Convert to lowercase
        text = text.lower()
        
        return text
    
    def balance_classes(self, df: pd.DataFrame, target_column: str = 'emotion') -> pd.DataFrame:
        """
        Balance emotion classes using oversampling
        
        Args:
            df: Input DataFrame
            target_column: Name of the emotion label column
            
        Returns:
            Balanced DataFrame
        """
        from sklearn.utils import resample
        
        class_counts = df[target_column].value_counts()
        max_size = class_counts.max()
        
        print("\n📊 Class distribution before balancing:")
        print(class_counts)
        
        balanced_data = []
        for emotion_class in class_counts.index:
            class_subset = df[df[target_column] == emotion_class]
            
            if len(class_subset) < max_size:
                upsampled = resample(
                    class_subset,
                    replace=True,
                    n_samples=max_size,
                    random_state=42
                )
                balanced_data.append(upsampled)
            else:
                balanced_data.append(class_subset)
        
        balanced_df = pd.concat(balanced_data, ignore_index=True)
        
        print("\n📊 Class distribution after balancing:")
        print(balanced_df[target_column].value_counts())
        
        return balanced_df
    
    def prepare_bert_input(self, texts: pd.Series, labels: pd.Series) -> Tuple:
        """
        Tokenize texts for BERT input
        
        Args:
            texts: Series of text strings
            labels: Series of emotion labels
            
        Returns:
            Tuple of (encodings, encoded_labels)
        """
        encodings = self.tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        return encodings, encoded_labels
    
    def create_train_test_splits(
        self,
        df: pd.DataFrame,
        text_column: str = 'text',
        label_column: str = 'emotion',
        test_size: float = 0.3
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Create stratified train/validation/test splits
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            label_column: Name of label column
            test_size: Proportion for test+validation sets
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Clean texts
        X = df[text_column].apply(self.clean_text)
        y = df[label_column]
        
        # Remove empty texts
        mask = X.str.len() > 0
        X = X[mask]
        y = y[mask]
        
        print(f"\n✓ Cleaned dataset size: {len(X)} samples")
        
        # First split: train vs (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )
        
        # Second split: val vs test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.5,
            random_state=42,
            stratify=y_temp
        )
        
        print(f"\n📊 Dataset splits:")
        print(f"  Training set:   {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Test set:       {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_processed_data(
        self,
        X_train: pd.Series,
        X_val: pd.Series,
        X_test: pd.Series,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
        output_dir: str = "data/processed"
    ):
        """Save processed datasets to CSV files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save training data
        train_df = pd.DataFrame({'text': X_train, 'emotion': y_train})
        train_df.to_csv(f"{output_dir}/train_data.csv", index=False)
        
        # Save validation data
        val_df = pd.DataFrame({'text': X_val, 'emotion': y_val})
        val_df.to_csv(f"{output_dir}/val_data.csv", index=False)
        
        # Save test data
        test_df = pd.DataFrame({'text': X_test, 'emotion': y_test})
        test_df.to_csv(f"{output_dir}/test_data.csv", index=False)
        
        print(f"\n✓ Processed data saved to {output_dir}/")
