"""Data loading utilities for EchoBridge"""

import pandas as pd
from pathlib import Path
from typing import Tuple
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from echobridge.config import PROCESSED_DATA_PATH

class EmotionDataLoader:
    """Load processed emotion datasets"""
    
    def __init__(self, data_dir: Path = PROCESSED_DATA_PATH):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory containing processed data files
        """
        self.data_dir = Path(data_dir)
        print(f"✓ EmotionDataLoader initialized with directory: {self.data_dir}")
    
    def load_train_data(self) -> Tuple[pd.Series, pd.Series]:
        """Load training data"""
        train_path = self.data_dir / "train_data.csv"
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found at {train_path}")
        
        df = pd.read_csv(train_path)
        print(f"✓ Loaded {len(df)} training samples")
        return df['text'], df['emotion']
    
    def load_val_data(self) -> Tuple[pd.Series, pd.Series]:
        """Load validation data"""
        val_path = self.data_dir / "val_data.csv"
        if not val_path.exists():
            raise FileNotFoundError(f"Validation data not found at {val_path}")
        
        df = pd.read_csv(val_path)
        print(f"✓ Loaded {len(df)} validation samples")
        return df['text'], df['emotion']
    
    def load_test_data(self) -> Tuple[pd.Series, pd.Series]:
        """Load test data"""
        test_path = self.data_dir / "test_data.csv"
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found at {test_path}")
        
        df = pd.read_csv(test_path)
        print(f"✓ Loaded {len(df)} test samples")
        return df['text'], df['emotion']
    
    def load_all_data(self) -> dict:
        """Load all datasets"""
        return {
            'train': self.load_train_data(),
            'val': self.load_val_data(),
            'test': self.load_test_data()
        }
