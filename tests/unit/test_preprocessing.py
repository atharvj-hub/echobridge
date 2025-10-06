"""Unit tests for data preprocessing module"""

import pytest
import pandas as pd
from echobridge.data.preprocessing import EmotionDataPreprocessor

def test_clean_text():
    """Test text cleaning function"""
    preprocessor = EmotionDataPreprocessor()
    
    # Test URL removal
    text = "Check this out http://example.com amazing!"
    cleaned = preprocessor.clean_text(text)
    assert "http" not in cleaned
    
    # Test special character removal
    text = "Hello @user #hashtag world!"
    cleaned = preprocessor.clean_text(text)
    assert "@" not in cleaned
    assert "#" not in cleaned
    
    # Test whitespace normalization
    text = "Hello    world    !"
    cleaned = preprocessor.clean_text(text)
    assert "  " not in cleaned

def test_balance_classes():
    """Test class balancing"""
    preprocessor = EmotionDataPreprocessor()
    
    # Create imbalanced dataset
    df = pd.DataFrame({
        'text': ['text1', 'text2', 'text3', 'text4', 'text5'],
        'emotion': ['joy', 'joy', 'joy', 'sadness', 'anger']
    })
    
    balanced_df = preprocessor.balance_classes(df)
    
    # Check if balanced
    emotion_counts = balanced_df['emotion'].value_counts()
    assert len(emotion_counts.unique()) == 1  # All counts should be equal

def test_create_splits():
    """Test train/val/test split creation"""
    preprocessor = EmotionDataPreprocessor()
    
    # Create sample dataset
    df = pd.DataFrame({
        'text': [f'text {i}' for i in range(100)],
        'emotion': ['joy'] * 50 + ['sadness'] * 50
    })
    
    X_train, X_val, X_test, y_train, y_val, y_test = \
        preprocessor.create_train_test_splits(df)
    
    # Check sizes
    total_size = len(X_train) + len(X_val) + len(X_test)
    assert total_size == 100
    assert len(X_train) > len(X_val)
    assert len(X_train) > len(X_test)
