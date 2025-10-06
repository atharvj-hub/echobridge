"""Helper utility functions"""

import re
from typing import Optional, List
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from echobridge.config import EMOTION_LABELS

def clean_text(text: str) -> str:
    """
    Clean text for processing
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:\'-]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def validate_emotion(emotion: str) -> bool:
    """
    Validate if emotion is in supported list
    
    Args:
        emotion: Emotion string
        
    Returns:
        True if valid, False otherwise
    """
    return emotion.lower() in EMOTION_LABELS

def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to maximum length
    
    Args:
        text: Input text
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def format_confidence(confidence: float) -> str:
    """
    Format confidence score as percentage
    
    Args:
        confidence: Confidence value (0-1)
        
    Returns:
        Formatted percentage string
    """
    return f"{confidence*100:.1f}%"

def get_emotion_color_code(emotion: str) -> str:
    """
    Get ANSI color code for terminal output
    
    Args:
        emotion: Emotion name
        
    Returns:
        ANSI color code
    """
    color_codes = {
        'joy': '\033[92m',      # Green
        'sadness': '\033[94m',  # Blue
        'anger': '\033[91m',    # Red
        'fear': '\033[95m',     # Magenta
        'surprise': '\033[93m', # Yellow
        'disgust': '\033[90m',  # Gray
        'neutral': '\033[97m'   # White
    }
    return color_codes.get(emotion, '\033[0m')

def batch_texts(texts: List[str], batch_size: int = 32) -> List[List[str]]:
    """
    Batch texts for efficient processing
    
    Args:
        texts: List of text strings
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    return [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
