"""Input validation utilities"""

from typing import Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from echobridge.config import EMOTION_LABELS, MAX_SEQUENCE_LENGTH

def validate_input(text: str, max_length: int = MAX_SEQUENCE_LENGTH) -> Tuple[bool, Optional[str]]:
    """
    Validate user input text
    
    Args:
        text: Input text
        max_length: Maximum allowed length
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if empty
    if not text or not text.strip():
        return False, "Text cannot be empty"
    
    # Check length
    if len(text) > max_length:
        return False, f"Text exceeds maximum length of {max_length} characters"
    
    # Check if text is too short
    if len(text.strip()) < 3:
        return False, "Text is too short (minimum 3 characters)"
    
    return True, None

def validate_confidence(confidence: float) -> Tuple[bool, Optional[str]]:
    """
    Validate confidence score
    
    Args:
        confidence: Confidence value
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(confidence, (int, float)):
        return False, "Confidence must be a number"
    
    if confidence < 0.0 or confidence > 1.0:
        return False, "Confidence must be between 0.0 and 1.0"
    
    return True, None

def validate_emotion_label(emotion: str) -> Tuple[bool, Optional[str]]:
    """
    Validate emotion label
    
    Args:
        emotion: Emotion string
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not emotion:
        return False, "Emotion cannot be empty"
    
    if emotion.lower() not in EMOTION_LABELS:
        return False, f"Invalid emotion. Must be one of: {', '.join(EMOTION_LABELS)}"
    
    return True, None

def validate_batch_size(batch_size: int) -> Tuple[bool, Optional[str]]:
    """
    Validate batch size
    
    Args:
        batch_size: Batch size value
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(batch_size, int):
        return False, "Batch size must be an integer"
    
    if batch_size < 1:
        return False, "Batch size must be at least 1"
    
    if batch_size > 128:
        return False, "Batch size cannot exceed 128"
    
    return True, None
