"""
Pytest configuration and fixtures for EchoBridge tests
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

@pytest.fixture
def sample_texts():
    """Sample texts for testing"""
    return [
        "I'm so happy about this!",
        "This makes me really sad.",
        "I'm furious about this!",
        "That's disgusting behavior.",
        "Wow, I didn't expect that!",
        "I'm scared about the future."
    ]

@pytest.fixture
def sample_emotions():
    """Sample emotion labels"""
    return ['joy', 'sadness', 'anger', 'disgust', 'surprise', 'fear']

@pytest.fixture
def emotion_labels():
    """All emotion labels"""
    return ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
