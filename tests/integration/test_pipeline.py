"""Integration tests for complete emotion detection pipeline"""

import pytest
from echobridge.models.bert_classifier import BertEmotionClassifier
from echobridge.models.spacy_pipeline import EmotionSpacyPipeline
from echobridge.visualization.emotion_renderer import AccessibleEmotionRenderer

@pytest.mark.slow
def test_complete_pipeline(sample_texts):
    """Test complete emotion detection pipeline"""
    
    # Initialize components
    bert_classifier = BertEmotionClassifier()
    spacy_pipeline = EmotionSpacyPipeline(bert_classifier)
    renderer = AccessibleEmotionRenderer()
    
    for text in sample_texts:
        # Process text
        doc = spacy_pipeline.process_text(text)
        
        # Check emotion detection
        assert hasattr(doc._, 'emotion')
        assert hasattr(doc._, 'emotion_confidence')
        assert doc._.emotion in ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
        assert 0.0 <= doc._.emotion_confidence <= 1.0
        
        # Check rendering
        html = renderer.render_html(text, doc._.emotion, doc._.emotion_confidence)
        assert text in html
        assert 'emotion-text' in html
