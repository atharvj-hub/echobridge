"""Unit tests for emotion color mapper"""

import pytest
from echobridge.visualization.color_mapper import EmotionColorMapper

def test_emotion_colors(emotion_labels):
    """Test that all emotions have color mappings"""
    color_mapper = EmotionColorMapper()
    
    for emotion in emotion_labels:
        assert emotion in color_mapper.emotion_colors
        assert 'hex' in color_mapper.emotion_colors[emotion]
        assert 'rgb' in color_mapper.emotion_colors[emotion]

def test_get_emotion_style():
    """Test emotion style generation"""
    color_mapper = EmotionColorMapper()
    
    style = color_mapper.get_emotion_style('joy', 0.8)
    
    assert 'color' in style
    assert 'background_color' in style
    assert 'pattern' in style
    assert 'text_indicator' in style
    assert style['intensity'] == 0.8

def test_check_color_contrast():
    """Test WCAG color contrast checking"""
    color_mapper = EmotionColorMapper()
    
    # Test with known colors
    result = color_mapper.check_color_contrast('#000000', '#FFFFFF')
    
    assert 'contrast_ratio' in result
    assert result['contrast_ratio'] > 4.5  # Should pass WCAG AA
    assert result['wcag_aa_normal'] == True

def test_generate_css():
    """Test CSS generation"""
    color_mapper = EmotionColorMapper()
    
    css = color_mapper.generate_css_classes()
    
    assert '.emotion-text' in css
    assert '.emotion-joy' in css
    assert '.emotion-sadness' in css
