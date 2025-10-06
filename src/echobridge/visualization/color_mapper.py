"""
Emotion-to-color mapping with accessibility compliance
WCAG-compliant color schemes for emotion visualization
"""

from typing import Dict, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from echobridge.config import WCAG_CONTRAST_RATIO_AA, WCAG_CONTRAST_RATIO_AAA

class EmotionColorMapper:
    """Map emotions to accessible colors following WCAG guidelines"""
    
    def __init__(self):
        """Initialize emotion-color mappings with accessibility features"""
        
        # Define emotion-color mapping with WCAG-compliant colors
        self.emotion_colors = {
            'joy': {
                'hex': '#4CAF50',
                'rgb': (76, 175, 80),
                'name': 'Green',
                'description': 'Success green for positive emotions'
            },
            'sadness': {
                'hex': '#2196F3',
                'rgb': (33, 150, 243),
                'name': 'Blue',
                'description': 'Calming blue for sadness'
            },
            'anger': {
                'hex': '#F44336',
                'rgb': (244, 67, 54),
                'name': 'Red',
                'description': 'Alert red for anger'
            },
            'fear': {
                'hex': '#9C27B0',
                'rgb': (156, 39, 176),
                'name': 'Purple',
                'description': 'Deep purple for fear'
            },
            'surprise': {
                'hex': '#FF9800',
                'rgb': (255, 152, 0),
                'name': 'Orange',
                'description': 'Bright orange for surprise'
            },
            'disgust': {
                'hex': '#795548',
                'rgb': (121, 85, 72),
                'name': 'Brown',
                'description': 'Earth brown for disgust'
            },
            'neutral': {
                'hex': '#607D8B',
                'rgb': (96, 125, 139),
                'name': 'Gray',
                'description': 'Neutral gray for neutral emotions'
            }
        }
        
        # Emoji patterns for each emotion
        self.emotion_patterns = {
            'joy': '😊',
            'sadness': '😢',
            'anger': '😠',
            'fear': '😨',
            'surprise': '😮',
            'disgust': '🤢',
            'neutral': '😐'
        }
        
        # Text-based indicators for screen readers
        self.emotion_indicators = {
            'joy': '[HAPPY]',
            'sadness': '[SAD]',
            'anger': '[ANGRY]',
            'fear': '[FEARFUL]',
            'surprise': '[SURPRISED]',
            'disgust': '[DISGUSTED]',
            'neutral': '[NEUTRAL]'
        }
    
    def get_emotion_style(self, emotion: str, intensity: float = 1.0) -> Dict:
        """
        Get comprehensive styling for emotion display
        
        Args:
            emotion: Emotion name
            intensity: Confidence/intensity value (0.0 to 1.0)
            
        Returns:
            Dictionary with color, pattern, and accessibility features
        """
        if emotion not in self.emotion_colors:
            emotion = 'neutral'
        
        base_color = self.emotion_colors[emotion]
        
        # Adjust color intensity based on confidence
        rgb = base_color['rgb']
        adjusted_rgb = tuple(int(c * intensity + (255 * (1 - intensity))) for c in rgb)
        
        return {
            'color': base_color['hex'],
            'background_color': f'rgba({adjusted_rgb[0]}, {adjusted_rgb[1]}, {adjusted_rgb[2]}, 0.2)',
            'border_color': base_color['hex'],
            'pattern': self.emotion_patterns[emotion],
            'text_indicator': self.emotion_indicators[emotion],
            'aria_label': f"Emotion: {emotion.title()} with {intensity:.0%} confidence",
            'intensity': intensity,
            'emotion_name': emotion,
            'color_name': base_color['name']
        }
    
    def generate_css_classes(self) -> str:
        """Generate CSS classes for web implementation"""
        css = """
        /* EchoBridge Emotion Styles */
        .emotion-text {
            padding: 4px 8px;
            border-radius: 4px;
            border-left: 4px solid;
            margin: 2px 0;
            display: inline-block;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        
        """
        
        for emotion, color_info in self.emotion_colors.items():
            css += f"""
        /* {emotion.title()} emotion styles */
        .emotion-{emotion} {{
            background-color: {color_info['hex']}20;
            border-left-color: {color_info['hex']};
            color: #000;
        }}
        
        .emotion-{emotion}::before {{
            content: "{self.emotion_patterns[emotion]} ";
            font-size: 1.1em;
            margin-right: 4px;
        }}
        
        """
        
        return css
    
    def check_color_contrast(self, foreground_color: str, background_color: str = '#FFFFFF') -> Dict:
        """
        Verify color combinations meet WCAG accessibility standards
        
        Args:
            foreground_color: Hex color code
            background_color: Hex color code (default white)
            
        Returns:
            Dictionary with contrast ratio and WCAG compliance
        """
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        def relative_luminance(rgb):
            def gamma_correction(channel):
                c = channel / 255.0
                if c <= 0.03928:
                    return c / 12.92
                else:
                    return pow((c + 0.055) / 1.055, 2.4)
            
            r, g, b = [gamma_correction(c) for c in rgb]
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
        
        fg_rgb = hex_to_rgb(foreground_color)
        bg_rgb = hex_to_rgb(background_color)
        
        fg_luminance = relative_luminance(fg_rgb)
        bg_luminance = relative_luminance(bg_rgb)
        
        if fg_luminance > bg_luminance:
            contrast_ratio = (fg_luminance + 0.05) / (bg_luminance + 0.05)
        else:
            contrast_ratio = (bg_luminance + 0.05) / (fg_luminance + 0.05)
        
        return {
            'contrast_ratio': round(contrast_ratio, 2),
            'wcag_aa_normal': contrast_ratio >= WCAG_CONTRAST_RATIO_AA,
            'wcag_aa_large': contrast_ratio >= 3.0,
            'wcag_aaa_normal': contrast_ratio >= WCAG_CONTRAST_RATIO_AAA,
            'wcag_aaa_large': contrast_ratio >= 4.5
        }
    
    def get_all_emotions(self) -> list:
        """Get list of all available emotions"""
        return list(self.emotion_colors.keys())
