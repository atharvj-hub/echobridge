"""
Emotion rendering for HTML, terminal, and browser display
Accessible emotion visualization with multi-format support
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from echobridge.visualization.color_mapper import EmotionColorMapper

class AccessibleEmotionRenderer:
    """Render emotion-coded text with full accessibility support"""
    
    def __init__(self):
        """Initialize renderer with color mapper"""
        self.color_mapper = EmotionColorMapper()
    
    def render_html(self, text: str, emotion: str, confidence: float) -> str:
        """
        Generate accessible HTML with emotion indicators
        
        Args:
            text: Input text
            emotion: Detected emotion
            confidence: Confidence score
            
        Returns:
            HTML string with emotion styling
        """
        style = self.color_mapper.get_emotion_style(emotion, confidence)
        
        html = f"""
<span class="emotion-text emotion-{emotion}" 
      style="background-color: {style['background_color']}; 
             border-left-color: {style['border_color']};
             padding: 4px 8px;
             border-radius: 4px;
             display: inline-block;"
      aria-label="{style['aria_label']}"
      title="Detected emotion: {emotion.title()} ({confidence:.0%} confidence)"
      role="mark">
    <span class="emotion-pattern" aria-hidden="true">{style['pattern']}</span>
    <span class="emotion-indicator sr-only">{style['text_indicator']}</span>
    <span class="emotion-content">{text}</span>
</span>
        """
        
        return html.strip()
    
    def render_terminal(self, text: str, emotion: str, confidence: float) -> str:
        """
        Render colored text for terminal/console display
        
        Args:
            text: Input text
            emotion: Detected emotion
            confidence: Confidence score
            
        Returns:
            Colored terminal string
        """
        try:
            from colorama import Fore, Style, init
            init(autoreset=True)
            
            color_map = {
                'joy': Fore.GREEN,
                'sadness': Fore.BLUE,
                'anger': Fore.RED,
                'fear': Fore.MAGENTA,
                'surprise': Fore.YELLOW,
                'disgust': Fore.BLACK,
                'neutral': Fore.WHITE
            }
            
            color = color_map.get(emotion, Fore.WHITE)
            pattern = self.color_mapper.emotion_patterns[emotion]
            
            return f"{color}{pattern} {text} [{emotion.upper()} {confidence:.0%}]{Style.RESET_ALL}"
        
        except ImportError:
            # Fallback without colors
            pattern = self.color_mapper.emotion_patterns[emotion]
            return f"{pattern} {text} [{emotion.upper()} {confidence:.0%}]"
    
    def render_plain_text(self, text: str, emotion: str, confidence: float) -> str:
        """
        Render plain text with emotion indicators
        
        Args:
            text: Input text
            emotion: Detected emotion
            confidence: Confidence score
            
        Returns:
            Plain text string
        """
        pattern = self.color_mapper.emotion_patterns[emotion]
        indicator = self.color_mapper.emotion_indicators[emotion]
        
        return f"{pattern} {indicator} {text} ({emotion.title()}: {confidence:.0%})"
    
    def render_json(self, text: str, emotion: str, confidence: float) -> dict:
        """
        Render as JSON for API responses
        
        Args:
            text: Input text
            emotion: Detected emotion
            confidence: Confidence score
            
        Returns:
            Dictionary with emotion data
        """
        style = self.color_mapper.get_emotion_style(emotion, confidence)
        
        return {
            'text': text,
            'emotion': emotion,
            'confidence': confidence,
            'display': {
                'pattern': style['pattern'],
                'color': style['color'],
                'background_color': style['background_color'],
                'text_indicator': style['text_indicator']
            },
            'accessibility': {
                'aria_label': style['aria_label'],
                'description': f"{emotion.title()} emotion detected with {confidence:.0%} confidence"
            }
        }
