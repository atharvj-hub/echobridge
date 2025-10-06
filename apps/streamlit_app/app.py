"""
Streamlit web application for EchoBridge
Interactive prototype for emotion detection and visualization
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from echobridge.models.bert_classifier import BertEmotionClassifier
from echobridge.models.spacy_pipeline import EmotionSpacyPipeline
from echobridge.visualization.color_mapper import EmotionColorMapper
from echobridge.visualization.emotion_renderer import AccessibleEmotionRenderer

# Page configuration
st.set_page_config(
    page_title="EchoBridge - Emotion Detector",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2196F3;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .emotion-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'bert_classifier' not in st.session_state:
    st.session_state.bert_classifier = None
if 'spacy_pipeline' not in st.session_state:
    st.session_state.spacy_pipeline = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize components
color_mapper = EmotionColorMapper()
renderer = AccessibleEmotionRenderer()

def load_models():
    """Load BERT and spaCy models"""
    with st.spinner("🔄 Loading AI models... This may take a minute on first run."):
        try:
            # Initialize BERT classifier
            if st.session_state.bert_classifier is None:
                st.session_state.bert_classifier = BertEmotionClassifier()
            
            # Initialize spaCy pipeline
            if st.session_state.spacy_pipeline is None:
                st.session_state.spacy_pipeline = EmotionSpacyPipeline(
                    st.session_state.bert_classifier
                )
            
            return True
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            return False

def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🌉 EchoBridge</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Real-Time Emotion-to-Tone Translator for Accessibility</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model loading
        if st.button("🔄 Load/Reload Models", use_container_width=True):
            if load_models():
                st.success("✅ Models loaded successfully!")
        
        st.divider()
        
        # Accessibility settings
        st.subheader("♿ Accessibility Options")
        use_colors = st.checkbox("Use Colors", value=True)
        use_patterns = st.checkbox("Use Emoji Patterns", value=True)
        use_text_indicators = st.checkbox("Use Text Indicators", value=True)
        
        st.divider()
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence to display emotion"
        )
        
        st.divider()
        
        # Info section
        st.subheader("ℹ️ About")
        st.info(
            "EchoBridge helps users understand emotional context in text "
            "through color-coded, accessible visualizations."
        )
        
        # Emotion legend
        st.subheader("🎨 Emotion Legend")
        for emotion in color_mapper.get_all_emotions():
            style = color_mapper.get_emotion_style(emotion, 1.0)
            st.markdown(
                f'<div style="background-color: {style["background_color"]}; '
                f'border-left: 4px solid {style["border_color"]}; '
                f'padding: 8px; margin: 4px 0; border-radius: 4px;">'
                f'{style["pattern"]} {emotion.title()}</div>',
                unsafe_allow_html=True
            )
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📝 Text Analysis", "💬 Live Chat", "📊 Statistics"])
    
    with tab1:
        text_analysis_tab(
            use_colors, use_patterns, use_text_indicators, confidence_threshold
        )
    
    with tab2:
        live_chat_tab(
            use_colors, use_patterns, use_text_indicators, confidence_threshold
        )
    
    with tab3:
        statistics_tab()

def text_analysis_tab(use_colors, use_patterns, use_text_indicators, threshold):
    """Text analysis interface"""
    
    st.header("📝 Text Emotion Analysis")
    
    # Text input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Input")
        text_input = st.text_area(
            "Enter text to analyze:",
            height=150,
            placeholder="Type or paste your text here...",
            help="Enter any text to detect its emotional tone"
        )
        
        analyze_button = st.button("🔍 Analyze Emotion", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("Quick Examples")
        examples = [
            "I'm so happy about this wonderful news!",
            "This makes me really sad and disappointed.",
            "I'm furious about what happened today!",
            "That's absolutely disgusting behavior.",
            "Wow, I didn't expect that at all!",
            "I'm really scared about the future."
        ]
        
        for idx, example in enumerate(examples):
            if st.button(f"📌 Example {idx+1}", key=f"example_{idx}", use_container_width=True):
                text_input = example
                st.rerun()
    
    # Analysis results
    if analyze_button and text_input.strip():
        if st.session_state.spacy_pipeline is None:
            st.warning("⚠️ Please load models first using the sidebar button!")
            return
        
        with st.spinner("🔍 Analyzing emotion..."):
            try:
                # Process text
                doc = st.session_state.spacy_pipeline.process_text(text_input)
                emotion = doc._.emotion
                confidence = float(doc._.emotion_confidence)
                
                if confidence >= threshold:
                    st.divider()
                    st.subheader("📊 Analysis Results")
                    
                    # Metrics row
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric("Detected Emotion", emotion.title())
                    
                    with metric_col2:
                        st.metric("Confidence", f"{confidence:.1%}")
                    
                    with metric_col3:
                        style = color_mapper.get_emotion_style(emotion, confidence)
                        st.metric("Visual Indicator", style['pattern'])
                    
                    # Visual representation
                    st.subheader("🎨 Visual Representation")
                    
                    style = color_mapper.get_emotion_style(emotion, confidence)
                    
                    if use_colors and use_patterns:
                        st.markdown(
                            f'<div class="emotion-box" style="'
                            f'background-color: {style["background_color"]}; '
                            f'border-left-color: {style["border_color"]};">'
                            f'<span style="font-size: 1.3em;">{style["pattern"]}</span> '
                            f'{text_input}</div>',
                            unsafe_allow_html=True
                        )
                    
                    if use_text_indicators:
                        st.info(f"**Screen Reader Text:** {style['text_indicator']} {text_input}")
                    
                    # Detailed analysis
                    with st.expander("🔎 Detailed Analysis"):
                        analysis = st.session_state.spacy_pipeline.analyze_text(text_input)
                        
                        st.write("**Tokens:**", ", ".join(analysis['tokens']))
                        st.write("**Lemmas:**", ", ".join(analysis['lemmas']))
                        
                        if analysis['entities']:
                            st.write("**Named Entities:**")
                            for entity, label in analysis['entities']:
                                st.write(f"- {entity} ({label})")
                        
                        if analysis['noun_chunks']:
                            st.write("**Noun Chunks:**", ", ".join(analysis['noun_chunks']))
                
                else:
                    st.warning(
                        f"⚠️ Low confidence ({confidence:.1%}). "
                        f"The model is not confident about the detected emotion."
                    )
            
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
    
    elif analyze_button:
        st.warning("⚠️ Please enter some text to analyze.")

def live_chat_tab(use_colors, use_patterns, use_text_indicators, threshold):
    """Live chat simulation interface"""
    
    st.header("💬 Live Chat Simulation")
    st.write("See emotion detection in action with real-time chat messages")
    
    # Chat input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        message_input = st.text_input(
            "Type a message:",
            key="chat_input",
            placeholder="Type your message here..."
        )
    
    with col2:
        st.write("")  # Spacing
        send_button = st.button("📤 Send", type="primary", use_container_width=True)
    
    # Send message
    if send_button and message_input.strip():
        if st.session_state.spacy_pipeline is None:
            st.warning("⚠️ Please load models first!")
            return
        
        try:
            doc = st.session_state.spacy_pipeline.process_text(message_input)
            emotion = doc._.emotion
            confidence = float(doc._.emotion_confidence)
            
            st.session_state.chat_history.append({
                'message': message_input,
                'emotion': emotion,
                'confidence': confidence,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
            
            st.rerun()
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    # Display chat history
    st.divider()
    st.subheader("💬 Chat Messages")
    
    if st.session_state.chat_history:
        # Reverse to show newest first
        for msg in reversed(st.session_state.chat_history[-10:]):
            style = color_mapper.get_emotion_style(msg['emotion'], msg['confidence'])
            
            st.markdown(
                f'<div class="emotion-box" style="'
                f'background-color: {style["background_color"]}; '
                f'border-left-color: {style["border_color"]};">'
                f'<small style="color: #666;">{msg["timestamp"]}</small><br>'
                f'<span style="font-size: 1.2em;">{style["pattern"]}</span> '
                f'{msg["message"]}'
                f'<br><small>Emotion: {msg["emotion"].title()} '
                f'({msg["confidence"]:.0%})</small></div>',
                unsafe_allow_html=True
            )
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("👋 Start chatting to see emotion detection in action!")

def statistics_tab():
    """Statistics and analytics"""
    
    st.header("📊 Statistics")
    
    if not st.session_state.chat_history:
        st.info("💡 Send some messages in the Live Chat tab to see statistics!")
        return
    
    # Emotion distribution
    from collections import Counter
    
    emotions = [msg['emotion'] for msg in st.session_state.chat_history]
    emotion_counts = Counter(emotions)
    
    st.subheader("🎭 Emotion Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart
        import pandas as pd
        df = pd.DataFrame.from_dict(emotion_counts, orient='index', columns=['Count'])
        df.index.name = 'Emotion'
        st.bar_chart(df)
    
    with col2:
        # Table
        st.dataframe(df, use_container_width=True)
    
    # Average confidence by emotion
    st.subheader("📈 Average Confidence by Emotion")
    
    emotion_confidences = {}
    for msg in st.session_state.chat_history:
        emotion = msg['emotion']
        if emotion not in emotion_confidences:
            emotion_confidences[emotion] = []
        emotion_confidences[emotion].append(msg['confidence'])
    
    avg_confidences = {
        emotion: sum(confs) / len(confs)
        for emotion, confs in emotion_confidences.items()
    }
    
    df_conf = pd.DataFrame.from_dict(
        avg_confidences, orient='index', columns=['Average Confidence']
    )
    df_conf.index.name = 'Emotion'
    
    st.dataframe(df_conf.style.format("{:.1%}"), use_container_width=True)
    
    # Recent messages
    st.subheader("📝 Recent Messages")
    
    for msg in reversed(st.session_state.chat_history[-5:]):
        st.text(
            f"[{msg['timestamp']}] {msg['emotion'].upper()}: {msg['message']}"
        )

if __name__ == "__main__":
    main()
