"""Chat interface component for Streamlit"""

import streamlit as st
from datetime import datetime

class ChatInterfaceComponent:
    """Reusable chat interface component"""
    
    @staticmethod
    def render_message(message: str, emotion: str, confidence: float, timestamp: str, style: dict):
        """Render a single chat message"""
        
        st.markdown(
            f"""
            <div style="
                background-color: {style['background_color']};
                border-left: 4px solid {style['border_color']};
                padding: 12px;
                margin: 8px 0;
                border-radius: 8px;
            ">
                <small style="color: #999;">{timestamp}</small><br>
                <span style="font-size: 1.3em;">{style['pattern']}</span>
                <span style="font-size: 1.1em;">{message}</span><br>
                <small style="color: #666;">
                    {emotion.title()} • {confidence:.0%} confidence
                </small>
            </div>
            """,
            unsafe_allow_html=True
        )
