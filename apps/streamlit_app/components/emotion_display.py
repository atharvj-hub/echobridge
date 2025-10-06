"""Emotion display component for Streamlit"""

import streamlit as st
from typing import Dict

class EmotionDisplayComponent:
    """Reusable emotion display component"""
    
    @staticmethod
    def render_emotion_card(emotion: str, confidence: float, style: Dict):
        """Render an emotion card with styling"""
        
        st.markdown(
            f"""
            <div style="
                background-color: {style['background_color']};
                border-left: 4px solid {style['border_color']};
                padding: 1rem;
                border-radius: 8px;
                margin: 0.5rem 0;
            ">
                <div style="font-size: 2rem;">{style['pattern']}</div>
                <div style="font-size: 1.5rem; font-weight: bold;">{emotion.title()}</div>
                <div style="font-size: 1rem; color: #666;">
                    Confidence: {confidence:.1%}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    @staticmethod
    def render_emotion_legend(color_mapper):
        """Render emotion legend"""
        
        st.subheader("🎨 Emotion Color Guide")
        
        for emotion in color_mapper.get_all_emotions():
            style = color_mapper.get_emotion_style(emotion, 1.0)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                st.markdown(
                    f'<div style="font-size: 2rem; text-align: center;">'
                    f'{style["pattern"]}</div>',
                    unsafe_allow_html=True
                )
            
            with col2:
                st.write(f"**{emotion.title()}**")
                st.write(f"Color: {style['color_name']}")
            
            with col3:
                st.markdown(
                    f'<div style="background-color: {style["color"]}; '
                    f'width: 50px; height: 50px; border-radius: 4px;"></div>',
                    unsafe_allow_html=True
                )
