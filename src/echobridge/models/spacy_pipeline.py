"""
spaCy pipeline integration for emotion detection
Combines BERT model with spaCy NLP processing
"""

import spacy
from spacy.language import Language
from spacy.tokens import Doc
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from echobridge.models.bert_classifier import BertEmotionClassifier
from echobridge.config import SPACY_MODEL_NAME

# Register custom extension attributes
if not Doc.has_extension("emotion"):
    Doc.set_extension("emotion", default="neutral")
if not Doc.has_extension("emotion_confidence"):
    Doc.set_extension("emotion_confidence", default=0.0)
if not Doc.has_extension("emotion_id"):
    Doc.set_extension("emotion_id", default=0)

@Language.component("emotion_classifier")
def emotion_classifier_component(doc):
    """
    Custom spaCy component for emotion detection
    Uses BERT classifier to detect emotions in text
    """
    # Get BERT classifier from nlp pipeline config
    if not hasattr(doc._, "bert_classifier"):
        return doc
    
    bert_classifier = doc._.bert_classifier
    
    # Predict emotion for full text
    emotion_name, confidence = bert_classifier.predict_emotion(doc.text)
    
    # Set custom attributes
    doc._.emotion = emotion_name
    doc._.emotion_confidence = float(confidence)
    
    return doc

class EmotionSpacyPipeline:
    """Integrated spaCy pipeline with emotion detection"""
    
    def __init__(self, bert_classifier: BertEmotionClassifier, spacy_model: str = SPACY_MODEL_NAME):
        """
        Initialize spaCy pipeline with emotion detection
        
        Args:
            bert_classifier: Pre-trained BERT emotion classifier
            spacy_model: spaCy model name to load
        """
        print(f"🔧 Initializing spaCy pipeline with {spacy_model}...")
        
        # Load spaCy model
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"⚠️ Model {spacy_model} not found. Downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", spacy_model])
            self.nlp = spacy.load(spacy_model)
        
        # Store BERT classifier
        self.bert_classifier = bert_classifier
        
        # Add emotion classifier component to pipeline
        if "emotion_classifier" not in self.nlp.pipe_names:
            self.nlp.add_pipe("emotion_classifier", last=True)
        
        print(f"✓ spaCy pipeline ready with components: {self.nlp.pipe_names}")
    
    def process_text(self, text: str) -> Doc:
        """
        Process text through complete NLP pipeline
        
        Args:
            text: Input text string
            
        Returns:
            spaCy Doc object with emotion attributes
        """
        doc = self.nlp(text)
        
        # Inject BERT classifier into doc
        doc._.bert_classifier = self.bert_classifier
        
        # Process through emotion classifier component
        doc = emotion_classifier_component(doc)
        
        return doc
    
    def batch_process(self, texts: list) -> list:
        """
        Process multiple texts efficiently
        
        Args:
            texts: List of text strings
            
        Returns:
            List of dictionaries with text and emotion data
        """
        results = []
        
        for doc in self.nlp.pipe(texts):
            # Inject BERT classifier
            doc._.bert_classifier = self.bert_classifier
            doc = emotion_classifier_component(doc)
            
            results.append({
                'text': doc.text,
                'emotion': doc._.emotion,
                'confidence': doc._.emotion_confidence,
                'entities': [(ent.text, ent.label_) for ent in doc.ents],
                'tokens': [token.text for token in doc],
                'pos_tags': [(token.text, token.pos_) for token in doc]
            })
        
        return results
    
    def analyze_text(self, text: str) -> dict:
        """
        Comprehensive text analysis with emotion detection
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with full analysis results
        """
        doc = self.process_text(text)
        
        return {
            'text': doc.text,
            'emotion': doc._.emotion,
            'confidence': doc._.emotion_confidence,
            'tokens': [token.text for token in doc],
            'lemmas': [token.lemma_ for token in doc],
            'pos_tags': [(token.text, token.pos_) for token in doc],
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'noun_chunks': [chunk.text for chunk in doc.noun_chunks],
            'sentences': [sent.text for sent in doc.sents]
        }
