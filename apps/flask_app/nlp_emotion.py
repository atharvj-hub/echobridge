from transformers import pipeline

class EmotionDetector:
    def __init__(self):
        print("Loading emotion detection model...")
        self.pipeline = pipeline(
            'sentiment-analysis',
            model='bhadresh-savani/distilbert-base-uncased-emotion'
        )
        print("Model loaded successfully!")
    
    def analyze(self, text):
        if not text or not text.strip():
            return {"error": "Text cannot be empty"}
        
        try:
            result = self.pipeline(text)[0]
            return {
                "emotion": result['label'],
                "confidence": round(result['score'], 3),
                "text": text
            }
        except Exception as e:
            return {"error": str(e)}

# Initialize detector once
detector = EmotionDetector()
