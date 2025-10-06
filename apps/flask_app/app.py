from flask import Flask, request, jsonify
from nlp_emotion import detector

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "EchoBridge Emotion API",
        "status": "online",
        "endpoints": {
            "analyze": "/api/emotion/analyze (POST)",
            "health": "/api/emotion/health (GET)"
        }
    }), 200

@app.route('/api/emotion/analyze', methods=['POST'])
def analyze_emotion():
    """Analyze emotion from text input"""
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' field"}), 400
    
    result = detector.analyze(data['text'])
    
    if 'error' in result:
        return jsonify(result), 400
    
    return jsonify(result), 200

@app.route('/api/emotion/health', methods=['GET'])
def emotion_health():
    """Check if emotion detection service is running"""
    return jsonify({
        "status": "healthy", 
        "service": "emotion_detection"
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
