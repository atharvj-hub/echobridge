from flask import Flask, request, jsonify
# ... your other existing imports ...

from nlp_emotion import detector  # NEW import

app = Flask(__name__)

# ... your existing configuration and routes ...

# === NEW EMOTION DETECTION ROUTES ===

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

# === END OF NEW ROUTES ===

# ... rest of your existing code ...

if __name__ == '__main__':
    app.run(debug=True)  # or however you run your 