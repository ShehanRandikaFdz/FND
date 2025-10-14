"""
Minimal Flask app to test JSON serialization
"""

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/test', methods=['POST'])
def test():
    """Simple test endpoint"""
    try:
        data = request.get_json()
        text = data.get('text', '') if data else ''
        
        response = {
            'prediction': 'TRUE',
            'confidence': 75.0,
            'text': text,
            'message': 'Test successful'
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting minimal test app...")
    app.run(debug=True, host='0.0.0.0', port=5001)
