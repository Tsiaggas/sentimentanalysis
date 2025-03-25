from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import sys
import traceback

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our sentiment analyzer
try:
    from sentiment_analyzer import SentimentAnalyzer
except ImportError as e:
    print(f"Error importing SentimentAnalyzer: {e}")
    traceback.print_exc()
    # Create a minimal analyzer to not crash the app
    class SentimentAnalyzer:
        def analyze_sentiment(self, text):
            return {
                'text': text,
                'overall_sentiment': 'neutral',
                'sentiment': {'vader': 0, 'textblob': 0, 'bert': 0},
                'emotions': {
                    'joy': 0.3,
                    'trust': 0.4,
                    'pleasure': 0.2,
                    'anxiety': 0.1,
                    'anger': 0.1,
                    'sadness': 0.1
                },
                'confidence': 0.5
            }
        def batch_analyze(self, texts):
            return [self.analyze_sentiment(text) for text in texts]

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Basic configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev')
app.config['MONGODB_URI'] = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/sentiment_analysis')

# Initialize sentiment analyzer
analyzer = SentimentAnalyzer()

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/api/analyze', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.get_json()
        text = data.get('text')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
            
        # Use our sentiment analyzer
        result = analyzer.analyze_sentiment(text)
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Error in analyze_sentiment: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({"error": "No texts provided"}), 400
            
        # Use our batch analyzer
        results = analyzer.batch_analyze(texts)
            
        return jsonify({"results": results}), 200
        
    except Exception as e:
        print(f"Error in batch_analyze: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 