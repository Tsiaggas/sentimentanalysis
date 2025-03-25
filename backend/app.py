import os
import re
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from sentiment_analyzer import SentimentAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize sentiment analyzer
analyzer = SentimentAnalyzer()
logger.info("Sentiment analyzer initialized")

@app.route('/')
def home():
    logger.info("Home route accessed")
    return jsonify({
        'status': 'API is running',
        'message': 'Welcome to the Sentiment Analysis API',
        'endpoints': {
            '/api/health': 'Health check endpoint',
            '/api/analyze': 'Analyze sentiment for a single text (POST)',
            '/api/batch-analyze': 'Analyze sentiment for multiple texts (POST)'
        }
    })
    
@app.route('/api/health')
def health():
    logger.info("Health check endpoint accessed")
    return jsonify({
        'status': 'ok',
        'message': 'API is healthy'
    })

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    
    if not data or 'text' not in data:
        logger.error("Invalid request: missing 'text' field")
        return jsonify({'error': 'Missing text field'}), 400
        
    text = data['text']
    logger.info(f"Analyzing text: {text[:50]}...")
    
    try:
        result = analyzer.analyze_sentiment(text)
        logger.info(f"Analysis complete. Sentiment keys: {list(result['sentiment'].keys())}")
        return jsonify(result)
    except Exception as e:
        logger.exception(f"Error during analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    data = request.get_json()
    
    if not data or 'texts' not in data:
        logger.error("Invalid request: missing 'texts' field")
        return jsonify({'error': 'Missing texts field'}), 400
        
    texts = data['texts']
    if not isinstance(texts, list):
        logger.error("Invalid request: 'texts' field is not a list")
        return jsonify({'error': 'Texts field must be a list'}), 400
        
    logger.info(f"Batch analyzing {len(texts)} texts")
    
    try:
        results = analyzer.batch_analyze(texts)
        return jsonify(results)
    except Exception as e:
        logger.exception(f"Error during batch analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Get port from environment variable or use 5000 as default
    port = int(os.environ.get('PORT', 5000))
    # Use 0.0.0.0 to make the server publicly available
    debug = os.environ.get("DEBUG", "False").lower() in ('true', '1', 't')
    logger.info(f"Starting app on port {port}, debug={debug}")
    app.run(host='0.0.0.0', port=port, debug=debug) 