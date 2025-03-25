import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline
import re
import string

class SentimentAnalyzer:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
            
        # Initialize sentiment analyzers
        self.vader = SentimentIntensityAnalyzer()
        self.bert_sentiment = pipeline("sentiment-analysis")
        
        # Define emotion keywords
        self.emotion_keywords = {
            'joy': ['happy', 'excited', 'delighted', 'joyful', 'wonderful'],
            'trust': ['trust', 'reliable', 'dependable', 'honest', 'sincere'],
            'pleasure': ['pleasure', 'enjoy', 'satisfied', 'content', 'gratified'],
            'anxiety': ['anxious', 'worried', 'concerned', 'nervous', 'uneasy'],
            'anger': ['angry', 'furious', 'outraged', 'irritated', 'annoyed'],
            'sadness': ['sad', 'depressed', 'unhappy', 'gloomy', 'miserable']
        }
        
    def preprocess_text(self, text):
        """Preprocess the input text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
        
    def analyze_emotions(self, text):
        """Analyze emotions in the text using keyword matching."""
        text = self.preprocess_text(text)
        emotions = {}
        
        for emotion, keywords in self.emotion_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text)
            emotions[emotion] = count / len(keywords)  # Normalize to 0-1 range
            
        return emotions
        
    def analyze_sentiment(self, text):
        """Perform comprehensive sentiment analysis using multiple models."""
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # VADER sentiment analysis
        vader_scores = self.vader.polarity_scores(processed_text)
        
        # TextBlob sentiment analysis
        blob = TextBlob(processed_text)
        textblob_sentiment = blob.sentiment.polarity
        
        # BERT sentiment analysis
        bert_result = self.bert_sentiment(processed_text)[0]
        
        # Emotion analysis
        emotions = self.analyze_emotions(processed_text)
        
        # Combine results
        result = {
            'text': text,
            'sentiment': {
                'vader': vader_scores['compound'],
                'textblob': textblob_sentiment,
                'bert': float(bert_result['score']) if bert_result['label'] == 'POSITIVE' else -float(bert_result['score'])
            },
            'emotions': emotions,
            'confidence': (abs(vader_scores['compound']) + abs(textblob_sentiment) + abs(float(bert_result['score']))) / 3
        }
        
        # Determine overall sentiment
        avg_sentiment = (vader_scores['compound'] + textblob_sentiment + 
                       (float(bert_result['score']) if bert_result['label'] == 'POSITIVE' else -float(bert_result['score']))) / 3
        
        if avg_sentiment > 0.2:
            result['overall_sentiment'] = 'positive'
        elif avg_sentiment < -0.2:
            result['overall_sentiment'] = 'negative'
        else:
            result['overall_sentiment'] = 'neutral'
            
        return result
        
    def batch_analyze(self, texts):
        """Analyze multiple texts."""
        return [self.analyze_sentiment(text) for text in texts] 