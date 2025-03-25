import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import re
import string
import os
import json
import requests
from langdetect import detect
import spacy
import logging
from textblob import TextBlob
try:
    import el_core_news_sm
except ImportError:
    os.system('python -m spacy download el_core_news_sm')
    import el_core_news_sm

# Ρύθμιση καταγραφής με πιο λεπτομερή τρόπο
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sentiment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        logger.info("Initializing SentimentAnalyzer...")
        # Download required NLTK data
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
            
        print("Initializing sentiment analyzers...")
        
        # Initialize basic sentiment analyzers
        self.vader = SentimentIntensityAnalyzer()
        
        # Initialize multilingual BERT model for sentiment analysis
        model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        try:
            self.multilingual_model = pipeline("sentiment-analysis", model=model_name)
        except Exception as e:
            print(f"Error loading multilingual model: {e}")
            # Fallback to English model
            self.multilingual_model = pipeline("sentiment-analysis")
        
        # Initialize English-specific model
        self.english_model = pipeline("sentiment-analysis")
        
        # Hugging Face API key - Read from environment variable
        self.hf_api_key = os.environ.get("HUGGING_FACE_API_KEY", "")
        print(f"Environment variable HUGGING_FACE_API_KEY: {'Found and not empty' if self.hf_api_key else 'Missing or empty'}")
        if self.hf_api_key:
            print(f"API key first 4 chars: {self.hf_api_key[:4]}...")
        
        self.use_remote_deepseek = bool(self.hf_api_key)
        if self.use_remote_deepseek:
            print("Hugging Face API key found - DeepSeek API integration enabled")
        else:
            print("No Hugging Face API key found - DeepSeek API integration disabled")
        
        # Load Greek spaCy model
        self.greek_nlp = None
        try:
            self.greek_nlp = spacy.load('el_core_news_sm')
            logger.info("Greek NLP model successfully loaded")
        except OSError:
            logger.warning("Greek NLP model not found. Will download it.")
            # Προσπαθούμε να κατεβάσουμε το μοντέλο
            try:
                import subprocess
                subprocess.run(['python', '-m', 'spacy', 'download', 'el_core_news_sm'], check=True)
                self.greek_nlp = spacy.load('el_core_news_sm')
                logger.info("Greek NLP model downloaded and loaded")
            except Exception as e:
                logger.error(f"Failed to download Greek NLP model: {e}")
        
        # Greek sentiment lexicon (enhanced)
        self.greek_sentiment_lexicon = {
            # Positive words
            'καλός': 1, 'τέλειος': 1.5, 'εξαιρετικός': 1.8, 'υπέροχος': 1.8, 'φανταστικός': 1.7,
            'ωραίος': 0.8, 'ευχάριστος': 0.7, 'χαρούμενος': 1.2, 'εντυπωσιακός': 1.3, 'αγαπημένος': 1.1,
            'άριστος': 1.5, 'καταπληκτικός': 1.6, 'έξοχος': 1.4, 'αξιόλογος': 0.9, 'αξιοθαύμαστος': 1.3,
            'θετικός': 0.8, 'επιτυχημένος': 1.0, 'άψογος': 1.4, 'ικανοποιητικός': 0.7, 'ποιοτικός': 0.9,
            
            # New positive words
            'αγαπώ': 1.5, 'λατρεύω': 1.8, 'απολαμβάνω': 1.2, 'χαίρομαι': 1.3, 'ενθουσιάζομαι': 1.6,
            'εκτιμώ': 0.9, 'θαυμάζω': 1.2, 'προτιμώ': 0.6, 'ευχαριστώ': 1.0, 'επιδοκιμάζω': 0.8,
            'ευτυχής': 1.4, 'γενναιόδωρος': 0.8, 'έξυπνος': 0.7, 'δυνατός': 0.6, 'όμορφος': 0.8,
            'εντυπωσιακά': 1.2, 'άνετα': 0.7, 'αξιοθαύμαστα': 1.1, 'έξοχα': 1.3, 'τέλεια': 1.5,
            'βολικός': 0.7, 'γρήγορος': 0.6, 'αποτελεσματικός': 0.8, 'φιλικός': 0.9, 'χρήσιμος': 0.8,
            'αξιόπιστος': 1.0, 'ενδιαφέρων': 0.7, 'εύκολος': 0.6, 'ευέλικτος': 0.7, 'προσιτός': 0.8,
            
            # Negative words
            'κακός': -1, 'απαίσιος': -1.5, 'φρικτός': -1.8, 'χάλια': -1.7, 'απογοητευτικός': -1.3,
            'άσχημος': -0.8, 'δυσάρεστος': -0.7, 'λυπημένος': -1.2, 'τρομερός': -1.4, 'αηδιαστικός': -1.6,
            'άθλιος': -1.5, 'αποτυχημένος': -1.0, 'ανεπαρκής': -1.1, 'προβληματικός': -0.9, 'κατώτερος': -0.8,
            'αρνητικός': -0.8, 'αποτυχία': -1.2, 'ανεπιθύμητος': -1.0, 'ελαττωματικός': -1.1, 'μέτριος': -0.5,
            
            # New negative words
            'μισώ': -1.5, 'απεχθάνομαι': -1.7, 'ενοχλώ': -0.8, 'θυμώνω': -1.1, 'απογοητεύομαι': -1.2,
            'φοβάμαι': -0.9, 'ανησυχώ': -0.7, 'πενθώ': -1.3, 'κατακρίνω': -0.8, 'αποδοκιμάζω': -0.9,
            'δυστυχισμένος': -1.4, 'επιθετικός': -0.8, 'ανόητος': -0.7, 'αδύναμος': -0.6, 'άσχημος': -0.8,
            'απαράδεκτα': -1.2, 'δύσκολα': -0.7, 'απογοητευτικά': -1.1, 'φρικτά': -1.5, 'χειρότερα': -1.3,
            'άβολος': -0.7, 'αργός': -0.6, 'αναποτελεσματικός': -0.8, 'εχθρικός': -0.9, 'άχρηστος': -1.0,
            'αναξιόπιστος': -1.0, 'βαρετός': -0.7, 'δύσκολος': -0.6, 'δύσχρηστος': -0.8, 'ακριβός': -0.7,
            
            # Common modifiers
            'πολύ': 0.3, 'αρκετά': 0.2, 'λίγο': 0.1, 'εξαιρετικά': 0.4, 'απίστευτα': 0.4,
            'καθόλου': -0.3, 'σχεδόν': 0.1, 'απόλυτα': 0.3, 'σίγουρα': 0.2, 'μάλλον': 0.1
        }
        
        # Define emotion keywords (English)
        self.emotion_keywords_en = {
            'joy': ['happy', 'excited', 'delighted', 'joyful', 'wonderful', 'great', 'pleased', 'glad'],
            'trust': ['trust', 'reliable', 'dependable', 'honest', 'sincere', 'faithful', 'confident'],
            'pleasure': ['pleasure', 'enjoy', 'satisfied', 'content', 'gratified', 'pleased', 'delighted'],
            'anxiety': ['anxious', 'worried', 'concerned', 'nervous', 'uneasy', 'tense', 'stressed'],
            'anger': ['angry', 'furious', 'outraged', 'irritated', 'annoyed', 'mad', 'frustrated'],
            'sadness': ['sad', 'depressed', 'unhappy', 'gloomy', 'miserable', 'down', 'heartbroken']
        }
        
        # Define emotion keywords (Greek)
        self.emotion_keywords_gr = {
            'joy': ['χαρούμενος', 'ενθουσιασμένος', 'χαρά', 'ευτυχισμένος', 'υπέροχος', 'τέλειος', 'ικανοποιημένος'],
            'trust': ['εμπιστοσύνη', 'αξιόπιστος', 'ειλικρινής', 'έντιμος', 'πιστός', 'βέβαιος', 'σίγουρος'],
            'pleasure': ['απόλαυση', 'ευχαρίστηση', 'ικανοποίηση', 'ευχάριστος', 'απολαυστικός', 'ηδονικός'],
            'anxiety': ['άγχος', 'ανησυχία', 'αγχωμένος', 'ανήσυχος', 'νευρικός', 'αγωνία', 'φοβισμένος'],
            'anger': ['θυμός', 'οργή', 'θυμωμένος', 'εξοργισμένος', 'ενοχλημένος', 'εκνευρισμένος', 'αγανακτισμένος'],
            'sadness': ['λύπη', 'θλίψη', 'στεναχωρημένος', 'λυπημένος', 'απογοητευμένος', 'μελαγχολικός', 'δυστυχισμένος']
        }
        
        print("Sentiment analyzer initialization complete")
        
    def detect_language(self, text):
        """Detect the language of the text."""
        try:
            # Έλεγχος αν υπάρχουν ελληνικοί χαρακτήρες
            greek_chars = set('αβγδεζηθικλμνξοπρστυφχψωςάέήίόύώΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩΆΈΉΊΌΎΏ')
            text_chars = set(text.lower())
            
            # Αν το ποσοστό ελληνικών χαρακτήρων είναι αρκετά υψηλό
            greek_char_count = sum(1 for c in text.lower() if c in greek_chars)
            if greek_char_count > len(text) * 0.3:  # Αν πάνω από 30% είναι ελληνικοί χαρακτήρες
                print(f"Detected Greek language based on character set (Greek chars: {greek_char_count}/{len(text)})")
                return 'el'
                
            # Fallback to langdetect
            detected = detect(text)
            print(f"Language detected by langdetect: {detected}")
            return detected
        except Exception as e:
            print(f"Language detection error: {e}")
            # Default to English if detection fails
            return 'en'
            
    def preprocess_text(self, text, language='en'):
        """Preprocess the input text based on language."""
        # Convert to lowercase
        text = text.lower()
        
        if language == 'el':
            # For Greek text
            if self.greek_nlp:
                doc = self.greek_nlp(text)
                # Keep lemmatized tokens, excluding punctuation
                processed_tokens = [token.lemma_ for token in doc if not token.is_punct]
                return ' '.join(processed_tokens)
            else:
                # Basic preprocessing if spaCy model isn't available
                # Remove special characters and numbers, keep Greek letters
                text = re.sub(r'[^α-ωΑ-Ωάέήίόύώϊϋΐΰ\s]', '', text)
        else:
            # For English text
            # Remove special characters and numbers
            text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
        
    def analyze_greek_sentiment(self, text):
        """Analyze sentiment in Greek text using lexicon-based approach."""
        processed_text = self.preprocess_text(text, 'el')
        
        if self.greek_nlp:
            doc = self.greek_nlp(processed_text)
            words = [token.lemma_ for token in doc]
        else:
            words = processed_text.split()
            
        # Calculate sentiment score using lexicon
        total_score = 0
        found_words = 0
        
        for word in words:
            if word in self.greek_sentiment_lexicon:
                total_score += self.greek_sentiment_lexicon[word]
                found_words += 1
                
        # Normalize score to range similar to VADER (-1 to 1)
        sentiment_score = 0
        if found_words > 0:
            sentiment_score = total_score / found_words
            
        # Scale to match VADER compound score
        scaled_score = max(min(sentiment_score, 1.0), -1.0)
        
        return {
            'compound': scaled_score,
            'pos': max(0, scaled_score),
            'neg': max(0, -scaled_score),
            'neu': 1.0 - abs(scaled_score)
        }
        
    def analyze_emotions(self, text, language='en'):
        """Analyze emotions in the text using keyword matching based on language."""
        text = self.preprocess_text(text, language)
        emotions = {}
        
        # Select appropriate emotion keywords dictionary
        emotion_keywords = self.emotion_keywords_en if language == 'en' else self.emotion_keywords_gr
        
        for emotion, keywords in emotion_keywords.items():
            # Count occurrences of keywords for this emotion
            count = sum(1 for keyword in keywords if keyword in text)
            weight = count / len(keywords)  # Normalize to 0-1 range
            emotions[emotion] = min(weight, 1.0)  # Cap at 1.0
            
        return emotions
        
    def query_deepseek_api(self, text, language='en'):
        """Query DeepSeek model via Hugging Face Inference API."""
        if not self.use_remote_deepseek:
            print("DeepSeek API integration is disabled - no API key found")
            return {"score": 0, "error": "DeepSeek API integration disabled"}
            
        print(f"Calling DeepSeek API with API key: {self.hf_api_key[:4]}...")
        
        # Different prompts based on language
        if language == 'el':
            prompt = f"""Είσαι ένας εμπειρογνώμονας ανάλυσης συναισθημάτων. Δοσμένου ενός κειμένου στα ελληνικά, ανάλυσε το συναίσθημα και δώσε μια τιμή από -1 (πολύ αρνητικό) έως 1 (πολύ θετικό).

Κείμενο: "{text}"

Αξιολόγηση συναισθήματος (μόνο μία αριθμητική τιμή από -1 έως 1):"""
        else:
            prompt = f"""You are a sentiment analysis expert. Given a text in English, analyze the sentiment and provide a score from -1 (very negative) to 1 (very positive).

Text: "{text}"

Sentiment score (only a single numeric value from -1 to 1):"""
            
        print(f"Using prompt: {prompt[:50]}...")
            
        try:
            # Καταγραφή πριν την αποστολή του αιτήματος
            logger.info(f"Sending request to Hugging Face API for DeepSeek model")
            logger.debug(f"API URL: {API_URL}")
            logger.debug(f"Headers: {headers}")
            
            # Αποστολή αιτήματος
            API_URL = "https://api-inference.huggingface.co/models/deepseek-ai/deepseek-coder-6.7b-instruct"
            headers = {"Authorization": f"Bearer {self.hf_api_key}"}
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 20,
                    "temperature": 0.1,
                    "return_full_text": False
                }
            }
            
            print(f"Sending request to HF API: {API_URL}")
            
            response = requests.post(API_URL, headers=headers, json=payload)
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                # Extract generated text from response
                result = response.json()
                print(f"API Response: {result}")
                
                generated_text = result[0].get("generated_text", "")
                print(f"Generated text: {generated_text}")
                
                # Try to extract the score from the response
                try:
                    # Extract just the first number that appears in the output
                    score_match = re.search(r'(-?\d+(\.\d+)?)', generated_text)
                    if score_match:
                        score = float(score_match.group(1))
                        # Ensure the score is within the valid range
                        score = max(-1, min(1, score))
                        print(f"Extracted score: {score}")
                    else:
                        # Fallback - determine if text seems positive or negative
                        if any(word in generated_text.lower() for word in ['positive', 'θετικό']):
                            score = 0.7
                            print("No score found, fallback to positive: 0.7")
                        elif any(word in generated_text.lower() for word in ['negative', 'αρνητικό']):
                            score = -0.7
                            print("No score found, fallback to negative: -0.7")
                except:
                    score = 0  # Default to neutral if parsing fails
                
                return {"score": score}
            else:
                error_msg = f"API request failed with status code {response.status_code}"
                print(error_msg)
                if response.status_code == 401:
                    print("Hugging Face API error: Unauthorized. Check your API key.")
                elif response.status_code == 503:
                    print("Hugging Face API error: Service unavailable. The model might be loading or unavailable.")
                print(f"Response content: {response.text}")
                return {"score": 0, "error": error_msg}
                
        except Exception as e:
            error_msg = f"Error querying DeepSeek API: {e}"
            print(error_msg)
            import traceback
            print(f"Exception traceback: {traceback.format_exc()}")
            return {"score": 0, "error": str(e)}

    def analyze_sentiment(self, text):
        """Perform comprehensive sentiment analysis using multiple models and language detection."""
        # Detect language
        language = self.detect_language(text)
        print(f"Detected language: {language}")
        
        # Preprocess text
        processed_text = self.preprocess_text(text, language)
        
        # Initialize result structure
        result = {
            'text': text,
            'language': language,
            'overall_sentiment': 'neutral',
            'sentiment': {},
            'emotions': {},
            'confidence': 0.5
        }
        
        try:
            # Apply sentiment analysis based on language
            if language == 'el':
                # Greek text analysis
                print("Performing Greek sentiment analysis")
                
                # Use lexicon-based approach for Greek
                greek_scores = self.analyze_greek_sentiment(processed_text)
                
                # Try DeepSeek API for Greek if enabled
                deepseek_result = {"score": 0}
                if self.use_remote_deepseek:
                    print("Using DeepSeek API for Greek sentiment analysis")
                    deepseek_result = self.query_deepseek_api(text, 'el')
                
                # Try multilingual BERT model
                try:
                    multi_result = self.multilingual_model(processed_text)[0]
                    multi_score = float(multi_result['score'])
                    multi_label = multi_result['label']
                    
                    # Parse BERT result correctly - similar to English version
                    # Convert BERT score to -1 to 1 scale for multilingual model
                    if 'LABEL_' in multi_label:
                        # Extract the label number (e.g., "LABEL_4" -> 4)
                        try:
                            label_num = int(multi_label.split('_')[1])
                            # Convert to -1 to 1 scale (assuming LABEL_0 to LABEL_4)
                            multi_score = (label_num / 2) - 1  # 0->-1, 2->0, 4->1
                        except:
                            # Keep original score but adjust if negative
                            if 'NEGATIVE' in multi_label:
                                multi_score = -multi_score
                    elif multi_label.isdigit() or (multi_label[0].isdigit() and ' star' in multi_label):
                        # Handle "5 star" or "5" format
                        try:
                            stars = int(multi_label.split(' ')[0])
                            # Convert 1-5 scale to -1 to 1
                            multi_score = ((stars - 3) / 2)  # 1->-1, 3->0, 5->1
                        except:
                            # Keep original score but adjust if negative
                            if 'NEGATIVE' in multi_label:
                                multi_score = -multi_score
                    
                    # Add normal sentiment scores
                    result['sentiment'] = {
                        'lexicon': greek_scores['compound'],
                        'multilingual': multi_score
                    }
                    
                    # Add DeepSeek score if available
                    if self.use_remote_deepseek and 'error' not in deepseek_result:
                        result['sentiment']['deepseek'] = deepseek_result['score']
                        print(f"Added DeepSeek score to sentiment results: {deepseek_result['score']}")
                    else:
                        print(f"DeepSeek score not added to results. use_remote_deepseek={self.use_remote_deepseek}, error={'error' in deepseek_result}")
                    
                    # Calculate average sentiment score
                    if self.use_remote_deepseek and 'error' not in deepseek_result:
                        # DeepSeek available - give it higher weight
                        weights = {"lexicon": 0.25, "multilingual": 0.25, "deepseek": 0.5}
                        avg_sentiment = (
                            weights["lexicon"] * greek_scores['compound'] + 
                            weights["multilingual"] * multi_score + 
                            weights["deepseek"] * deepseek_result['score']
                        )
                        print(f"Using weighted average with DeepSeek (w=0.5): {avg_sentiment}")
                    else:
                        # No DeepSeek - use plain average
                        avg_sentiment = (greek_scores['compound'] + multi_score) / 2
                        print(f"Using plain average without DeepSeek: {avg_sentiment}")
                    
                except Exception as e:
                    print(f"Error in multilingual model: {e}")
                    
                    if self.use_remote_deepseek and 'error' not in deepseek_result:
                        # Use lexicon and DeepSeek if multilingual failed
                        result['sentiment'] = {
                            'lexicon': greek_scores['compound'],
                            'deepseek': deepseek_result['score']
                        }
                        avg_sentiment = (greek_scores['compound'] + deepseek_result['score']) / 2
                    else:
                        # Use only lexicon if both multilingual and DeepSeek failed
                        result['sentiment'] = {
                            'lexicon': greek_scores['compound']
                        }
                        avg_sentiment = greek_scores['compound']
                
                # Analyze emotions in Greek
                result['emotions'] = self.analyze_emotions(processed_text, 'el')
                
            else:
                # English (or other language) analysis - fallback to our existing methods
                print("Performing English sentiment analysis")
                
                # VADER sentiment analysis
                vader_scores = self.vader.polarity_scores(processed_text)
                
                # BERT sentiment analysis
                bert_result = self.english_model(processed_text)[0]
                
                # Parse BERT result correctly
                bert_score = float(bert_result['score'])
                bert_label = bert_result['label']
                
                # Convert BERT score to -1 to 1 scale
                # Most BERT models use labels like "LABEL_0" through "LABEL_4" or "1 star" through "5 stars"
                # Check if the label contains a number
                if 'LABEL_' in bert_label:
                    # Extract the label number (e.g., "LABEL_4" -> 4)
                    try:
                        label_num = int(bert_label.split('_')[1])
                        # Convert to -1 to 1 scale (assuming LABEL_0 to LABEL_4)
                        normalized_bert_score = (label_num / 2) - 1  # 0->-1, 2->0, 4->1
                    except:
                        # Fallback to original logic
                        normalized_bert_score = bert_score if bert_label == 'POSITIVE' else -bert_score
                elif bert_label.isdigit() or (bert_label[0].isdigit() and ' star' in bert_label):
                    # Handle "5 star" or "5" format
                    try:
                        stars = int(bert_label.split(' ')[0])
                        # Convert 1-5 scale to -1 to 1
                        normalized_bert_score = ((stars - 3) / 2)  # 1->-1, 3->0, 5->1
                    except:
                        normalized_bert_score = bert_score if bert_label == 'POSITIVE' else -bert_score
                elif bert_label == 'POSITIVE':
                    normalized_bert_score = bert_score
                elif bert_label == 'NEGATIVE':
                    normalized_bert_score = -bert_score
                else:
                    # For any other format, try to determine if it's positive or negative
                    positive_keywords = ['positive', 'good', 'great', 'excellent']
                    negative_keywords = ['negative', 'bad', 'poor', 'terrible']
                    
                    if any(keyword in bert_label.lower() for keyword in positive_keywords):
                        normalized_bert_score = bert_score
                    elif any(keyword in bert_label.lower() for keyword in negative_keywords):
                        normalized_bert_score = -bert_score
                    else:
                        # If we can't determine, assume neutral
                        normalized_bert_score = 0
                
                # Try DeepSeek API for English if enabled
                deepseek_result = {"score": 0}
                if self.use_remote_deepseek:
                    print("Using DeepSeek API for English sentiment analysis")
                    deepseek_result = self.query_deepseek_api(text, 'en')
                
                # Add normal sentiment scores
                result['sentiment'] = {
                    'vader': vader_scores['compound'],
                    'bert': normalized_bert_score
                }
                
                # Add DeepSeek score if available
                if self.use_remote_deepseek and 'error' not in deepseek_result:
                    result['sentiment']['deepseek'] = deepseek_result['score']
                    print(f"Added DeepSeek score to sentiment results: {deepseek_result['score']}")
                else:
                    print(f"DeepSeek score not added to results. use_remote_deepseek={self.use_remote_deepseek}, error={'error' in deepseek_result}")
                
                # Calculate average sentiment score
                if self.use_remote_deepseek and 'error' not in deepseek_result:
                    # DeepSeek available - give it higher weight
                    weights = {"vader": 0.25, "bert": 0.25, "deepseek": 0.5}
                    avg_sentiment = (
                        weights["vader"] * vader_scores['compound'] + 
                        weights["bert"] * normalized_bert_score + 
                        weights["deepseek"] * deepseek_result['score']
                    )
                    print(f"Using weighted average with DeepSeek (w=0.5): {avg_sentiment}")
                else:
                    # No DeepSeek - use plain average
                    avg_sentiment = (vader_scores['compound'] + normalized_bert_score) / 2
                    print(f"Using plain average without DeepSeek: {avg_sentiment}")
                
                # Analyze emotions in English
                result['emotions'] = self.analyze_emotions(processed_text, 'en')
            
            # Determine overall sentiment and confidence
            if avg_sentiment > 0.2:
                result['overall_sentiment'] = 'positive'
            elif avg_sentiment < -0.2:
                result['overall_sentiment'] = 'negative'
            else:
                result['overall_sentiment'] = 'neutral'
                
            # Calculate confidence as the absolute value of sentiment
            result['confidence'] = min(abs(avg_sentiment) + 0.5, 1.0)
            
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            # Provide basic fallback results
            result['overall_sentiment'] = 'neutral'
            result['confidence'] = 0.5
            result['sentiment'] = {'error': str(e)}
            
        return result
        
    def batch_analyze(self, texts):
        """Analyze multiple texts."""
        return [self.analyze_sentiment(text) for text in texts] 