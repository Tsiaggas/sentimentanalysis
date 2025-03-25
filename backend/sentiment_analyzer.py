import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import re
import string
import os
import json
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
            
            # Additional positive Greek words
            'συνιστώ': 1.2, 'προτείνω': 0.9, 'συστήνω': 1.0, 'επαινώ': 1.3, 'συγχαίρω': 1.4,
            'εύγε': 1.5, 'μπράβο': 1.5, 'ναι': 0.5, 'συμφωνώ': 0.7, 'αποδέχομαι': 0.6,
            'σωστό': 0.8, 'σωστά': 0.8, 'επιθυμητό': 0.9, 'αγαπητό': 1.0, 'αξιέπαινος': 1.2,
            'αξιοσέβαστος': 1.0, 'αξιομνημόνευτος': 0.9, 'ενδιαφέρον': 0.6, 'εξυπηρετικός': 0.9,
            'χαμογελαστός': 1.0, 'πρόθυμος': 0.8, 'διασκεδαστικός': 1.1, 'καθαρός': 0.7,
            'δωρεάν': 0.6, 'επωφελής': 0.8, 'συμφέρον': 0.7, 'αξίζει': 1.0, 'ευγενικός': 0.9,
            'ειλικρινής': 0.8, 'σημαντικός': 0.6, 'ισχυρός': 0.7, 'δίκαιος': 0.8, 'ακριβής': 0.7,
            'αυθεντικός': 0.8, 'γνήσιος': 0.7, 'ωφέλιμος': 0.8, 'ταχύς': 0.7, 'επιμελής': 0.6,
            'ευγένεια': 0.9, 'ευκολία': 0.7, 'άνεση': 0.8, 'διασκέδαση': 1.0, 'χαρά': 1.2,
            'μαγεία': 1.1, 'μαγευτικός': 1.2, 'ομορφιά': 0.9, 'αρέσει': 1.0, 'αρεστός': 0.9,
            'επιτυχία': 1.0, 'επιτυχημένο': 1.0, 'πετυχημένο': 1.0, 'πρωτοπόρος': 0.9,
            'καινοτόμος': 0.8, 'ευκαιρία': 0.7, 'δημοφιλής': 0.8, 'ευτυχία': 1.3, 'ευχάριστα': 0.8,
            'συμπαθώ': 0.9, 'σέβομαι': 0.7, 'εκτίμηση': 0.8, 'πλεονέκτημα': 0.7, 'κερδίζω': 0.8,
            'εορταστικός': 0.9, 'ενθαρρυντικός': 0.8, 'ελπιδοφόρος': 0.8, 'υποσχόμενος': 0.7,
            'φωτεινός': 0.7, 'λαμπρός': 0.9, 'ενθουσιώδης': 1.0, 'ζωντανός': 0.8, 'ζωηρός': 0.8,
            'νικητής': 1.0, 'επιτυχημένα': 1.0, 'υγιής': 0.7, 'ευημερία': 0.9, 'ευχαρίστηση': 1.1,
            
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
            
            # Additional negative Greek words
            'όχι': -0.5, 'διαφωνώ': -0.7, 'απορρίπτω': -0.9, 'αρνούμαι': -0.8, 'υποτιμώ': -0.9,
            'λάθος': -0.8, 'σφάλμα': -0.7, 'ανακριβής': -0.7, 'απατεώνας': -1.3, 'εξαπάτηση': -1.2,
            'κλέβω': -1.1, 'κλοπή': -1.2, 'εξαπατώ': -1.1, 'ψεύδομαι': -1.0, 'ψέμα': -1.0,
            'αποτυχία': -1.1, 'αποτυγχάνω': -1.0, 'χάνω': -0.8, 'ήττα': -0.9, 'ηττημένος': -0.9,
            'αδικία': -1.0, 'άδικος': -0.9, 'ανέντιμος': -1.0, 'ανειλικρινής': -0.9, 'υποκριτικός': -0.9,
            'επικίνδυνος': -0.9, 'κίνδυνος': -0.8, 'επιβλαβής': -0.9, 'επιζήμιος': -0.8, 'βλαβερός': -0.9,
            'τοξικός': -1.2, 'μολυσμένος': -1.0, 'μόλυνση': -0.9, 'ρύπανση': -0.8, 'βρώμικος': -0.7,
            'ακάθαρτος': -0.8, 'σπασμένος': -0.7, 'χαλασμένος': -0.8, 'κατεστραμμένος': -0.9, 'καταστροφή': -1.0,
            'αποτυχημένα': -1.0, 'έλλειψη': -0.6, 'απουσία': -0.5, 'ελλειπής': -0.7, 'ανεπαρκώς': -0.7,
            'ψευδής': -0.9, 'ψεύτικος': -0.8, 'απάτη': -1.1, 'παραπλανητικός': -0.9, 'εξοργιστικός': -1.2,
            'οργίζομαι': -1.1, 'εξοργίζομαι': -1.2, 'τρελαίνομαι': -1.0, 'τρέλα': -0.8, 'εκνευρισμένος': -0.9,
            'πικραμένος': -0.9, 'πίκρα': -0.8, 'δυσφορία': -0.7, 'δυσφορώ': -0.7, 'αρρώστια': -0.8,
            'άρρωστος': -0.8, 'αδιαθεσία': -0.7, 'αδιάθετος': -0.7, 'κόπωση': -0.6, 'κουρασμένος': -0.6,
            'εξαντλημένος': -0.8, 'εξάντληση': -0.8, 'αναστατωμένος': -0.8, 'αναστάτωση': -0.7, 'ταραγμένος': -0.8,
            'ταραχή': -0.7, 'ανησυχητικός': -0.7, 'δυσοίωνος': -0.8, 'καταστροφικός': -1.1, 'μοιραίος': -0.9,
            'τραγικός': -1.0, 'τραγωδία': -1.1, 'δράμα': -0.7, 'δραματικός': -0.6, 'κατάθλιψη': -1.2,
            'καταθλιπτικός': -1.1, 'απελπισία': -1.2, 'απελπιστικός': -1.1, 'απελπισμένος': -1.1, 'μάταιος': -0.8,
            'ματαιότητα': -0.7, 'ανώφελος': -0.8, 'ανούσιος': -0.7, 'ανούσια': -0.7, 'χάσιμο': -0.8,
            'σπατάλη': -0.8, 'αμφιβολία': -0.6, 'διστάζω': -0.5, 'διστακτικός': -0.5, 'φόβος': -0.9,
            'φοβάμαι': -0.8, 'φοβισμένος': -0.8, 'τρομοκρατημένος': -1.1, 'τρόμος': -1.0,
            
            # Common modifiers
            'πολύ': 0.3, 'αρκετά': 0.2, 'λίγο': 0.1, 'εξαιρετικά': 0.4, 'απίστευτα': 0.4,
            'καθόλου': -0.3, 'σχεδόν': 0.1, 'απόλυτα': 0.3, 'σίγουρα': 0.2, 'μάλλον': 0.1,
            'υπερβολικά': 0.3, 'τελείως': 0.3, 'αναμφίβολα': 0.3, 'αδιαμφισβήτητα': 0.3,
            'ελάχιστα': -0.2, 'μηδαμινά': -0.3, 'απολύτως': 0.3, 'ιδιαίτερα': 0.3,
            'εντελώς': 0.3, 'ιδιαιτέρως': 0.3, 'ουσιαστικά': 0.2, 'περισσότερο': 0.2,
            'λιγότερο': -0.2, 'ασυνήθιστα': 0.2, 'αρκετά': 0.2, 'απλά': 0.1,
            'μόνο': -0.1, 'μονάχα': -0.1, 'απλώς': 0.1, 'ακριβώς': 0.2,
            'ειδικά': 0.2, 'κυρίως': 0.1, 'ακόμα': 0.1, 'επιπλέον': 0.1,
            'επιπροσθέτως': 0.1, 'ακόμη': 0.1, 'περαιτέρω': 0.1, 'παραπάνω': 0.1
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
            
        # Εντοπισμός άρνησης
        negation_words = {'δεν', 'μην', 'όχι', 'ούτε', 'μηδέ', 'καθόλου', 'μηδέν', 'τίποτα', 'ποτέ'}
        negation_active = False
        negation_distance = 0
        max_negation_distance = 3  # Πόσες λέξεις επηρεάζει η άρνηση
        
        # Calculate sentiment score using lexicon
        total_score = 0
        found_words = 0
        words_scores = []
        
        for i, word in enumerate(words):
            # Έλεγχος για άρνηση
            if word in negation_words:
                negation_active = True
                negation_distance = 0
                continue
                
            # Αυξάνουμε την απόσταση από την τελευταία άρνηση
            if negation_active:
                negation_distance += 1
                if negation_distance > max_negation_distance:
                    negation_active = False
            
            # Έλεγχος για συναίσθημα
            score = 0
            if word in self.greek_sentiment_lexicon:
                score = self.greek_sentiment_lexicon[word]
                
                # Αντιστροφή πολικότητας αν υπάρχει άρνηση
                if negation_active:
                    score = -score
                    
                total_score += score
                found_words += 1
                words_scores.append((word, score))
        
        # Έλεγχος για παρουσία λέξεων συναισθήματος
        if found_words == 0:
            # Αν δε βρέθηκαν λέξεις συναισθήματος, προσπαθούμε να εντοπίσουμε συναισθηματικές φράσεις
            # Για παράδειγμα: "είναι καλή", "δεν είναι καλή"
            common_phrases = [
                ('είναι', 'καλό', 0.8), ('είναι', 'κακό', -0.8),
                ('είναι', 'ωραίο', 0.8), ('είναι', 'άσχημο', -0.8),
                ('πολύ', 'καλό', 1.0), ('πολύ', 'κακό', -1.0)
            ]
            
            # Αναζήτηση για συνδυασμούς λέξεων
            for i in range(len(words) - 1):
                for phrase in common_phrases:
                    if words[i] == phrase[0] and words[i+1] == phrase[1]:
                        score = phrase[2]
                        
                        # Έλεγχος για άρνηση πριν από τη φράση
                        if i > 0 and words[i-1] in negation_words:
                            score = -score
                            
                        total_score += score
                        found_words += 1
        
        # Ελέγχουμε επίσης για επιφωνήματα και εκφράσεις συναισθήματος
        expressions = {
            'μπράβο': 1.0, 'ζήτω': 1.0, 'ουφ': -0.7, 'αχ': -0.5, 
            'ωχ': -0.5, 'οχ': -0.5, 'εύγε': 1.0, 'α μπράβο': 1.0, 
            'τι καλά': 1.0, 'τι ωραία': 1.0, 'τι κρίμα': -0.8, 
            'κρίμα': -0.8, 'αίσχος': -1.0, 'ντροπή': -0.9
        }
        
        for expr, score in expressions.items():
            if expr in text.lower():
                total_score += score
                found_words += 1
                
        # Normalize score to range similar to VADER (-1 to 1)
        sentiment_score = 0
        if found_words > 0:
            sentiment_score = total_score / found_words
            
        # Προσαρμογή της έντασης του αποτελέσματος για τα ελληνικά
        # (Διατήρηση της θετικότητας/αρνητικότητας αλλά με μικρή ενίσχυση)
        if sentiment_score != 0:
            sentiment_score = sentiment_score * 1.2
            
        # Scale to match VADER compound score
        scaled_score = max(min(sentiment_score, 1.0), -1.0)
        
        # Καταγραφή για έλεγχο
        print(f"Greek sentiment analysis: score={scaled_score}, found_words={found_words}, word_scores={words_scores}")
        
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

    def analyze_sentiment(self, text):
        """Perform comprehensive sentiment analysis using multiple models and language detection."""
        # Detect language
        language = self.detect_language(text)
        print(f"Detected language: {language}")
        
        # Προστατεύομαστε από κενό κείμενο
        if not text or text.strip() == "":
            return {
                'text': text,
                'language': 'unknown',
                'overall_sentiment': 'neutral',
                'sentiment': {'error': 'empty text'},
                'emotions': {},
                'confidence': 0.5
            }
            
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
                
                try:
                    # Use multilingual BERT for sentiment analysis
                    multi_result = self.multilingual_model(text[:512])  # Using truncated text to avoid exceeding model's max length
                    
                    print(f"Multilingual model result: {multi_result}")
                    multi_label = multi_result[0]['label']
                    multi_score = multi_result[0]['score']
                    
                    # Convert BERT score to -1 to 1 range
                    if 'stars' in multi_label.lower():
                        # Extract star rating (e.g., "5 stars" -> 5)
                        star_match = re.search(r'(\d+)', multi_label)
                        if star_match:
                            stars = int(star_match.group(1))
                            # Convert 1-5 scale to -1 to 1
                            normalized_multi_score = (stars - 3) / 2
                        else:
                            normalized_multi_score = 0
                    else:
                        # For non-star labels, convert based on sentiment words
                        if any(pos_word in multi_label.lower() for pos_word in ['positive', 'θετικό']):
                            normalized_multi_score = multi_score
                        elif any(neg_word in multi_label.lower() for neg_word in ['negative', 'αρνητικό']):
                            normalized_multi_score = -multi_score
                        else:
                            normalized_multi_score = 0
                    
                    multi_score = normalized_multi_score
                    print(f"Normalized multilingual score: {multi_score}")
                    
                    # Store sentiment scores
                    result['sentiment'] = {
                        'lexicon': greek_scores['compound'],
                        'multilingual': multi_score
                    }
                    
                    # Δίνουμε μεγαλύτερο βάρος στο λεξικό για ελληνικά καθώς είναι πιο εξειδικευμένο
                    avg_sentiment = (greek_scores['compound'] * 0.7 + multi_score * 0.3)
                    print(f"Using weighted average (lexicon=0.7, multilingual=0.3): {avg_sentiment}")
                    
                except Exception as e:
                    print(f"Error in multilingual model: {e}")
                    
                    # Use only lexicon if multilingual failed
                    result['sentiment'] = {
                        'lexicon': greek_scores['compound']
                    }
                    avg_sentiment = greek_scores['compound']
                
                # Analyze emotions in Greek
                result['emotions'] = self.analyze_emotions(processed_text, 'el')
                
            else:
                # English text analysis
                print("Performing English sentiment analysis")
                
                # Use VADER for sentiment analysis
                vader_scores = self.vader.polarity_scores(processed_text)
                print(f"VADER scores: {vader_scores}")
                
                # Use HuggingFace transformer for sentiment classification
                bert_result = self.english_model(text[:512])  # Using truncated text
                print(f"BERT result: {bert_result}")
                
                bert_label = bert_result[0]['label']
                bert_score = bert_result[0]['score']
                
                # Convert transformer score to -1 to 1 range
                if 'star' in bert_label.lower():
                    # Extract star rating if available
                    star_match = re.search(r'(\d+)', bert_label)
                    if star_match:
                        stars = int(star_match.group(1))
                        # Convert 1-5 scale to -1 to 1
                        normalized_bert_score = (stars - 3) / 2
                    else:
                        normalized_bert_score = 0
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
                
                # Add normal sentiment scores
                result['sentiment'] = {
                    'vader': vader_scores['compound'],
                    'bert': normalized_bert_score
                }
                
                # Για τα αγγλικά δίνουμε ίση βαρύτητα καθώς και τα δύο μοντέλα έχουν καλή απόδοση
                avg_sentiment = (vader_scores['compound'] + normalized_bert_score) / 2
                print(f"Using plain average: {avg_sentiment}")
                
                # Analyze emotions in English
                result['emotions'] = self.analyze_emotions(processed_text, 'en')
            
            # Determine overall sentiment and confidence with πιο ευαίσθητα κατώφλια 
            # για καλύτερη ταξινόμηση των ουδέτερων περιπτώσεων
            if avg_sentiment > 0.15:
                result['overall_sentiment'] = 'positive'
            elif avg_sentiment < -0.15:
                result['overall_sentiment'] = 'negative'
            else:
                result['overall_sentiment'] = 'neutral'
                
            # Calculate confidence as the absolute value of sentiment with scaling
            # για να εμφανίζει υψηλότερες τιμές εμπιστοσύνης
            if abs(avg_sentiment) > 0.5:
                confidence = 0.9  # Υψηλό συναίσθημα = υψηλή εμπιστοσύνη
            elif abs(avg_sentiment) > 0.3:
                confidence = 0.8
            elif abs(avg_sentiment) > 0.15:
                confidence = 0.7
            elif abs(avg_sentiment) > 0.05:
                confidence = 0.6
            else:
                confidence = 0.5  # Ουδέτερο = μεσαία εμπιστοσύνη
                
            result['confidence'] = confidence
            
            # Αποθηκεύουμε και το πραγματικό σκορ για αναφορά
            result['score'] = avg_sentiment
            
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            # Provide basic fallback results
            result['overall_sentiment'] = 'neutral'
            result['confidence'] = 0.5
            result['sentiment'] = {'error': str(e)}
            
        return result
        
    def batch_analyze(self, texts):
        """Analyze a batch of texts."""
        return [self.analyze_sentiment(text) for text in texts] 