# Sentiment Analysis Web Application

Μια πλήρης εφαρμογή ανάλυσης συναισθημάτων για κριτικές πελατών, χρησιμοποιώντας τεχνικές NLP και μηχανικής μάθησης.

## Χαρακτηριστικά

- Ανάλυση συναισθημάτων με πολλαπλά μοντέλα (VADER, BERT, TextBlob)
- Ανίχνευση σαρκασμού και ειρωνείας
- Πολυδιάστατη ανάλυση συναισθημάτων
- RESTful API με Flask
- React.js frontend με Material-UI
- Οπτικοποίηση δεδομένων με Chart.js
- Αυτόματη δημιουργία αναφορών
- Υποστήριξη πολλαπλών γλωσσών

## Προαπαιτούμενα

- Python 3.9+
- Node.js 16+
- MongoDB (προαιρετικό)

## Εγκατάσταση

### Backend

1. Δημιουργήστε ένα virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Εγκαταστήστε τις εξαρτήσεις:
```bash
pip install -r requirements.txt
```

3. Εκκινήστε τον server:
```bash
python backend/app.py
```

### Frontend

1. Εγκαταστήστε τις εξαρτήσεις:
```bash
cd frontend
npm install
```

2. Εκκινήστε την εφαρμογή:
```bash
npm start
```

## Χρήση

1. Ανοίξτε τον browser στο `http://localhost:3000`
2. Εισάγετε κείμενο για ανάλυση
3. Δείτε τα αποτελέσματα και τα γραφήματα
4. Δημιουργήστε αναφορές

## Ανάπτυξη

- Backend: `http://localhost:5000`
- Frontend: `http://localhost:3000`

## Δοκιμές

```bash
# Backend tests
pytest

# Frontend tests
cd frontend
npm test
```

## Άδεια

MIT License 
>>>>>>> b89d53d (initial project)
