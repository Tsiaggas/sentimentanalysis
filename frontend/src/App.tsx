import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Box, 
  Typography, 
  TextField, 
  Button, 
  Paper,
  CircularProgress,
  Grid,
  Alert,
  Snackbar,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Tabs,
  Tab,
  Divider,
  Card,
  CardContent
} from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Bar, Radar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  RadialLinearScale,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  RadialLinearScale,
  Title,
  Tooltip,
  Legend,
  Filler
);

// Create theme
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

// Define interface for the API response
interface SentimentResult {
  text: string;
  language: string;
  overall_sentiment: string;
  sentiment: {
    vader?: number;
    textblob?: number;
    bert?: number;
    lexicon?: number;
    multilingual?: number;
    [key: string]: number | undefined;
  };
  emotions: {
    joy: number;
    trust: number;
    pleasure: number;
    anxiety: number;
    anger: number;
    sadness: number;
    [key: string]: number;
  };
  confidence: number;
}

// Example texts for the user to try
const exampleTexts = {
  en: [
    "This product is excellent! I love it and would recommend it to everyone.",
    "I am very disappointed with this service. It's terrible and I will not use it again.",
    "The food was okay, not great but not bad either. The service was good though."
  ],
  el: [
    "Αυτό το προϊόν είναι εξαιρετικό! Το αγαπώ και θα το συνιστούσα σε όλους.",
    "Είμαι πολύ απογοητευμένος με αυτήν την υπηρεσία. Είναι απαίσια και δεν θα την ξαναχρησιμοποιήσω.",
    "Το φαγητό ήταν εντάξει, όχι τέλειο αλλά ούτε κακό. Η εξυπηρέτηση όμως ήταν καλή."
  ]
};

// Translation map for emotion labels
const emotionTranslations = {
  en: {
    joy: 'Joy',
    trust: 'Trust',
    pleasure: 'Pleasure',
    anxiety: 'Anxiety',
    anger: 'Anger',
    sadness: 'Sadness'
  },
  el: {
    joy: 'Χαρά',
    trust: 'Εμπιστοσύνη',
    pleasure: 'Ευχαρίστηση',
    anxiety: 'Άγχος',
    anger: 'Θυμός',
    sadness: 'Λύπη'
  }
};

// Translate sentiment labels
const translateSentiment = (sentiment: string, language: string) => {
  if (language === 'el') {
    switch (sentiment) {
      case 'positive': return 'Θετικό';
      case 'negative': return 'Αρνητικό';
      case 'neutral': return 'Ουδέτερο';
      default: return sentiment;
    }
  }
  return sentiment;
};

function App() {
  const [text, setText] = useState('');
  const [selectedLanguage, setSelectedLanguage] = useState<'en' | 'el'>('en');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SentimentResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showError, setShowError] = useState(false);
  const [backendAlive, setBackendAlive] = useState(false);
  const [currentTab, setCurrentTab] = useState(0);

  // Check if backend is alive on component mount
  useEffect(() => {
    const checkBackendStatus = async () => {
      try {
        const response = await fetch('http://127.0.0.1:5000/api/health');
        if (response.ok) {
          setBackendAlive(true);
        } else {
          setBackendAlive(false);
          setError('Το backend API δεν είναι διαθέσιμο');
          setShowError(true);
        }
      } catch (err) {
        setBackendAlive(false);
        setError('Δεν είναι δυνατή η σύνδεση με το backend API');
        setShowError(true);
      }
    };
    
    checkBackendStatus();
  }, []);

  const analyzeSentiment = async () => {
    if (!text.trim()) {
      setError('Παρακαλώ εισάγετε κείμενο για ανάλυση');
      setShowError(true);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      console.log('Sending request to:', 'http://127.0.0.1:5000/api/analyze');
      console.log('With payload:', { text });
      
      const response = await fetch('http://127.0.0.1:5000/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      console.log('Response status:', response.status);

      if (!response.ok) {
        throw new Error(`Σφάλμα HTTP: ${response.status}`);
      }

      const data = await response.json();
      console.log('Response data:', data);
      
      console.log('Sentiment keys:', Object.keys(data.sentiment || {}));
      
      setResult(data);
      setSelectedLanguage(data.language === 'el' ? 'el' : 'en');
    } catch (err) {
      console.error('Error during analysis:', err);
      setError(err instanceof Error ? err.message : 'Σφάλμα στην ανάλυση');
      setShowError(true);
    } finally {
      setLoading(false);
    }
  };

  const renderEmotionBarChart = () => {
    if (!result?.emotions) return null;

    const emotionLabels = Object.keys(result.emotions);
    const translatedLabels = emotionLabels.map(label => 
      emotionTranslations[selectedLanguage][label as keyof typeof emotionTranslations.en] || label);
    
    const emotionValues = emotionLabels.map(label => result.emotions[label]);

    const data = {
      labels: translatedLabels,
      datasets: [
        {
          label: selectedLanguage === 'el' ? 'Επίπεδο Συναισθήματος' : 'Emotion Level',
          data: emotionValues,
          backgroundColor: [
            'rgba(255, 99, 132, 0.7)',
            'rgba(54, 162, 235, 0.7)',
            'rgba(255, 206, 86, 0.7)',
            'rgba(75, 192, 192, 0.7)',
            'rgba(153, 102, 255, 0.7)',
            'rgba(255, 159, 64, 0.7)'
          ],
          borderColor: [
            'rgba(255, 99, 132, 1)',
            'rgba(54, 162, 235, 1)',
            'rgba(255, 206, 86, 1)',
            'rgba(75, 192, 192, 1)',
            'rgba(153, 102, 255, 1)',
            'rgba(255, 159, 64, 1)'
          ],
          borderWidth: 1
        },
      ],
    };

    const options = {
      responsive: true,
      plugins: {
        legend: {
          position: 'top' as const,
        },
        title: {
          display: true,
          text: selectedLanguage === 'el' ? 'Ανάλυση Συναισθημάτων' : 'Emotion Analysis',
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 1,
        },
      },
    };

    return <Bar data={data} options={options} />;
  };

  const renderEmotionRadarChart = () => {
    if (!result?.emotions) return null;

    const emotionLabels = Object.keys(result.emotions);
    const translatedLabels = emotionLabels.map(label => 
      emotionTranslations[selectedLanguage][label as keyof typeof emotionTranslations.en] || label);
    
    const emotionValues = emotionLabels.map(label => result.emotions[label]);

    const data = {
      labels: translatedLabels,
      datasets: [
        {
          label: selectedLanguage === 'el' ? 'Συναισθήματα' : 'Emotions',
          data: emotionValues,
          fill: true,
          backgroundColor: 'rgba(54, 162, 235, 0.2)',
          borderColor: 'rgb(54, 162, 235)',
          pointBackgroundColor: 'rgb(54, 162, 235)',
          pointBorderColor: '#fff',
          pointHoverBackgroundColor: '#fff',
          pointHoverBorderColor: 'rgb(54, 162, 235)'
        }
      ]
    };

    const options = {
      scales: {
        r: {
          beginAtZero: true,
          min: 0,
          max: 1,
          ticks: {
            stepSize: 0.2
          }
        }
      }
    };

    return <Radar data={data} options={options} />;
  };

  const renderSentimentValues = () => {
    if (!result?.sentiment) return null;

    return (
      <Grid container spacing={2} sx={{ mb: 3 }}>
        {Object.entries(result.sentiment).map(([key, value]) => (
          <Grid item xs={6} sm={4} key={key}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6" gutterBottom component="div" align="center">
                  {key.charAt(0).toUpperCase() + key.slice(1)}
                </Typography>
                <Typography variant="h4" component="div" align="center" color={
                  value && value > 0.2 ? 'success.main' : 
                  value && value < -0.2 ? 'error.main' : 'text.primary'
                }>
                  {value !== undefined ? value.toFixed(2) : 'N/A'}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    );
  };

  const handleExampleClick = (index: number) => {
    setText(exampleTexts[selectedLanguage][index]);
  };

  const handleCloseError = () => {
    setShowError(false);
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="lg">
        <Box sx={{ my: 4 }}>
          <Typography variant="h3" component="h1" gutterBottom align="center">
            {selectedLanguage === 'el' ? 'Ανάλυση Συναισθημάτων' : 'Sentiment Analysis'}
          </Typography>

          {!backendAlive && (
            <Alert severity="error" sx={{ mb: 3 }}>
              {selectedLanguage === 'el' 
                ? 'Το backend API δεν είναι διαθέσιμο. Βεβαιωθείτε ότι ο server τρέχει στο http://127.0.0.1:5000'
                : 'The backend API is not available. Please make sure the server is running at http://127.0.0.1:5000'}
            </Alert>
          )}

          <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
            <Grid container spacing={2}>
              <Grid item xs={12} md={4}>
                <FormControl fullWidth>
                  <InputLabel id="language-select-label">
                    {selectedLanguage === 'el' ? 'Προτιμώμενη γλώσσα' : 'Preferred Language'}
                  </InputLabel>
                  <Select
                    labelId="language-select-label"
                    value={selectedLanguage}
                    label={selectedLanguage === 'el' ? 'Προτιμώμενη γλώσσα' : 'Preferred Language'}
                    onChange={(e) => setSelectedLanguage(e.target.value as 'en' | 'el')}
                  >
                    <MenuItem value="en">English</MenuItem>
                    <MenuItem value="el">Ελληνικά</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={8}>
                <TextField
                  fullWidth
                  multiline
                  rows={4}
                  variant="outlined"
                  label={selectedLanguage === 'el' ? 'Εισάγετε το κείμενο για ανάλυση' : 'Enter text to analyze'}
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  error={!!error && !showError}
                  helperText={error && !showError ? error : ''}
                />
              </Grid>
              <Grid item xs={12}>
                <Button
                  fullWidth
                  variant="contained"
                  onClick={analyzeSentiment}
                  disabled={loading || !backendAlive}
                >
                  {loading 
                    ? <CircularProgress size={24} /> 
                    : selectedLanguage === 'el' ? 'Ανάλυση' : 'Analyze'}
                </Button>
              </Grid>
              <Grid item xs={12}>
                <Typography variant="subtitle2" gutterBottom>
                  {selectedLanguage === 'el' ? 'Παραδείγματα:' : 'Examples:'}
                </Typography>
                <Grid container spacing={1}>
                  {exampleTexts[selectedLanguage].map((example, index) => (
                    <Grid item key={index}>
                      <Button 
                        variant="outlined" 
                        size="small" 
                        onClick={() => handleExampleClick(index)}
                      >
                        {selectedLanguage === 'el' ? `Παράδειγμα ${index+1}` : `Example ${index+1}`}
                      </Button>
                    </Grid>
                  ))}
                </Grid>
              </Grid>
            </Grid>
          </Paper>

          {result && (
            <Paper elevation={3} sx={{ p: 3 }}>
              <Typography variant="h5" gutterBottom>
                {selectedLanguage === 'el' ? 'Αποτελέσματα Ανάλυσης' : 'Analysis Results'}
              </Typography>
              
              <Typography variant="body1" gutterBottom>
                <strong>{selectedLanguage === 'el' ? 'Γλώσσα:' : 'Language:'}</strong> {
                  result.language === 'el' ? 'Ελληνικά' : 
                  result.language === 'en' ? 'English' : 
                  result.language
                }
              </Typography>
              
              <Typography variant="body1" gutterBottom>
                <strong>{selectedLanguage === 'el' ? 'Κείμενο:' : 'Text:'}</strong> {result.text}
              </Typography>
              
              <Typography variant="body1" gutterBottom>
                <strong>{selectedLanguage === 'el' ? 'Συνολικό Συναίσθημα:' : 'Overall Sentiment:'}</strong> {
                  translateSentiment(result.overall_sentiment, selectedLanguage)
                }
              </Typography>
              
              <Typography variant="body1" gutterBottom>
                <strong>{selectedLanguage === 'el' ? 'Εμπιστοσύνη:' : 'Confidence:'}</strong> {(result.confidence * 100).toFixed(2)}%
              </Typography>
              
              <Divider sx={{ my: 2 }} />
              
              <Typography variant="h6" sx={{ mb: 2 }}>
                {selectedLanguage === 'el' ? 'Λεπτομέρειες Συναισθημάτων' : 'Sentiment Details'}
              </Typography>
              
              {renderSentimentValues()}
              
              <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
                <Tabs value={currentTab} onChange={handleTabChange} aria-label="chart tabs">
                  <Tab label={selectedLanguage === 'el' ? 'Ραβδόγραμμα' : 'Bar Chart'} />
                  <Tab label={selectedLanguage === 'el' ? 'Γράφημα Ραντάρ' : 'Radar Chart'} />
                </Tabs>
              </Box>
              
              <Box sx={{ mt: 3, height: 300 }}>
                {currentTab === 0 ? renderEmotionBarChart() : renderEmotionRadarChart()}
              </Box>
            </Paper>
          )}
        </Box>
      </Container>

      <Snackbar open={showError} autoHideDuration={6000} onClose={handleCloseError}>
        <Alert onClose={handleCloseError} severity="error">
          {error}
        </Alert>
      </Snackbar>
    </ThemeProvider>
  );
}

export default App; 