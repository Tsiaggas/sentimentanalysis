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
  Snackbar
} from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
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
  overall_sentiment: string;
  sentiment: {
    vader: number;
    textblob: number;
    bert: number;
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

function App() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SentimentResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showError, setShowError] = useState(false);
  const [backendAlive, setBackendAlive] = useState(false);

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

      const data: SentimentResult = await response.json();
      console.log('Response data:', data);
      
      setResult(data);
    } catch (err) {
      console.error('Error during analysis:', err);
      setError(err instanceof Error ? err.message : 'Σφάλμα στην ανάλυση');
      setShowError(true);
    } finally {
      setLoading(false);
    }
  };

  const renderEmotionChart = () => {
    if (!result?.emotions) return null;

    const emotionLabels = Object.keys(result.emotions);
    const emotionValues = emotionLabels.map(label => result.emotions[label]);

    const data = {
      labels: emotionLabels,
      datasets: [
        {
          label: 'Επίπεδο Συναισθήματος',
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
          text: 'Ανάλυση Συναισθημάτων',
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

  const handleCloseError = () => {
    setShowError(false);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="lg">
        <Box sx={{ my: 4 }}>
          <Typography variant="h3" component="h1" gutterBottom align="center">
            Ανάλυση Συναισθημάτων
          </Typography>

          {!backendAlive && (
            <Alert severity="error" sx={{ mb: 3 }}>
              Το backend API δεν είναι διαθέσιμο. Βεβαιωθείτε ότι ο server τρέχει στο http://127.0.0.1:5000
            </Alert>
          )}

          <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  multiline
                  rows={4}
                  variant="outlined"
                  label="Εισάγετε το κείμενο για ανάλυση"
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
                  {loading ? <CircularProgress size={24} /> : 'Ανάλυση'}
                </Button>
              </Grid>
            </Grid>
          </Paper>

          {result && (
            <Paper elevation={3} sx={{ p: 3 }}>
              <Typography variant="h5" gutterBottom>
                Αποτελέσματα Ανάλυσης
              </Typography>
              
              <Typography variant="body1" gutterBottom>
                <strong>Κείμενο:</strong> {result.text}
              </Typography>
              
              <Typography variant="body1" gutterBottom>
                <strong>Συνολικό Συναίσθημα:</strong> {result.overall_sentiment === 'positive' ? 'Θετικό' : 
                                         result.overall_sentiment === 'negative' ? 'Αρνητικό' : 'Ουδέτερο'}
              </Typography>
              
              <Typography variant="body1" gutterBottom>
                <strong>Εμπιστοσύνη:</strong> {(result.confidence * 100).toFixed(2)}%
              </Typography>
              
              <Typography variant="h6" sx={{ mt: 2, mb: 1 }}>
                Επιμέρους Συναισθήματα
              </Typography>

              <Box sx={{ mt: 3 }}>
                {renderEmotionChart()}
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