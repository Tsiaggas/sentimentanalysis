import React, { useState } from 'react';
import { 
  Container, 
  Box, 
  Typography, 
  TextField, 
  Button, 
  Paper,
  CircularProgress,
  Grid
} from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
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

function App() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const analyzeSentiment = async () => {
    if (!text.trim()) {
      setError('Παρακαλώ εισάγετε κείμενο για ανάλυση');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://127.0.0.1:5000/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error('Σφάλμα στην ανάλυση');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Σφάλμα στην ανάλυση');
    } finally {
      setLoading(false);
    }
  };

  const renderEmotionChart = () => {
    if (!result?.emotions) return null;

    const data = {
      labels: Object.keys(result.emotions),
      datasets: [
        {
          label: 'Επίπεδο Συναισθήματος',
          data: Object.values(result.emotions),
          borderColor: 'rgb(75, 192, 192)',
          tension: 0.1,
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

    return <Line data={data} options={options} />;
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="lg">
        <Box sx={{ my: 4 }}>
          <Typography variant="h3" component="h1" gutterBottom align="center">
            Ανάλυση Συναισθημάτων
          </Typography>

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
                  error={!!error}
                  helperText={error}
                />
              </Grid>
              <Grid item xs={12}>
                <Button
                  fullWidth
                  variant="contained"
                  onClick={analyzeSentiment}
                  disabled={loading}
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
                Συνολικό Συναίσθημα: {result.overall_sentiment}
              </Typography>
              
              <Typography variant="body1" gutterBottom>
                Εμπιστοσύνη: {(result.confidence * 100).toFixed(2)}%
              </Typography>

              <Box sx={{ mt: 3 }}>
                {renderEmotionChart()}
              </Box>
            </Paper>
          )}
        </Box>
      </Container>
    </ThemeProvider>
  );
}

export default App; 