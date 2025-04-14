let historicalChart;

document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM fully loaded and parsed');
    fetchHistoricalData();
    fetchVolatilityForecast();
    fetchSentimentAnalysis();
});

async function fetchHistoricalData() {
    try {
        console.log('Fetching historical data...');
        const response = await fetch('/api/historical_data');
        console.log('Response status:', response.status);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        console.log('Received data:', JSON.stringify(data, null, 2));
        console.log('Sentiment data:', data.sentiment);
        renderHistoricalChart(data);
    } catch (error) {
        console.error('Error in fetchHistoricalData:', error);
        document.getElementById('historicalChart').innerHTML = '<p>Error loading historical data. Please try again later.</p>';
    }
}

function renderHistoricalChart(data) {
    console.log('Rendering historical chart...');
    const ctx = document.getElementById('historicalChart');
    if (!ctx) {
        console.error('Cannot find canvas element with id "historicalChart"');
        return;
    }

    console.log('Canvas element found');

    if (historicalChart) {
        console.log('Destroying existing chart');
        historicalChart.destroy();
    }
    
    console.log('Creating new chart with data:', JSON.stringify(data, null, 2));
    console.log('Sentiment data for chart:', data.sentiment);

    historicalChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.timestamp,
            datasets: [{
                label: 'Price (USD)',
                data: data.price,
                borderColor: 'rgb(75, 192, 192)',
                yAxisID: 'y-axis-1',
            }, {
                label: 'Sentiment',
                data: data.sentiment,
                borderColor: 'rgb(255, 99, 132)',
                yAxisID: 'y-axis-2',
            }]
        },
        options: {
            responsive: true,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day',
                        parser: 'yyyy-MM-dd HH:mm:ss'
                    }
                },
                'y-axis-1': {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Price (USD)'
                    }
                },
                'y-axis-2': {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Sentiment'
                    },
                    grid: {
                        drawOnChartArea: false,
                    },
                }
            }
        }
    });

    console.log('Chart rendering complete');

}

async function fetchVolatilityForecast() {
    try {
        const response = await fetch('/api/volatility_forecast');
        if (!response.ok) {
            throw new Error('Failed to fetch volatility forecast');
        }
        const data = await response.json();
        displayVolatilityForecast(data);
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('forecast-result').innerHTML = '<p>Error loading volatility forecast. Please try again later.</p>';
    }
}

function displayVolatilityForecast(data) {
    const forecastElement = document.getElementById('forecast-result');
    forecastElement.innerHTML = `
        <p>Predicted Volatility: ${data.predicted_volatility.toFixed(4)}</p>
        <p>Forecast Period: ${data.forecast_period}</p>
    `;
}

async function fetchSentimentAnalysis() {
    try {
        const response = await fetch('/api/sentiment_analysis');
        if (!response.ok) {
            throw new Error('Failed to fetch sentiment analysis');
        }
        const data = await response.json();
        displaySentimentAnalysis(data);
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('sentiment-result').innerHTML = '<p>Error loading sentiment analysis. Please try again later.</p>';
    }
}

function displaySentimentAnalysis(data) {
    const sentimentElement = document.getElementById('sentiment-result');
    sentimentElement.innerHTML = `
        <p>Sentiment Score: ${data.sentiment_score.toFixed(2)}</p>
        <p>Analysis Period: ${data.analysis_period}</p>
    `;
}

setInterval(fetchHistoricalData, 5 * 60 * 1000);
setInterval(fetchVolatilityForecast, 5 * 60 * 1000);
setInterval(fetchSentimentAnalysis, 5 * 60 * 1000);