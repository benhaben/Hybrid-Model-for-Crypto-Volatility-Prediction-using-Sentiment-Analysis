
# Hybrid Model for Cryptocurrency Volatility Prediction using Social Media Sentiment Analysis 🚀

This project is designed to forecast cryptocurrency market volatility by integrating historical market data with real-time sentiment extracted from social media platforms such as Reddit. It leverages NLP techniques for sentiment analysis and machine learning models for volatility prediction. The system is built using Python and served via a Flask-based web interface.

---

## 📌 Project Highlights

- **Hybrid Approach:** Combines social media sentiment data with crypto market prices for better volatility prediction.
- **NLP + ML Models:** Uses DistilBERT for sentiment classification and Random Forest for volatility forecasting.
- **Full Data Pipeline:** From data collection, preprocessing, and sentiment scoring to modeling and deployment.
- **Interactive Web Interface:** Displays real-time sentiment scores, price movements, and volatility predictions.

---

## 🧱 Folder Structure

```
.
├── data_pipeline.py   # Sentiment + market data collection pipeline
│ 
├── Data_set/
│   ├── reddit_data_raw.csv
│   ├── final_data.csv
│   └── crypto_market_dataset.csv
│
├── Models/
│   └── best_random_forest_model.joblib
│
├── static/
│   ├── index.html                # Web UI
│   ├── script.js
│   └── style.css
│
├── app.py                        # Flask app for backend APIs
├── .env                          # Reddit API credentials
├── requirements.txt              # Required packages
└── README.md                     # Documentation
```

---

## Installation

1. Clone the repo:

```bash
git clone <your-repo-url>
cd your-project-directory
```

2. Create a virtual environment and activate:

```bash
python -m venv venv
source venv/Scripts/activate   # On Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file and insert:

```
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_secret
REDDIT_USER_AGENT=your_user_agent
```

---

## 🧪 How It Works

###  Data Collection

- Reddit data is scraped using `PRAW` for crypto-related subreddits like `r/Bitcoin`, `r/Crypto`, etc.
- Historical price data is fetched from CoinGecko API.

### Sentiment Analysis

- Reddit `title` and `body` text is preprocessed using regex.
- `DistilBERT` model fine-tuned on SST-2 is used to score sentiment for each post.
- Scores are normalized between [-1, 1] and aggregated daily.

### Market Data Engineering

- Price and volume data is processed for volatility using rolling standard deviation of returns.
- Lagged features (1 to 7 days) are generated for market and sentiment trends.
- Combined into a single feature-rich dataset (`final_data.csv`).

---

## Model Selection & Training

The model training process consists of:
- **Feature Engineering:** From sentiment scores (title & description), market returns, volume & volatility, lag features, and trend indicators.
- **Model Selection Process:**
We explored multiple modeling strategies, comparing traditional statistical models with modern machine learning approaches:

- **Time Series Models:**
  - ARIMA: For capturing autoregressive and moving average patterns in volatility
  - GARCH: Known for modeling financial time series volatility

- **Machine Learning Models:**
  - **Random Forest (Selected):** Provided the best performance and interpretability
  - XGBoost and LightGBM were also evaluated but offered no significant accuracy gain
  
- **Validation Strategy:** Time-series aware cross-validation.
- **Model Evaluation:** Metrics used include Mean Absolute Error and Mean Squared Error.
- **Persistence:** Best performing model saved via `joblib` in `Models/` directory.
- **Retraining Ready:** Retrains are supported using updated final_data.csv if desired.

---

## Usage

### 1. Run the pipeline (to update data)

```bash
python data_pipeline.py
```

### 2. Start the Flask server

```bash
python app.py
```

Then visit `http://127.0.0.1:5000` in your browser.

---

## 🌐 API Endpoints

- `/api/historical_data` → Last 30 days of price + sentiment data
- `/api/volatility_forecast` → Predicts next 24hr volatility
- `/api/sentiment_analysis` → Average sentiment from recent Reddit data

---

## 💻 Web Interface

- Built using HTML + JS + CSS (under `static/`)
- Dynamically fetches predictions and plots using Chart.js

---

## 📈 Future Scope

- Add LSTM + ARIMA forecasting models
- Include other social platforms (Twitter/X)
- Schedule automated retraining with Airflow
- Deploy to cloud (Render, AWS EC2, or Vercel)

> Developed with 💡 by Somya Jain
