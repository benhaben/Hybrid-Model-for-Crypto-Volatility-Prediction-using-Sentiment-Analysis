from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler
import pandas as pd
import joblib
from Data_pipeline.data_pipeline import process_dataset

from datetime import datetime, timedelta
import logging
import os

app = Flask(__name__, static_folder='static')
CORS(app)
logging.basicConfig(level=logging.DEBUG)

# Correct relative path to the model
model_path = os.path.join(os.path.dirname(__file__), 'Models', 'best_random_forest_model.joblib')
volatility_model = joblib.load(model_path)


data = pd.DataFrame()
def update_data():
    global data, sentiment_data
    try:
        new_market_data = process_dataset()
        
        if new_market_data is not None and not new_market_data.empty:
            if data.empty:
                data = new_market_data
            else:
                data = pd.concat([data, new_market_data]).drop_duplicates(subset='timestamp', keep='last')
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(days=7)
        data = data[data['timestamp'] > cutoff_time]
        logging.info(f"Data updated successfully. Market data shape: {data.shape}")
    except Exception as e:
        logging.error(f"Error updating data: {str(e)}")

# scheduler = BackgroundScheduler()
# scheduler.add_job(func=update_data, trigger="interval", hours=1)
# scheduler.start()


@app.route('/', methods=['GET'])
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/historical_data', methods=['GET'])
def get_historical_data():
    global data
    logging.info(f"get_historical_data called. Data shape: {data.shape}")
    if not data.empty:
        historical_data = data.sort_values('timestamp').tail(7) 
        result = {
            'timestamp': historical_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'price': historical_data['bitcoin_price'].tolist(),
            'sentiment': historical_data['total_sentiment'].tolist(),
        }
        logging.info(f"Returning historical data: {result}")
        return jsonify(result)
    else:
        logging.warning("No historical data available")
        return jsonify({'error': 'No historical data available'}), 404


@app.route('/api/volatility_forecast', methods=['GET'])
def get_volatility_forecast():
    global data
    if not data.empty:
        features = data.drop(['timestamp', 'bitcoin_volatility'], axis=1).tail(1)
        prediction = volatility_model.predict(features)[0]
        return jsonify({
            'predicted_volatility': float(prediction),
            'forecast_period': '24 hours'
        })
    else:
        return jsonify({'error': 'Unable to make volatility forecast'}), 500

@app.route('/api/sentiment_analysis', methods=['GET'])
def get_sentiment_analysis():
    global data
    if not data.empty:
        recent_sentiment = data
        sentiment= recent_sentiment['total_sentiment'].to_list()
        return jsonify({
            'sentiment_score': float(sum(sentiment)/len(sentiment)),
            'analysis_period': '24 hours'
        })
    else:
        return jsonify({'error': 'No sentiment data available'}), 404

if __name__ == '__main__':
    update_data() 
    data.to_csv('final.csv')
    app.run(debug=True, use_reloader=False, port=8001)