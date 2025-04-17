from flask import Flask, jsonify, send_from_directory, request  # 新增 request
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler
import pandas as pd
import joblib
from Data_pipeline.data_pipeline import process_dataset

from datetime import datetime, timedelta
import logging
import os
import numpy as np  # 确保全局导入
import nltk
nltk.download('vader_lexicon')

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

scheduler = BackgroundScheduler()
scheduler.add_job(func=update_data, trigger="interval", hours=1)
scheduler.start()


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

@app.route('/api/article_analysis', methods=['POST'])
def analyze_article():
    try:
        # 数据验证与解析
        req_data = request.get_json()
        if not req_data or 'content' not in req_data:
            logging.warning("Invalid article payload received")
            return jsonify({'error': 'Missing article content'}), 400
            
        # 文本预处理（示例实现）
        article_content = req_data['content']
        clean_content = preprocess_text(article_content)  # 需实现预处理函数
        
        # 情感分析（集成现有数据管道）
        sentiment_score = calculate_sentiment(clean_content)  # 示例函数
        
        # 波动性关联分析（结合现有模型）
        # prediction_input = build_prediction_features(sentiment_score)  # 特征工程
        # volatility_pred = volatility_model.predict(prediction_input)[0]

        # # 记录分析结果
        # logging.info(f"Article analysis completed | Sentiment: {sentiment_score} | Volatility: {volatility_pred}")
        
        return jsonify({
            'sentiment_score': float(sentiment_score),
            # 'volatility_impact': float(volatility_pred),
            # 'analysis_result': classify_impact(sentiment_score, volatility_pred),
            'processed_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        logging.error(f"Article analysis failed: {str(e)}", exc_info=True)
        return jsonify({'error': 'Analysis engine failure'}), 500

# 文本预处理（参考网页6的数据处理逻辑）
def preprocess_text(text):
    import re
    from nltk.sentiment import SentimentIntensityAnalyzer
    # 清洗特殊字符
    clean_text = re.sub(r'[^\w\s]', '', text)
    # 情感分析器初始化
    return clean_text.lower().strip()

# 情感计算（扩展网页3的验证逻辑）
def calculate_sentiment(text):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)['compound']

# 特征工程（结合网页7的数据接口方案）
def build_prediction_features(sentiment):
    global data
    market_features = data[['bitcoin_price', 'bitcoin_volume']].tail(1).values
    return np.concatenate([market_features, [[sentiment]]], axis=1)

# 影响分类（自定义业务逻辑）
def classify_impact(sentiment, volatility):
    if sentiment < -0.5 and volatility > 0.3:
        return "High Negative Impact"
    elif sentiment > 0.5 and volatility < 0.1:
        return "Stable Positive"
    return "Neutral"

if __name__ == '__main__':
    update_data() 
    data.to_csv('final.csv')
    app.run(debug=True, use_reloader=False, port=8001, host='0.0.0.0')