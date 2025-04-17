import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# 加载数据
data_path = os.path.join(os.path.dirname(__file__), 'Data_set', 'final_data.csv')
df = pd.read_csv(data_path)
all_features = [
    "timestamp",
    "bitcoin_price",
    "bitcoin_volume",
    "bitcoin_market_cap",
    "bitcoin_returns",
    "bitcoin_volatility",
    "score",
    "No_Comments",
    "title_sentiment",
    "description_sentiment",
    "score_lag1",
    "score_lag2",
    "score_lag3",
    "score_lag4",
    "score_lag5",
    "score_lag6",
    "score_lag7",
    "No_Comments_lag1",
    "No_Comments_lag2",
    "No_Comments_lag3",
    "No_Comments_lag4",
    "No_Comments_lag5",
    "No_Comments_lag6",
    "No_Comments_lag7",
    "title_sentiment_lag1",
    "title_sentiment_lag2",
    "title_sentiment_lag3",
    "title_sentiment_lag4",
    "title_sentiment_lag5",
    "title_sentiment_lag6",
    "title_sentiment_lag7",
    "description_sentiment_lag1",
    "description_sentiment_lag2",
    "description_sentiment_lag3",
    "description_sentiment_lag4",
    "description_sentiment_lag5",
    "description_sentiment_lag6",
    "description_sentiment_lag7",
    "sentiment",
    "sentiment1",
    "sentiment2",
    "sentiment3",
    "sentiment4",
    "sentiment5",
    "sentiment6",
    "sentiment7",
    "total_sentiment"
]

features = all_features.copy()
features.remove("timestamp")  # 一般 timestamp 不用于训练
features.remove("bitcoin_volatility")  # 这是你要预测的目标
target = "bitcoin_volatility"

X = df[features]
y = df[target]

# 拆分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 训练随机森林模型
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 保存模型
joblib.dump(model, './Models/best_random_forest_model.joblib')
