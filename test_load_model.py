import os
import joblib
import numpy as np

model_path = os.path.join(os.path.dirname(__file__), 'Models', 'best_random_forest_model.joblib')
model = joblib.load(model_path)


sample = np.array([[96561.66, 29084371910.69, 1914315097486.38, -0.0015, 17.63, -0.59, -0.86, -0.72]])
prediction = model.predict(sample)

print("预测波动率：", prediction[0])
