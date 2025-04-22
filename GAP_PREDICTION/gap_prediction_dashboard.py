# gap_prediction_dashboard.py

import streamlit as st
import pandas as pd
import datetime
from sklearn.ensemble import RandomForestClassifier
import joblib


# 1. Load model (pretrained or placeholder)
def load_model():
    try:
        model = joblib.load("gap_prediction_model.pkl")
    except:
        model = RandomForestClassifier()
        import numpy as np

        X_dummy = np.array([[1, 0.3, 1.1, 85, 120, 1],
                            [-1, -0.2, 0.8, 65, -90, -1],
                            [-1, -0.4, 0.9, 10, -50, -1]])
        y_dummy = np.array(["Gap Up", "Gap Down", "Gap Down"])

        model.fit(X_dummy, y_dummy)
    return model


# 2. Fetch Data (Dummy for now - Replace with real-time scrapers or API integrations)
def fetch_stock_data():
    # Sample data to mimic real features
    data = pd.DataFrame([
        {"symbol": "NIFTY", "oi_change": 1, "iv_change": 0.3, "pcr": 1.1, "sgx_nifty_diff": 85, "dow_futures": 120,
         "price_vs_cpr": 1},
        {"symbol": "BANKNIFTY", "oi_change": -1, "iv_change": -0.2, "pcr": 0.8, "sgx_nifty_diff": 65,
         "dow_futures": -90, "price_vs_cpr": -1},
        {"symbol": "RELIANCE", "oi_change": -1, "iv_change": -0.4, "pcr": 0.9, "sgx_nifty_diff": 10, "dow_futures": -50,
         "price_vs_cpr": -1},
    ])
    return data


# 3. Predict Gap Bias
def predict_gap(data, model):
    features = data.drop(columns=["symbol"])
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    data["gap_bias"] = predictions
    data["confidence"] = probabilities.max(axis=1)
    return data


# 4. UI with Streamlit
st.set_page_config(page_title="Gap Prediction Dashboard", layout="wide")
st.title("📊 Gap Up / Gap Down Predictor - F&O Universe")

st.markdown("Runs daily after 3 PM. Uses ML model to predict next-day opening bias.")

# Main Workflow
model = load_model()
data = fetch_stock_data()
results = predict_gap(data, model)

# Render Table
st.dataframe(results.style.format({"confidence": "{:.2%}"}))
