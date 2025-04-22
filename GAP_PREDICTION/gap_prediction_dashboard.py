import streamlit as st
import pandas as pd
import joblib
import datetime
import numpy as np

# Load the trained model and encoder
model = joblib.load("nse_data/gap_model.pkl")
label_encoder = joblib.load("nse_data/label_encoder.pkl")

# Load today's features from previously generated daily data
@st.cache_data
def load_today_features():
    try:
        df = pd.read_csv("nse_data/latest_intraday_features.csv")
        df['DATE'] = pd.to_datetime(df['DATE'])
        return df
    except FileNotFoundError:
        return pd.DataFrame()

def predict(df):
    features = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VWAP', 'TOTTRDQTY', 'PUT_OI', 'CALL_OI', 'PCR']
    df["prediction"] = model.predict(df[features])
    df["probabilities"] = model.predict_proba(df[features]).tolist()
    df["GAP_LABEL"] = label_encoder.inverse_transform(df["prediction"])
    df[["PROB_GAP_DOWN", "PROB_FLAT", "PROB_GAP_UP"]] = pd.DataFrame(df["probabilities"].to_list(), index=df.index)
    return df

st.title("🔮 NSE Gap Up / Down Prediction Dashboard")
st.markdown("Model-based probability forecast for the next trading day's open gap")

# Load today's feature data
today_data = load_today_features()

if today_data.empty:
    st.warning("Today's feature data not found. Please ensure the latest data is generated and saved at 'nse_data/latest_intraday_features.csv'")
else:
    predictions = predict(today_data)

    # Filter to only Nifty, Banknifty, and Nifty50 symbols
    symbols = ['NIFTY', 'BANKNIFTY', 'RELIANCE', 'HDFCBANK', 'ICICIBANK', 'INFY', 'SBIN', 'TCS', 'LT', 'AXISBANK', 'ITC', 'KOTAKBANK']
    predictions = predictions[predictions['SYMBOL'].isin(symbols)]

    # Display table
    st.subheader("📈 Predictions")
    display_cols = ['SYMBOL', 'GAP_LABEL', 'PROB_GAP_UP', 'PROB_GAP_DOWN', 'PROB_FLAT']
    st.dataframe(predictions[display_cols].sort_values(by='PROB_GAP_UP', ascending=False).reset_index(drop=True))

    # Download option
    csv = predictions[display_cols].to_csv(index=False)
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv,
        file_name="gap_predictions.csv",
        mime="text/csv",
    )
