
import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="BTC Price Predictor", layout="wide")
st.title("📈 Real-Time Bitcoin Price Predictor (5-Min Ahead)")

# Fetch BTC 1-minute OHLCV data (last 60 minutes)
def fetch_binance_ohlcv():
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "1m",
        "limit": 60
    }
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

# Simulate a trained LSTM model
@st.cache_resource
def create_dummy_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Prepare data for LSTM
def prepare_data(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['close']])

    x = []
    for i in range(len(scaled_data) - 60):
        x.append(scaled_data[i:i+60])
    x = np.array(x)
    return x[-1:], scaler

# MAIN
try:
    df = fetch_binance_ohlcv()
    st.subheader("Live BTC Price: $")
    st.metric("Current Price", f"{df['close'].iloc[-1]:,.2f}")

    st.line_chart(df['close'])

    x_input, scaler = prepare_data(df)
    model = create_dummy_model((60, 1))
    pred_scaled = model.predict(x_input)
    predicted_price = scaler.inverse_transform(pred_scaled)[0][0]

    st.subheader("Predicted BTC Price (5 min later):")
    st.success(f"${predicted_price:,.2f}")

    st.subheader("Prediction vs Current")
    fig, ax = plt.subplots()
    ax.plot([0, 1], [df['close'].iloc[-1], predicted_price], marker='o')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Now', 'In 5 min'])
    ax.set_title("Price Prediction")
    st.pyplot(fig)

except Exception as e:
    st.error(f"Failed to fetch or process data: {e}")
