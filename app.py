import streamlit as st
import pandas as pd
import numpy as np
import joblib

from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

# =========================
# LOAD MODEL
# =========================

model = joblib.load("volatility_model.pkl")

# =========================
# LOAD HISTORICAL DATA
# =========================

data = pd.read_csv("processed_btc_data.csv")

# Convert numeric columns
numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Crypto Volatility Predictor",
    # page_icon="",
    layout="centered"
)

# =========================
# TITLE
# =========================

st.title(" Cryptocurrency Volatility Prediction")

st.write("""
Enter OHLCV market values below.
The system will automatically calculate
technical indicators and predict volatility.
""")

# =========================
# USER INPUTS
# =========================

open_price = st.number_input(
    "Open Price",
    # value=65000.0
)

high_price = st.number_input(
    "High Price",
    # value=66500.0
)

low_price = st.number_input(
    "Low Price",
    # value=64500.0
)

close_price = st.number_input(
    "Close Price",
    # value=66000.0
)

volume = st.number_input(
    "Volume",
    # value=35000000.0
)

# =========================
# PREDICT BUTTON
# =========================

if st.button("Predict Volatility"):

    # =========================
    # CREATE NEW INPUT ROW
    # =========================

    new_row = {
        'Open': open_price,
        'High': high_price,
        'Low': low_price,
        'Close': close_price,
        'Volume': volume
    }

    # Add new row
    updated_data = pd.concat(
        [data, pd.DataFrame([new_row])],
        ignore_index=True
    )

    # =========================
    # CALCULATE RETURNS
    # =========================

    updated_data['Returns'] = (
        updated_data['Close']
        .pct_change(fill_method=None)
    )

    # =========================
    # CALCULATE VOLATILITY
    # =========================

    updated_data['Volatility'] = (
        updated_data['Returns']
        .rolling(window=7)
        .std()
    )

    # =========================
    # RSI
    # =========================

    rsi = RSIIndicator(
        close=updated_data['Close'],
        window=14
    )

    updated_data['RSI'] = rsi.rsi()

    # =========================
    # MACD
    # =========================

    macd = MACD(
        close=updated_data['Close']
    )

    updated_data['MACD'] = macd.macd()

    # =========================
    # BOLLINGER BANDS
    # =========================

    bb = BollingerBands(
        close=updated_data['Close'],
        window=20
    )

    updated_data['BB_High'] = (
        bb.bollinger_hband()
    )

    updated_data['BB_Low'] = (
        bb.bollinger_lband()
    )

    # =========================
    # CREATE LAG FEATURES
    # =========================

    updated_data['Volatility_Lag1'] = (
        updated_data['Volatility'].shift(1)
    )

    updated_data['Volatility_Lag2'] = (
        updated_data['Volatility'].shift(2)
    )

    updated_data['Volatility_Lag3'] = (
        updated_data['Volatility'].shift(3)
    )

    updated_data['Return_Lag1'] = (
        updated_data['Returns'].shift(1)
    )

    updated_data['Return_Lag2'] = (
        updated_data['Returns'].shift(2)
    )

    # =========================
    # GET LATEST ROW
    # =========================

    latest = updated_data.iloc[-1]

    # =========================
    # FEATURE ARRAY
    # =========================

    features = np.array([[
        latest['Open'],
        latest['High'],
        latest['Low'],
        latest['Close'],
        latest['Volume'],
        latest['RSI'],
        latest['MACD'],
        latest['BB_High'],
        latest['BB_Low'],
        latest['Volatility_Lag1'],
        latest['Volatility_Lag2'],
        latest['Volatility_Lag3'],
        latest['Return_Lag1'],
        latest['Return_Lag2']
    ]])

    # =========================
    # PREDICTION
    # =========================

    prediction = model.predict(features)

    volatility = prediction[0] * 100

    # =========================
    # SHOW RESULT
    # =========================

    st.success(
        f"Predicted Volatility: {volatility:.2f}%"
    )

    # =========================
    # MARKET CONDITION
    # =========================

    if prediction[0] < 0.01:
        st.success(
            "Market Condition: Low Volatility"
        )

    elif prediction[0] < 0.03:
        st.warning(
            "Market Condition: Moderate Volatility"
        )

    else:
        st.error(
            "Market Condition: High Volatility"
        )

    # =========================
    # SHOW GENERATED FEATURES
    # =========================

    st.subheader(
        "Generated Technical Indicators"
    )

    st.write(f"RSI: {latest['RSI']:.2f}")

    st.write(f"MACD: {latest['MACD']:.2f}")

    st.write(
        f"BB High: {latest['BB_High']:.2f}"
    )

    st.write(
        f"BB Low: {latest['BB_Low']:.2f}"
    )

    st.write(
        f"Volatility Lag1: "
        f"{latest['Volatility_Lag1']:.6f}"
    )

    st.write(
        f"Return Lag1: "
        f"{latest['Return_Lag1']:.6f}"
    )

    # =========================
    # OPTIONAL CHART
    # =========================

    st.subheader("Price Overview")

    chart_data = updated_data[['Close']].tail(30)

    st.line_chart(chart_data)

# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption(
    "AI-powered Cryptocurrency Volatility Prediction System",
    
)