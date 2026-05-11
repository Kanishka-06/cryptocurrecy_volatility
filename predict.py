import pandas as pd
import numpy as np
import joblib

from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

# =========================
# LOAD TRAINED MODEL
# =========================

model = joblib.load("volatility_model.pkl")

# =========================
# LOAD HISTORICAL DATA
# =========================

data = pd.read_csv("processed_btc_data.csv")
# Convert columns to numeric
numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')
# =========================
# USER INPUTS
# =========================

open_price = float(input("Enter Open Price: "))
high_price = float(input("Enter High Price: "))
low_price = float(input("Enter Low Price: "))
close_price = float(input("Enter Close Price: "))
volume = float(input("Enter Volume: "))

# =========================
# CREATE NEW ROW
# =========================

new_row = {
    'Open': open_price,
    'High': high_price,
    'Low': low_price,
    'Close': close_price,
    'Volume': volume
}

# Add new row to dataframe
data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)

# =========================
# CALCULATE RETURNS
# =========================

data['Returns'] = data['Close'].pct_change(fill_method=None)

# =========================
# CALCULATE VOLATILITY
# =========================

data['Volatility'] = data['Returns'].rolling(window=7).std()

# =========================
# RSI
# =========================

rsi = RSIIndicator(close=data['Close'], window=14)
data['RSI'] = rsi.rsi()

# =========================
# MACD
# =========================

macd = MACD(close=data['Close'])

data['MACD'] = macd.macd()

# =========================
# BOLLINGER BANDS
# =========================

bb = BollingerBands(close=data['Close'], window=20)

data['BB_High'] = bb.bollinger_hband()
data['BB_Low'] = bb.bollinger_lband()

# =========================
# CREATE LAG FEATURES
# =========================

data['Volatility_Lag1'] = data['Volatility'].shift(1)
data['Volatility_Lag2'] = data['Volatility'].shift(2)
data['Volatility_Lag3'] = data['Volatility'].shift(3)

data['Return_Lag1'] = data['Returns'].shift(1)
data['Return_Lag2'] = data['Returns'].shift(2)

# =========================
# GET LATEST ROW
# =========================

latest = data.iloc[-1]

# =========================
# CREATE FEATURE ARRAY
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
# PREDICT VOLATILITY
# =========================

prediction = model.predict(features)

# =========================
# DISPLAY RESULTS
# =========================

print("\n========== GENERATED FEATURES ==========")

print(f"RSI: {latest['RSI']:.2f}")
print(f"MACD: {latest['MACD']:.2f}")

print(f"BB High: {latest['BB_High']:.2f}")
print(f"BB Low: {latest['BB_Low']:.2f}")

print(f"Volatility Lag1: {latest['Volatility_Lag1']:.6f}")
print(f"Volatility Lag2: {latest['Volatility_Lag2']:.6f}")
print(f"Volatility Lag3: {latest['Volatility_Lag3']:.6f}")

print(f"Return Lag1: {latest['Return_Lag1']:.6f}")
print(f"Return Lag2: {latest['Return_Lag2']:.6f}")

print("\n========== PREDICTION ==========")

print(f"Predicted Volatility: {prediction[0]:.6f}")

# =========================
# MARKET CONDITION
# =========================

vol = prediction[0]

if vol < 0.01:
    print("Market Condition: Low Volatility")

elif vol < 0.03:
    print("Market Condition: Moderate Volatility")

else:
    print("Market Condition: High Volatility")