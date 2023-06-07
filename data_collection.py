import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np


amd = yf.Ticker("AMD")

data = yf.download(tickers='AMD', period='30d', interval='15m')

# Get the opening price for the day
# open_price = yf.download('AMD', start='2023-06-06', end='2023-06-07')
# open_price = open_price.iloc[0, 0]
# print("value: ", open_price)

# # Calculate the 10% range
# lower_bound = open_price * 0.9
# upper_bound = open_price * 1.1

# print("Upper bound: ", upper_bound)
# print("Lower Bound: ", lower_bound)

# Calculate volume profile for each price level
volume_profile = data.groupby('Close').Volume.sum()

# append volume profile to dataset
data['VolumeProfile'] = data['Close'].map(volume_profile)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)


def create_sequences(data, seq_length):
    xs = []
    ys = []
  
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

seq_length = 60  # choose sequence length
X, y = create_sequences(scaled_data, seq_length)

train_size = int(len(X) * 0.7)  # 70% of data for training

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]




# ideas:
# show age of point level by opacity