import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

#     # append volume profile to dataset
#     # data['VolumeProfile'] = data['Close'].map(volume_profile)
# test
def preprocess():
    data = yf.download(tickers='AMD', period='30d', interval='15m')
    data = data.drop(columns=['Adj Close'])

    print("Data: ", data)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    def create_sequences(data, seq_length):
        xs = []
        ys = []
    
        for i in range(len(data)-seq_length-1):
            x = data[i:(i+seq_length)]
            y = data[i+seq_length, 3]  # get the closing price
            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)

    seq_length = 10  # choose sequence length
    X, y = create_sequences(scaled_data, seq_length)

    train_size = int(len(X) * 0.7)  # 70% of data for training

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 5))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 5))
    return X_train, X_test, y_train, y_test, scaler
