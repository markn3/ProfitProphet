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

def get_options(tSym):
    'define ticker symbol'
    stock = yf.Ticker(tSym)

    expirations = stock.options
    print(stock.options)
    print(expirations[2])

    options = stock.option_chain(f'{expirations[2]}')

    print("OPTIONS: ", options)

    calls = options.calls
    puts = options.puts

    # Calculate volume profile for each strike price for calls
    volume_profile_calls = calls.groupby('strike').volume.sum()

    # Calculate volume profile for each strike price for puts
    volume_profile_puts = puts.groupby('strike').volume.sum()


    current_price = stock.history().tail(1)['Close'].values[0]
    print("Current price: ", current_price)

    # Calculate the bounds
    lower_bound = current_price * 0.9
    upper_bound = current_price * 1.1

    filtered_volume_profile = volume_profile_calls[(volume_profile_calls.index >= lower_bound) & (volume_profile_calls.index <= upper_bound)]

    print("Volkuyme ", filtered_volume_profile)

    plt.figure(figsize=(10,5))
    plt.barh(filtered_volume_profile.index, filtered_volume_profile.values, height=0.5)
    plt.xlabel("Volume")
    plt.ylabel("Price")
    plt.title("Volume Profile")
    plt.grid(True)
    plt.show()



    # return opt
