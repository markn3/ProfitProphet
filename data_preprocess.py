import yfinance as yf
import pandas as pd
import seaborn as sns
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

    # Calculate the bounds
    lower_bound = current_price * 0.9
    upper_bound = current_price * 1.1

    volume_profile_calls = volume_profile_calls[(volume_profile_calls.index >= lower_bound) & (volume_profile_calls.index <= upper_bound)]
    volume_profile_puts = volume_profile_puts[(volume_profile_puts.index >= lower_bound) & (volume_profile_puts.index <= upper_bound)]

    volume_profile_calls = volume_profile_calls.reset_index()
    volume_profile_puts = volume_profile_puts.reset_index()
    
    # Apply the default theme
    sns.set_theme()


    # Create a horizontal barplot for calls
    sns.barplot(x='volume', y='strike', data=volume_profile_calls, color='blue', label='Calls', orient='h', alpha=0.5)

    # Create a horizontal barplot for puts
    sns.barplot(x='volume', y='strike', data=volume_profile_puts, color='red', label='Puts', orient='h', alpha=0.5)

    # Provide labels
    plt.title('Options Volume Profile')
    plt.xlabel('Volume')
    plt.ylabel('Strike Price')

    # Reverse the y-axis
    plt.gca().invert_yaxis()

    # Add a legend
    plt.legend()

    # Display the plot
    plt.show()