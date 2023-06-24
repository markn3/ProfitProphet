import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from data_preprocess import preprocess

print("Preprocessing")
X_train, X_test, y_train, y_test, scaler = preprocess()
print("preprocess done")

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile and train the model
print("Compiling")
model.compile(optimizer='adam', loss='mean_squared_error')
print("Training")
model.fit(X_train, y_train, batch_size=64, epochs=200)

print("multi-step forecasting")
# Perform multi-step forecasting
n_steps = 5  # number of steps to predict into the future
input_seq = X_train[-1]  # start with the last sequence from the training data
predictions = []

for _ in range(n_steps):
    # Make sure your input sequence is in the correct shape for your model
    print(input_seq.shape)
    input_seq = input_seq.reshape((1, input_seq.shape[0], input_seq.shape[1]))
    
    # Make a prediction
    prediction = model.predict(input_seq)
    
    # Append the prediction to the list of predictions
    predictions.append(prediction[0, 0])
    
    # Repeat the prediction to match the number of features
    prediction_repeated = np.repeat(prediction, input_seq.shape[2]).reshape(1, -1)
    
    # Remove the first time step from the input sequence and append the prediction
    input_seq = np.concatenate((input_seq[0, 1:, :], prediction_repeated), axis=0)
    
# 'predictions' now contains the predicted values for the next 'n_steps' steps

# Create a dummy array with the same number of features as your original data
dummy_array = np.zeros((len(predictions), 5))

# Replace the column corresponding to the closing prices with your predictions
# Here, I'm assuming that 'Close' was the 4th column in your original data
dummy_array[:, 3] = predictions

# Rescale the predictions to the original price range
rescaled_predictions = scaler.inverse_transform(dummy_array)[:, 3]

print("Predictions: ", predictions)
print(f"Re-scaled predictions: ", rescaled_predictions)
print("Done")