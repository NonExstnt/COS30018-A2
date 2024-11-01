import  keras
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, RNN
from tensorflow.keras.optimizers import Adam


def Keras(model_type='dense', input_shape=None, layers=None, optimizer='adam', loss='mse', metrics=None):
    """
    A function to create and compile a Keras model.

    Parameters:
    - model_type: Type of model, e.g., 'dense', 'lstm', or 'gru'.
    - input_shape: Shape of input data.
    - layers: List of integers representing the number of units in each layer.
    - optimizer: Optimizer for the model.
    - loss: Loss function.
    - metrics: List of metrics to evaluate.

    Returns:
    - Compiled Keras model.
    """
    model = Sequential()

    if layers is None:
        layers = [64, 32]  # Default layers if none specified

    if model_type == 'dense':
        model.add(Dense(layers[0], activation='relu', input_shape=input_shape))
        for units in layers[1:]:
            model.add(Dense(units, activation='relu'))
    elif model_type == 'lstm':
        model.add(LSTM(layers[0], return_sequences=True, input_shape=input_shape))
        for units in layers[1:-1]:
            model.add(LSTM(units, return_sequences=True))
        model.add(LSTM(layers[-1]))
    elif model_type == 'gru':
        model.add(GRU(layers[0], return_sequences=True, input_shape=input_shape))
        for units in layers[1:-1]:
            model.add(GRU(units, return_sequences=True))
        model.add(GRU(layers[-1]))
    else:
        raise ValueError("Unsupported model type. Choose from 'dense', 'lstm', or 'gru'.")

    model.add(Dense(1))  # Output layer

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics if metrics else ['mae'])
    return model
