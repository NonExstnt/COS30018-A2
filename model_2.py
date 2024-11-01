"""
Definition of NN model
"""
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN
from tensorflow.keras.models import Sequential


def get_lstm(units):
    """LSTM(Long Short-Term Memory)
    Build LSTM Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(LSTM(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(LSTM(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model


def get_gru(units):
    """GRU(Gated Recurrent Unit)
    Build GRU Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(GRU(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(GRU(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model


def get_ae(x_train, input_output, hidden_sizes, last_ae): #num_hidden later
    """SAE(Auto-Encoders)
    Build SAE Model.

    # Arguments
        input_dim: Integer, number of predictor variables.
        inputs: Integer, number of input units.
        hidden: Integer, number of hidden units.
        num_hidden: Integer, number of hidden layers.
    # Returns
        model: Model, nn model.
    """
    input_dim = x_train[0].shape[0]

    model = Sequential()

    # Input Layer
    model.add(Dense(input_output, input_dim=input_dim, activation="relu", name="input"))

    # Hidden layers (sequential reduction)
    for i, size in enumerate(hidden_sizes):
        model.add(Dense(size, activation="relu", name=f"hidden_{i+1}"))

    # Output Layer
    if last_ae:
        model.add(Dense(1, activation="sigmoid", name="output"))
    else:
        model.add(Dense(input_output, activation="sigmoid", name="output"))

    return model


def get_rnn(units):
    """SRNN(Simple recurrent neural network)
    Build SRNN Model.
    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    # Create a Sequential model
    model = Sequential()

    # Add the first SimpleRNN Layer with unit count based on unit's input param
    model.add(SimpleRNN(units[1], input_shape=(units[0], 1), return_sequences=True))

    # Add the second SimpleRNN Layer with unit count based on unit's input param
    model.add(SimpleRNN(units[2]))

    # Add a Dropout layer of 0.2
    model.add(Dropout(0.2))

    # Add the output Layer with unit count based on unit's input parameter
    model.add(Dense(units[3], activation='sigmoid'))

    # Return the model from function scope
    return model
