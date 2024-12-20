"""
Train the NN model.
"""
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from data.data import process_data
from model import model
from keras.callbacks import EarlyStopping
warnings.filterwarnings("ignore")


def train_model(model, X_train, y_train, name, config, scat_number, lane_number):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    if name != 'saes':
        # Compile the model with mean squared error loss and RMSprop optimizer
        model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])

        # Train the model with early stopping (commented out) and validation split
        hist = model.fit(
            X_train, y_train,
            batch_size=config["batch"],
            epochs=config["epochs"],
            validation_split=0.05
        )

        # Save the trained model to a file
        model.save(f'model/{name}/{scat_number}/{lane_number}.h5')

        # Save the training history to a CSV file
        df = pd.DataFrame.from_dict(hist.history)
        df.to_csv(f'model/{name}/{scat_number}/{lane_number} loss.csv', encoding='utf-8', index=False)
    else:
        # Save the first model in the list (for 'saes')
        model[0].save(f'model/{name}/{scat_number}/{lane_number}.h5')

        # Save the training history of the second model in the list to a CSV file
        df = pd.DataFrame.from_dict(model[1].history)
        df.to_csv(f'model/{name}/{scat_number}/{lane_number} loss.csv', encoding='utf-8', index=False)


def train_saes(x_train, y_train, name, config, num_ae, hidden_sizes, scat_number, lane_number):
    """train
    train the SAEs model.

    # Arguments
        models: List, list of SAE model.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    saes = []  # List to store the last autoencoder and it's training history
    last_ae = False  # Flag to indicate if the current autoencoder is the last one

    for i in range(num_ae):
        print(str(i) + ": iterations in train_saes\n")
        if i == 0:  # First iteration uses x_train as input
            ae_input = x_train
            last_ae = False
            if i == (num_ae - 1):  # If the first autoencoder is also the last one
                last_ae = True
        elif i == (num_ae - 1):  # If the current autoencoder is the last one
            last_ae = True
        else:
            last_ae = False

        # Get the autoencoder model
        ae = model.get_ae(ae_input, 12, hidden_sizes, last_ae)
        ae.compile(loss="mse", optimizer="adam", metrics=['mape'])  # Compile the autoencoder model
        stack = ae.fit(ae_input, ae_input, batch_size=config["batch"], 
                       epochs=config["epochs"], validation_split=0.05)  # Train the autoencoder
        ae_input = ae.predict(ae_input)  # Use the output of the current autoencoder as input for the next

    ae.summary()  # Print the summary of the last autoencoder

    saes.append(ae)  # Append the last autoencoder to the list
    saes.append(stack)  # Append the last autoencoder trained to the list

    # Train the final model using the trained autoencoders
    train_model(saes, x_train, y_train, name, config, scat_number, lane_number)



def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", 
        default="lstm", 
        help="Model to train.")
    parser.add_argument(
        "--lane", 
        default=1, 
        type=int, 
        help="Lane to use.")
    parser.add_argument(
        "--scat", 
        default=970, 
        type=int, 
        help="SCAT site to train.")
    parser.add_argument(
        "--aes",
        default=3,
        type=int,
        help="Number of Auto Encoders in the SAE.")
    parser.add_argument(
        "--hiddenSizes",
        type=str,
        default="32,32",
        help="Comma-separated list of hidden layer sizes for SAE. Number of values = Number of hidden layers.")
    args = parser.parse_args()

    lag = 12
    config = {"batch": 256, "epochs": 20}
    
    lane_no = args.lane
    scat_number = args.scat

    # Set parameters
    lag = 12
    file1 = f"data/Scat_number_{scat_number}_train.csv"
    file2 = f"data/Scat_number_{scat_number}_test.csv"
    X_train, y_train, _, _, _ = process_data(file1, file2, lag, args.lane)

    

    if args.model == 'lstm':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_lstm([12, 64, 64, 1])
        train_model(m, X_train, y_train, args.model, config, scat_number, lane_no)
    if args.model == 'gru':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_gru([12, 64, 64, 1])
        train_model(m, X_train, y_train, args.model, config, scat_number, lane_no)
    if args.model == 'saes':
        hidden_sizes = list(map(int, args.hiddenSizes.split(',')))
        num_aes = int(args.aes)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
        train_saes(X_train, y_train, args.model,
                    config, num_aes, hidden_sizes,
                    scat_number, lane_no)
    if args.model == 'rnn':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_rnn([12, 64, 64, 1])
        train_model(m, X_train, y_train, args.model, config, scat_number, lane_no)


if __name__ == '__main__':
    main(sys.argv)
