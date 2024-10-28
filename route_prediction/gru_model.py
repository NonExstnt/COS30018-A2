import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime

import pandas as pd
import numpy as np

time_features = ['0:00', '0:15', '0:30', '0:45', '1:00', '1:15', '1:30', '1:45', '2:00', '2:15', '2:30', '2:45',
                 '3:00', '3:15', '3:30', '3:45', '4:00', '4:15', '4:30', '4:45', '5:00', '5:15', '5:30', '5:45',
                 '6:00', '6:15', '6:30', '6:45', '7:00', '7:15', '7:30', '7:45', '8:00', '8:15', '8:30', '8:45',
                 '9:00', '9:15', '9:30', '9:45', '10:00', '10:15', '10:30', '10:45', '11:00', '11:15', '11:30',
                 '11:45', '12:00', '12:15', '12:30', '12:45', '13:00', '13:15', '13:30', '13:45', '14:00', '14:15',
                 '14:30', '14:45', '15:00', '15:15', '15:30', '15:45', '16:00', '16:15', '16:30', '16:45', '17:00',
                 '17:15', '17:30', '17:45', '18:00', '18:15', '18:30', '18:45', '19:00', '19:15', '19:30', '19:45',
                 '20:00', '20:15', '20:30', '20:45', '21:00', '21:15', '21:30', '21:45', '22:00', '22:15', '22:30',
                 '22:45', '23:00', '23:15', '23:30', '23:45']

def get_time_series_for_site(site_number, scats_site_data):
    # Get only the SCATS traffic data for one site
    entries = scats_site_data.copy().loc[scats_site_data['SCATS Number'] == site_number]

    # Convert the Date column into a datetime type column (used in time series data)
    entries["Date"] = pd.to_datetime(entries["Date"])

    # Remove duplicate dates
    entries = entries.drop_duplicates(subset="Date")

    entries = entries.set_index(entries["Date"])

    del entries["Date"]

    entries = entries.reindex(
        pd.date_range(
            datetime(2006, 10, 1),
            datetime(2006, 10, 31),
            freq='1D'
        )
    )

    entries = entries.fillna(0)

    # Sort all entries by date
    entries = entries.sort_index()
    start_date = entries.index.min()
    entries_times = []

    # For all days in the recording of the SCATS site
    for i in range(0, len(entries) - 1):
        # Extract every 15 minutes worth of data
        time_data = entries.iloc[i][time_features]

        time_data = time_data.values

        entries_times.append(time_data)

    # Create an array of all traffic volumes for all times and dates for the scats site
    time_data_raw = np.array(entries_times).flatten()

    # Create a time index which has timestamps for every entry in time_data_raw
    time_index = pd.date_range(start=start_date, periods=len(time_data_raw), freq='15min')

    time_data = pd.DataFrame(time_data_raw, index=time_index)

    return time_data

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()

    # Convert data into array-like if we're working with a DataFrame
    if isinstance(sequence, pd.DataFrame):
        sequence = sequence.values

    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)


def train_test_split(data, train_size : float = 0.7, window_size_minutes = 240):
    """
    Split a dataset into train and test datasets using sliding window sampling.

    Args:
        data (DataFrame | ndarray): The dataset
        train_size (float): Percentage of data to use for the training dataset.
        window_size_minutes (int): How long each window size should be.

    Returns:
        X_train
        y_train
        X_test
        y_test
    """
    if (train_size <= 0 or train_size >= 1): raise ValueError("Train size must be a ratio between 0 and 1")

    # Convert data from DataFrame to array if necessary
    if isinstance(data, pd.DataFrame):
        data = data.values

    train_samples = len(data) / train_size # Floor division
    train_samples = int(round(train_samples))

    train_data = data[:train_samples]
    test_data = data[train_samples:]

    window_size = window_size_minutes // 15

    X_train, y_train = split_sequence(train_data, window_size)
    X_test, y_test = split_sequence(test_data, window_size)

    return X_train, y_train, X_test, y_test
    #return train_data, test_data

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)  # GRU returns (output, hidden state)
        out = self.fc(out[:, -1, :])  # Use the last output for prediction
        return out

class GRUTrainer():
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.0001):
        self.gru_model = GRUModel(input_size, hidden_size, output_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.gru_model.parameters(), lr=learning_rate)

    def save(self, save_path : str):
        # Create the save path if it does not exist.

        folder_to_save = os.path.split(save_path)[0]
        if not os.path.exists(folder_to_save):
            os.makedirs(folder_to_save)

        torch.save(self.gru_model.state_dict(), save_path) # Save the model weights.
        # torch.save(self.optimizer.state_dict(), os.path.join(save_path, 'gru_optimizer.pth'))

    def load(self, load_path : str):
        self.gru_model.load_state_dict(torch.load(load_path, weights_only = True)) # Load the model weights
        # self.optimizer = torch.load(os.path.join(load_path, 'gru_optimizer.pth'))
        self.gru_model.eval() # set model to evaluation mode (This prevents weights from being modified, effectively freezing the model)

    def train_gru(self, X_train, y_train, epochs=1):
        # Convert numpy arrays to torch tensors
        X_train = torch.tensor(X_train.astype("float32"))
        y_train = torch.tensor(y_train.astype("float32"))


        batch_size = X_train.shape[1]
        seq_length = X_train.shape[0]
        input_size = X_train.shape[2]

        self.gru_model.train()

        prog_bar = tqdm(range(epochs), desc="Training GRU model", unit='epoch')

        for epoch in prog_bar:
            self.optimizer.zero_grad()

            # Forward pass
            output = self.gru_model(X_train)

            # Ensure that X_train and y_train are PyTorch tensors
            if not isinstance(X_train, torch.Tensor):
                X_train = torch.tensor(X_train, dtype=torch.float32)
            if not isinstance(y_train, torch.Tensor):
                y_train = torch.tensor(y_train, dtype=torch.float32)

            # Compute loss
            loss = self.criterion(output, y_train)
            loss.backward()
            self.optimizer.step()
            #
            # try:
            #     prog_bar.set_postfix(f'Loss: {loss.item()[0]:.4f}')
            # except Exception:
            #     pass # doesnt matter.
            #print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    def predict(self, X_test):
        X_test = torch.tensor(X_test.astype("float32"))
        self.gru_model.eval()
        with torch.no_grad():
            y_pred = self.gru_model(X_test)
        return y_pred

def get_site_dataset(all_site_data, site_number : int, train_size : float, window_size_mins:int):

    site_data = get_time_series_for_site(site_number, all_site_data)

    X_train, y_train, X_test, y_test = train_test_split(site_data, train_size=train_size, window_size_minutes=window_size_mins)

    return X_train, y_train, X_test, y_test

def train_gru_for_scats_site(all_site_data, site_number : int, save_directory : str, window_size_mins = 240, train_size = 0.7, epochs=50, learning_rate = 0.0001):
    """
    Train a GRU model on a particular SCATS site.

    Args:
        all_site_data (ndarray | DataFrame): The total SCATS site dataset.
        site_number (int): Which SCATS site traffic data to train the GRU model on.
        save_directory (str): Which folder to save the model weights. Weights are saved as a file, e.g., "900.pt". The file name is the site number.
        train_size (float, optional): Percentage of data to use for the training dataset.
        window_size_minutes (int, optional): How long each window size should be.
        epochs (int, optional): How many epochs to train the GRU model for. This controls how many times the GRU model is trained over the entire dataset.
        learning_rate (float, optional): How much the loss function should affect the model weights. This is an important hyperparameter.
    """
    X_train, y_train, X_test, y_test = get_site_dataset(all_site_data, site_number, train_size, window_size_mins)

    gru_trainer = GRUTrainer(input_size=1, hidden_size=1984, output_size=1, learning_rate=learning_rate)

    # Train the model (X_train has shape [1984, 16, 1])
    gru_trainer.train_gru(X_train, y_train, epochs=epochs)

    # Set the save path to format: "gru_weights/900.pt" where "900" is the site number.
    save_path = os.path.join(save_directory, f"weights-{site_number}.pt")

    # Save the model weights.
    gru_trainer.save(save_path)

def train_gru_for_all_scats_sites(all_site_data, save_directory : str, window_size_mins = 240, train_size = 0.7, epochs=50, learning_rate = 0.0001):
    """
    Super GRU training function. Trains GRU models for *all* SCATS sites in all_site_data.

   Args:
        all_site_data (ndarray | DataFrame): The total SCATS site dataset.
        save_directory (str): Which folder to save the model weights. Weights are saved as a file, e.g., "900.pt". The file name is the site number.
        train_size (float, optional): Percentage of data to use for the training dataset.
        window_size_minutes (int, optional): How long each window size should be.
        epochs (int, optional): How many epochs to train the GRU model for. This controls how many times the GRU model is trained over the entire dataset.
        learning_rate (float, optional): How much the loss function should affect the model weights. This is an important hyperparameter.
    """
    scats_sites = np.unique(all_site_data["SCATS Number"].values).astype('int')

    print(scats_sites)

    for site in tqdm(scats_sites, desc=f"Training GRU Models and saving weights at: {save_directory}", unit="site"):

        # If the GRU model was already trained, skip this SCATS site.
        weight_file = os.path.join(save_directory, f"weights-{site}.pt")
        if os.path.isfile(weight_file): continue

        train_gru_for_scats_site(all_site_data, site_number = site, save_directory=save_directory, window_size_mins=window_size_mins, train_size=train_size, epochs=epochs, learning_rate=learning_rate)


def load_gru_for_scats_site(site_number, save_path, input_size=1, hidden_size=1984, output_size=1, learning_rate=0.0001):
    """
    Load a pre-trained GRU model for SCATS site ``site_number``.
    """
    weights_file = os.path.join(save_path, f"weights-{site_number}.pt")

    if not os.path.isfile(weights_file):
        raise FileNotFoundError(f"Weights file for SCATS site {site_number} does not exist.")

    model = GRUTrainer(input_size, hidden_size, output_size, learning_rate)
    model.load(weights_file)

    return model