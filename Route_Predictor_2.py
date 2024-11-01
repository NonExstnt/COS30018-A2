import numpy as np
import datetime
import math
from enum import Enum

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os
from model_2 import get_lstm, get_gru, get_ae, get_rnn
from data_2 import TrafficFlowDataProcessor

# create_series, create_datetime_series

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf



# Enum for Model Types
class TrafficFlowModels(Enum):
    LSTM = 'LSTM'
    GRU = 'GRU'
    SAES = 'SAES'
    RNN = 'RNN'


class RoutePredictor:

    from data_2 import TrafficFlowDataProcessor
    def __init__(self, lags=12):
        self.processor = TrafficFlowDataProcessor(
            train_path=r'C:\Users\ASUS\OneDrive\Semester 3\Intelligent System\GIT\COS30018-A2\train.csv',
            test_path=r'C:\Users\ASUS\OneDrive\Semester 3\Intelligent System\GIT\COS30018-A2\test.csv'


        )


        # Initialize the TrafficFlowDataProcessor with the specified paths
        self.processor = TrafficFlowDataProcessor(
            train_path=self.train_file,
            test_path=self.test_file
        )

        self.model = {}
        self.lags = lags
        self.location_series_data = {}

        # Initialize scalers and lookup data
        self.flow_scaler, self.days_scaler, self.times_scaler = self._init_scalers()
        self.series_data = self._init_lookup_data()

    def get_model(self, model_name: str):
        if self.model.get(model_name) == None:
            self.model[model_name] = load_model(
                os.path.join(os.path.dirname(__file__), 'model', f'{model_name}.h5'))
        return self.model.get(model_name)

    def _init_scalers(self):
        """Load scalers using training data."""
        # Call the instance method `create_datetime_series` directly
        _, _, _, _, self.flow_scaler, self.days_scaler, self.times_scaler = self.processor.create_datetime_series(self.train_file, self.test_file)
    def _init_lookup_data(self):
        """Load lookup data for time series predictions."""
        # Call the instance method `create_series` directly
        _, _, _,self.series_data, _, _, _ = self.processor.create_series(self.train_file, self.test_file ,self.lags)
        #return series_data

    def create_model(self, model_type, units):
        """Create a neural network model based on the specified type."""
        if model_type == TrafficFlowModels.LSTM.value:
            return get_lstm(units)
        elif model_type == TrafficFlowModels.GRU.value:
            return get_gru(units)
        elif model_type == TrafficFlowModels.SAES.value:
            # Assuming units include the input and hidden dimensions
            return get_ae(units[0], units[1], units[2:], last_ae=True)  # Example for Auto-Encoder
        elif model_type == TrafficFlowModels.RNN.value:
            return get_rnn(units)
        else:
            raise ValueError(f"Model type {model_type} is not recognized.")

    def load_model(self, model_name):
        """Load and cache model."""
        if model_name not in self.model:
            model_path = os.path.join(os.path.dirname(__file__), 'model', f'{model_name}.h5')
            self.model[model_name] = load_model(model_path)
        return self.model[model_name]

    def predict_traffic_flow(self, location: int, date: datetime, steps: int, model_name: str):
        model = self.get_model(model_name)

        X = None
        if model_name == "average":
            X = self.get_datetime_inputs(location, date, steps)
            if X is None: return 0
            y_pred = self.predict_datetime(model, X)
        else:
            X = self.get_timeseries_inputs(location, date, steps)
            if X is None: return 0
            y_pred = self.predict_series(model, X)

        return y_pred.sum()

    def get_datetime_inputs(self, location: int, date: datetime, steps: int):
        dayindex = date.weekday()  # determine weekday
        actual_time = date.hour * 60 + date.minute  # determine time in minutes
        rounded_time = 15 * math.floor(actual_time / 15)  # get current 15 minute interval

        days = self.days_scaler.transform(np.array([dayindex for _ in range(steps)]).reshape(-1, 1)).reshape(1, -1)[0]
        times = \
        self.times_scaler.transform(np.array([actual_time + t * 15 for t in range(steps)]).reshape(-1, 1)).reshape(1,
                                                                                                                   -1)[
            0]
        scats = self.scats_scaler.transform(np.array([location for _ in range(steps)]).reshape(-1, 1)).reshape(1, -1)[0]

        X = np.array([np.array([days[i], times[i], scats[i]]) for i in range(steps)])
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X

    def predict_datetime(self, model, X):
        y_pred = model.predict(X)
        y_pred = self.flow_scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(1, -1)[0]

        return y_pred

    def lookup_location_data(self,location:int):
        scaled_location = self.scats_scaler.transform(np.array([location]).reshape(-1,1)).reshape(1,-1)[0][0]
        if self.location_series_data.get(location) is None:
            location_indices = [i for i in range(len(self.series_data)) if self.series_data[i][self.lags] == scaled_location]
            self.location_series_data[location] = self.series_data[location_indices]

        return self.location_series_data[location]

    def get_timeseries_inputs(self,location: int,date:datetime,steps:int):
        day = date.day
        actual_time = date.hour * 60 + date.minute # determine time in minutes
        rounded_time = 15 * math.floor(actual_time / 15) # get current 15 minute interval
        time_index = int(rounded_time / 15)

        location_X = self.lookup_location_data(location)
        if len(location_X) == 0:
            raise Exception(f"No Data exists for location {location}")

        day_X = location_X[(day-1)*96:day*96]

        # fix for bad data having incomplete days
        while len(day_X) == 0 and day >= 0:
            day -= 7
            day_X = location_X[(day-1)*96:day*96]

        if len(day_X) == 0:
            return None

        X = np.array([day_X[time_index + i] for i in range(steps)])
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X

    def predict_series(self, model, X):
        y_pred = model.predict(X)
        y_pred = self.flow_scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(1, -1)[0]
        return y_pred