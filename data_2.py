import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from Route_Predictor_2 import RoutePredictor

from model_2 import get_lstm, get_gru, get_ae, get_rnn

class TrafficFlowDataProcessor:
    def __init__(self, train_path, test_path):

        self.df_train = pd.read_csv(train_path, parse_dates=[0], dayfirst=True)
        self.df_test = pd.read_csv(test_path, parse_dates=[0], dayfirst=True)

        # Rename the columns
        self.df_train.columns = ['date_time', 'lane_flow', 'lane_points', 'percent_observed']
        self.df_test.columns = ['date_time', 'lane_flow', 'lane_points', 'percent_observed']

        self.flow_attr = ['Lane 1 Flow (Veh/5 Minutes)']
        self.date_time_attr = 'date_time'
        self.flow_train = self.df_train['lane_flow'].values
        self.scats_train = self.df_train['lane_points'].values

        # Initialize scalers
        self.flow_scaler = MinMaxScaler(feature_range=(0, 1))
        self.scats_scaler = MinMaxScaler(feature_range=(0, 1))
        self.day_scaler = None
        self.time_scaler = None

        # Scale the relevant columns
        self.flow_train = self.scale_column(self.df_train, self.flow_attr, self.flow_scaler)
        self.flow_test = self.scale_column(self.df_test, self.flow_attr, self.flow_scaler)
        self.scats_train = self.scale_column(self.df_train, self.scats_attr, self.scats_scaler)
        self.scats_test = self.scale_column(self.df_test, self.scats_attr, self.scats_scaler)


    def scale_column(self, df, column, scaler):
        """Scale a column using the provided scaler."""
        return scaler.fit_transform(df[column].values.reshape(-1, 1)).reshape(-1)

    def parse_date(date_string):
        """Parse a date string into a datetime object."""
        date, time = date_string.split()
        day, month, year = date.split('/')
        hour, minute = time.split(':')
        return datetime.datetime(int(year), int(month), int(day), int(hour), int(minute))

    def create_datetime_series(self, df_train=None, df_test=None, scats_id=1,
                               day=0):
        """Generate series based on datetime attributes like day and time."""
        if df_train is None:
            df_train = self.processor.df_train
        if df_test is None:
            df_test = self.processor.df_test

        # Convert datetime column to correct format and extract features
        dates_train = df_train[self.processor.date_time_attr].dt.to_pydatetime()
        dates_test = df_test[self.processor.date_time_attr].dt.to_pydatetime()


        # Weekdays scaling
        days_train = np.array([d.weekday() for d in dates_train])
        days_test = np.array([d.weekday() for d in dates_test])
        self.day_scaler = MinMaxScaler(feature_range=(0, 1)).fit(days_train.reshape(-1, 1))
        days_train = self.day_scaler.transform(days_train.reshape(-1, 1)).flatten()
        days_test = self.day_scaler.transform(days_test.reshape(-1, 1)).flatten()

        # Time scaling
        times_train = np.array([d.hour * 60 + d.minute for d in dates_train])
        times_test = np.array([d.hour * 60 + d.minute for d in dates_test])
        self.time_scaler = MinMaxScaler(feature_range=(0, 1)).fit(times_train.reshape(-1, 1))
        times_train = self.time_scaler.transform(times_train.reshape(-1, 1)).flatten()
        times_test = self.time_scaler.transform(times_test.reshape(-1, 1)).flatten()

        # Prepare train and test data with datetime features
        train_data = []
        for i in range(len(self.flow_train)):
            row = [days_train[i], times_train[i], self.scats_train[i], self.flow_train[i]]
            train_data.append(row)

        train_data = np.array(train_data)
        np.random.shuffle(train_data)

        # Separate features and targets
        X_train, y_train = train_data[:, :-1], train_data[:, -1]

        # Scale day and SCATS ID
        day = self.day_scaler.transform(np.array([day]).reshape(-1, 1)).flatten()[0]
        scats_id = self.scats_train[0]  # Example to set scats_id to the first entry, adjust as needed
        location_indices = [i for i in range(len(X_train)) if X_train[i][2] == scats_id and X_train[i][0] == day]
        X_location, y_location = X_train[location_indices], y_train[location_indices]

        return X_location, y_location  # Return processed data