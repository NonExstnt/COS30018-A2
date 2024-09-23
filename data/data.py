import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def process_data(Scat_number_970_train, Scat_number_970_test, lags, lane):
    """Process data
    Reshape and split train\test data based on the lane (Location).

    # Arguments
        train: String, name of .csv train file.
        test: String, name of .csv test file.
        lags: integer, time lag.
        lane: integer, lane number to filter data.
    # Returns
        X_train: ndarray.
        y_train: ndarray.
        X_test: ndarray.
        y_test: ndarray.
        scaler: StandardScaler.
    """
    attr = 'Vehicle Count'
    
    # Read the data and filter by the given lane (Location)
    df1 = pd.read_csv(Scat_number_970_train, encoding='utf-8').fillna(0)
    df2 = pd.read_csv(Scat_number_970_test, encoding='utf-8').fillna(0)
    
    df1 = df1[df1['VR Internal Loc'] == lane]
    df2 = df2[df2['VR Internal Loc'] == lane]

    # Scale the Vehicle Count data
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df1[attr].values.reshape(-1, 1))
    flow1 = scaler.transform(df1[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    flow2 = scaler.transform(df2[attr].values.reshape(-1, 1)).reshape(1, -1)[0]

    # Create time-lagged sequences
    Scat_number_970_train, Scat_number_970_test = [], []
    for i in range(lags, len(flow1)):
        Scat_number_970_train.append(flow1[i - lags: i + 1])
    for i in range(lags, len(flow2)):
        Scat_number_970_test.append(flow2[i - lags: i + 1])

    Scat_number_970_train = np.array(Scat_number_970_train)
    Scat_number_970_test = np.array(Scat_number_970_test)
    np.random.shuffle(Scat_number_970_train)

    X_train = Scat_number_970_train[:, :-1]
    y_train = Scat_number_970_train[:, -1]
    X_test = Scat_number_970_test[:, :-1]
    y_test = Scat_number_970_test[:, -1]

    return X_train, y_train, X_test, y_test, scaler
