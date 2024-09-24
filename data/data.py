import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def process_data(file1, file2, lags, lane):
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
    df1 = pd.read_csv(file1, encoding='utf-8').fillna(0)
    df2 = pd.read_csv(file2, encoding='utf-8').fillna(0)
    
    df1 = df1[df1['VR Internal Loc'] == lane]
    df2 = df2[df2['VR Internal Loc'] == lane]

    # Scale the Vehicle Count data
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df1[attr].values.reshape(-1, 1))
    flow1 = scaler.transform(df1[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    flow2 = scaler.transform(df2[attr].values.reshape(-1, 1)).reshape(1, -1)[0]

    # Create time-lagged sequences
    file1, file2 = [], []
    for i in range(lags, len(flow1)):
        file1.append(flow1[i - lags: i + 1])
    for i in range(lags, len(flow2)):
        file2.append(flow2[i - lags: i + 1])

    file1 = np.array(file1)
    file2 = np.array(file2)
    np.random.shuffle(file1)

    X_train = file1[:, :-1]
    y_train = file1[:, -1]
    X_test = file2[:, :-1]
    y_test = file2[:, -1]

    return X_train, y_train, X_test, y_test, scaler
