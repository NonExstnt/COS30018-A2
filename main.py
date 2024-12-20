import sys
import argparse
import math
import warnings
import numpy as np
import pandas as pd
from data.data import process_data
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error
    Calculate the mape.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    # Returns
        mape: Double, result data for train.
    """
    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]
    num = len(y_pred)
    sums = 0
    for i in range(num):
        tmp = abs(y[i] - y_pred[i]) / y[i]
        sums += tmp
    mape = sums * (100 / num)
    return mape

def eva_regress(y_true, y_pred):
    """Evaluation
    evaluate the predicted resul.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    """
    mape = MAPE(y_true, y_pred)
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print('explained_variance_score:%f' % vs)
    print('mape:%f%%' % mape)
    print('mae:%f' % mae)
    print('mse:%f' % mse)
    print('rmse:%f' % math.sqrt(mse))
    print('r2:%f' % r2)

def plot_results(y_true, y_preds, names):
    """Plot
    Plot the true data and predicted data.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
        names: List, Method names.
    """
    d = '10/1/2006 00:00'
    x = pd.date_range(d, periods=96, freq='15min')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y_true, label='True Data')
    for name, y_pred in zip(names, y_preds):
        ax.plot(x, y_pred, label=name)

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Vehicle Count')

    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.show()

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lane", 
        default=1, 
        type=int, 
        help="Lane to use.")
    parser.add_argument(
        "--scat", 
        default=970, 
        type=int, 
        help="SCAT site to predict.")
    args = parser.parse_args()

    scat = args.scat
    lane = args.lane

    # Load models
    lstm = load_model(f'model/lstm/{scat}/{lane}.h5')
    gru = load_model(f'model/gru/{scat}/{lane}.h5')
    saes = load_model(f'model/saes/{scat}/{lane}.h5')
    rnn = load_model(f'model/rnn/{scat}/{lane}.h5')
    models = [lstm, gru, saes, rnn]
    names = ['LSTM', 'GRU', 'SAEs', 'RNN']
    
    # Set parameters
    lag = 12
    file1 = f"data/Scat_number_{scat}_train.csv"
    file2 = f"data/Scat_number_{scat}_test.csv"

    # Process data for the selected lane
    _, _, X_test, y_test, scaler = process_data(file1, file2, lag, lane)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]

    y_preds = []
    for name, model in zip(names, models):
        if name == 'SAEs':
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
        else:
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        file = 'images/' + name + '.png'
        plot_model(model, to_file=file, show_shapes=True)
        predicted = model.predict(X_test)
        predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
        y_preds.append(predicted[:96])
        print(name)
        eva_regress(y_test, predicted)

    plot_results(y_test[:96], y_preds, names)

if __name__ == '__main__':
    main(sys.argv)
