import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from numpy import array

data_path = os.path.join(os.getcwd())


def get_stock_data(filename, folder='data'):
    """
    Gets stock data from Kaggle files
    """
    df = pd.read_csv(os.path.join(data_path, folder, f"{filename}"), index_col="Date", parse_dates=True)
    return df


def get_scaled_stock_data(filename, folder='data', columns=None):
    """
    Gets scaled data from a symbol
    """

    if columns is None:
        columns = ['Open', 'High', 'Low', 'Close', 'Vol']

    df = get_stock_data(filename, folder)

    scaler_ = MinMaxScaler(feature_range=(0, 1))

    for col in columns:
        if col not in df.columns:
            print('No requested columns in dataset:')
            print('Dataset cols:' + ','.join(df.columns))
            exit()

    for col in df.columns:
        if col not in columns:
            df = df.drop(col, axis=1)

    scaled_data_ = scaler_.fit_transform(df)

    return scaler_, scaled_data_, columns


def get_raw_stock_data(filename, folder='data', columns=None):
    """
    Gets scaled data from a symbol
    """

    if columns is None:
        columns = ['Open', 'High', 'Low', 'Close', 'Vol']

    df = get_stock_data(filename, folder)

    for col in columns:
        if col not in df.columns:
            print('No requested columns in dataset:')
            print('Dataset cols:' + ','.join(df.columns))
            exit()

    for col in df.columns:
        if col not in columns:
            df = df.drop(col, axis=1)

    return df, columns



# split a univariate sequence
def split_sequence(sequence, n_steps, predict_step=5):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - predict_step:
            break
        # gather input and output parts of the pattern
        seq_x = sequence[i:end_ix]
        seq_y = sequence[end_ix:predict_step + end_ix]

        current_price = seq_x[-1]
        min_buy = current_price + current_price * 0.05
        sell = current_price - current_price * 0.05

        # if min(seq_y) > min_buy:
        if sum(seq_y) / len(seq_y) > min_buy:
            seq_y = [0, 0, 1]  # buy
        # elif max(seq_y) < sell:
        elif sum(seq_y) / len(seq_y) < sell:
            seq_y = [1, 0, 0]  # sell
        else:
            seq_y = [0, 1, 0]  # hold

        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)
