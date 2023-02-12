import os

import pandas as pd
from numpy import array
from sklearn.preprocessing import MinMaxScaler

data_path = os.path.join(os.getcwd())


def get_stock_data(filename, folder='data'):
    """
    Gets stock data from Kaggle files
    """
    df = pd.read_csv(os.path.join(data_path, folder, f"{filename}"), index_col="Date", parse_dates=True)
    return df


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


def split_sequence(sequence, n_steps, predict_step=5):
    x, y = list(), list()
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

        x.append(seq_x)
        y.append(seq_y)
    return array(x), array(y)


def split_sequence_v5(sequence, columns, col_to_predict="Close", n_steps=15, predict_steps=5, predict_percent=0.01):
    counter = 0
    threshold = int(predict_steps / 2) + 1
    print(n_steps, predict_steps)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(sequence[columns].values)
    x = []
    y = []

    for i in range(len(sequence) - n_steps - predict_steps):
        counter += 1
        x_i = sequence[i:i + n_steps][columns]
        future_n = sequence[i + n_steps:i + n_steps + predict_steps][col_to_predict].values
        curent_price = x_i[col_to_predict].values[-1]

        raised_items = sum(i > curent_price + curent_price * predict_percent for i in future_n)
        fallen_items = sum(i < curent_price - curent_price * predict_percent for i in future_n)

        # print( f"Threshold: {threshold} Current price: {curent_price} Raised items: {raised_items} Fallen items: {
        # fallen_items}")

        if raised_items >= threshold:
            y_i = [0, 0, 1]  # raised
            # print("Raised")
        elif fallen_items >= threshold:
            y_i = [1, 0, 0]  # fallen
            # print("Fallen")
        else:
            y_i = [0, 1, 0]  # hold
            # print("Hold")
        x.append(scaler.transform(x_i.values))
        y.append(y_i)
        counter += 1
        if counter > 100:
            break
    return x, y


def split_sequence_v4(sequence, n_steps, predict_step=5):
    x, y = list(), list()
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

        if sum(seq_y) / len(seq_y) > current_price + current_price * 0.2:
            seq_y = [0, 0, 0, 0, 0, 0, 1]  # buy
        elif sum(seq_y) / len(seq_y) > current_price + current_price * 0.1:
            seq_y = [0, 0, 0, 0, 0, 1, 0]  # buy
        elif sum(seq_y) / len(seq_y) > current_price + current_price * 0.05:
            seq_y = [0, 0, 0, 0, 1, 0, 0]  # buy

        elif sum(seq_y) / len(seq_y) < current_price - current_price * 0.05:
            seq_y = [0, 0, 1, 0, 0, 0, 0]  # sell
        elif sum(seq_y) / len(seq_y) < current_price - current_price * 0.1:
            seq_y = [0, 1, 0, 0, 0, 0, 0]  # sell
        elif sum(seq_y) / len(seq_y) < current_price - current_price * 0.2:
            seq_y = [1, 0, 0, 0, 0, 0, 0]  # sell

        else:
            seq_y = [0, 0, 0, 1, 0, 0, 0]  # hold

        x.append(seq_x)
        y.append(seq_y)
    return array(x), array(y)
