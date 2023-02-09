# univariate bidirectional lstm example
import time

import keras.metrics
import numpy as np
import pandas as pd
import tensorflow as tf
import utils as u

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras import activations as act
from tensorboard.plugins.hparams import api as hp

import os

logs_base_dir = 'logs'
checkpoints_base_dir = 'checkpoints'

if not os.path.exists(logs_base_dir):
    os.mkdir(logs_base_dir)

if not os.path.exists(checkpoints_base_dir):
    os.mkdir(checkpoints_base_dir)

# Config parameters
TIMESTAMP = int(time.time())
n_features = 1
data_file = 'MVID_BBG004S68CP5_1m.csv'

# Config hyperparameters
HP_ADDITIONAL_LAYERS = hp.HParam('additional_layers', hp.Discrete([
    "0,0",

    "1,32",
    "2,32",
    "3,32",

    "1,64",
    "2,64",
    "3,64"
]))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([512, 1024, 2048]))
HP_PREDICTION_WINDOW = hp.HParam('prediction_window', hp.Discrete([5, 10, 15, 30]))
HP_WINDOW_SIZE = hp.HParam('window_size', hp.Discrete([15, 30, 60, 120, 360]))
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([32]))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.1]))
HP_EPOCHS = hp.HParam('epochs', hp.Discrete([100]))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.0001]))
HP_DENSE_LAYERS = hp.HParam('dense_layers', hp.Discrete([2]))

METRIC_ACCURACY = 'accuracy'
METRIC_LOSS = 'loss'
METRIC_VAL_ACCURACY = 'val_accuracy'
METRIC_VAL_LOSS = 'val_loss'


def run_train(log_dir, checkpoints_dir, h_params):
    add_layers, add_num_units = h_params[HP_ADDITIONAL_LAYERS].split(',')

    model = Sequential()
    model.add(LSTM(h_params[HP_NUM_UNITS], activation='tanh', input_shape=(h_params[HP_WINDOW_SIZE], n_features),
                   return_sequences=int(add_layers) > 0))
    model.add(Dropout(h_params[HP_DROPOUT]))

    print(add_layers, add_num_units)
    for i in range(int(add_layers)):
        model.add(LSTM(int(add_num_units), activation='tanh', return_sequences=i != int(add_layers) - 1))
        model.add(Dropout(h_params[HP_DROPOUT]))

    for i in range(h_params[HP_DENSE_LAYERS]):
        model.add(Dense(h_params[HP_NUM_UNITS], activation='relu'))
        model.add(Dropout(h_params[HP_DROPOUT]))

    model.add(Dense(3, activation='softmax'))
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=h_params[HP_LEARNING_RATE]),
        loss=tf.losses.CategoricalCrossentropy(),
        metrics=keras.metrics.categorical_accuracy
    )

    history = model.fit(
        train_ts,
        train_value,
        epochs=h_params[HP_EPOCHS],
        batch_size=h_params[HP_BATCH_SIZE],
        verbose=1,
        validation_data=(validate_ts, validate_value),
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir, ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoints_dir,
                monitor='val_loss',
                mode='min',
                save_freq=50,
                save_weights_only=True,
            ),
        ]
    )

    loss = history.history['loss'][-1]
    accuracy = history.history['categorical_accuracy'][-1]
    validation_loss = history.history['val_loss'][-1]
    validation_accuracy = history.history['val_categorical_accuracy'][-1]

    return loss, accuracy, validation_loss, validation_accuracy


def run(tuning_name, run_name, h_params):
    run_dir = f"{logs_base_dir}/{tuning_name}/{run_name}"
    checkpoints_dir = f"{checkpoints_base_dir}/{tuning_name}/{run_name}"
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(h_params, trial_id=f"{tuning_name}/{run_name}")  # record the values used in this trial
        loss, accuracy, validation_loss, validation_accuracy = run_train(run_dir, checkpoints_dir, h_params)

        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
        tf.summary.scalar(METRIC_LOSS, loss, step=1)
        tf.summary.scalar(METRIC_VAL_ACCURACY, validation_accuracy, step=1)
        tf.summary.scalar(METRIC_VAL_LOSS, validation_loss, step=1)


session_num = 1
total_sessions = len(HP_BATCH_SIZE.domain.values) * \
                 len(HP_ADDITIONAL_LAYERS.domain.values) * \
                 len(HP_PREDICTION_WINDOW.domain.values) * \
                 len(HP_WINDOW_SIZE.domain.values) * \
                 len(HP_NUM_UNITS.domain.values) * \
                 len(HP_DROPOUT.domain.values) * \
                 len(HP_EPOCHS.domain.values) * \
                 len(HP_LEARNING_RATE.domain.values) * \
                 len(HP_DENSE_LAYERS.domain.values)

print(f"\n\n\n--------Total sessions: {total_sessions}--------------\n\n\n")

print("Data file:", data_file)

TUNING_NAME = f"{data_file}_{str(TIMESTAMP)}"

# config tensorboard logs
with tf.summary.create_file_writer(f"{logs_base_dir}/{TUNING_NAME}").as_default():
    hp.hparams_config(
        hparams=[
            HP_NUM_UNITS,
            HP_ADDITIONAL_LAYERS,
            HP_DROPOUT,
            HP_EPOCHS,
            HP_LEARNING_RATE,
            HP_DENSE_LAYERS,
            HP_BATCH_SIZE,
            HP_WINDOW_SIZE,
            HP_PREDICTION_WINDOW,
        ],
        metrics=[
            hp.Metric(METRIC_ACCURACY, display_name='Accuracy'),
            hp.Metric(METRIC_LOSS, display_name='Loss'),
            hp.Metric(METRIC_VAL_ACCURACY, display_name='VAL Accuracy'),
            hp.Metric(METRIC_VAL_LOSS, display_name='VAL Loss')
        ],
    )

scaled_data, cols = u.get_raw_stock_data(
    data_file,
    columns=['Close']
)
df = pd.DataFrame(scaled_data, columns=cols)
train_data, validation_data = np.split(df.sample(frac=1), [int(.8 * len(df))])

for prediction_window in HP_PREDICTION_WINDOW.domain.values:
    for window_size in HP_WINDOW_SIZE.domain.values:
        # split into samples
        train_ts, train_value = u.split_sequence(
            train_data[["Close"]].values,
            window_size,
            prediction_window
        )
        validate_ts, validate_value = u.split_sequence(
            validation_data[["Close"]].values,
            window_size,
            prediction_window
        )

        # check if we have enough data
        if len(train_data) * .3 > len(train_ts):
            session_num += 1
            print("Not enough data for this session. Skipping...")
            continue

        print("----------------%s------------------" % len(train_data))
        print("----------------%s------------------" % len(train_ts))
        print("--- prediction_window:%s --- windows_size:%s ---" % (prediction_window, window_size))
        print("buy:", len([a for a in train_value if np.array_equal(a, [0, 0, 1])]))
        print("hold:", len([a for a in train_value if np.array_equal(a, [0, 1, 0])]))
        print("sell:", len([a for a in train_value if np.array_equal(a, [1, 0, 0])]))
        print('----------------------------------')

        train_ts = train_ts.reshape((train_ts.shape[0], train_ts.shape[1], n_features))
        validate_ts = validate_ts.reshape((validate_ts.shape[0], validate_ts.shape[1], n_features))

        for epoch in HP_EPOCHS.domain.values:
            for learning_rate in HP_LEARNING_RATE.domain.values:
                for batch_size in HP_BATCH_SIZE.domain.values:
                    for dropout_rate in HP_DROPOUT.domain.values:
                        for num_units in HP_NUM_UNITS.domain.values:
                            for dense_layers in HP_DENSE_LAYERS.domain.values:
                                for additional_layers in HP_ADDITIONAL_LAYERS.domain.values:
                                    hparams = {
                                        HP_PREDICTION_WINDOW: prediction_window,
                                        HP_WINDOW_SIZE: window_size,
                                        HP_ADDITIONAL_LAYERS: additional_layers,
                                        HP_NUM_UNITS: num_units,
                                        HP_DROPOUT: dropout_rate,
                                        HP_EPOCHS: epoch,
                                        HP_LEARNING_RATE: learning_rate,
                                        HP_DENSE_LAYERS: dense_layers,
                                        HP_BATCH_SIZE: batch_size,
                                    }

                                    RUN_NAME = "run-%d" % session_num
                                    print(f"--- Starting trial: {RUN_NAME}")
                                    print(f"--- Session {session_num} of {total_sessions}")
                                    print({h.name: hparams[h] for h in hparams})
                                    run(TUNING_NAME, RUN_NAME, hparams)
                                    session_num += 1
