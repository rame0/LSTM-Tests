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

TIMESTAMP = int(time.time())

if not os.path.exists(logs_base_dir):
    os.mkdir(logs_base_dir)

if not os.path.exists(checkpoints_base_dir):
    os.mkdir(checkpoints_base_dir)

scaled_data, cols = u.get_raw_stock_data(
    # 'USD000000TOD_200701_220730_5m.txt',
    'BBG004730N88.csv',
    columns=['Close']
)
df = pd.DataFrame(scaled_data, columns=cols)
train_data, validation_data = np.split(df.sample(frac=1), [int(.8 * len(df))])

# print(train)
# print(val)
# print(test)
# exit()
# define input sequence
# raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220,
#            220, 230, 240, 250, 250, 250, 240, 230, 220, 210, 200, 190, 180, 170, 160, 150, 140, 130, 120, 110]
# choose a number of time steps
n_steps = 40
signal_base_on_next_N_candles = 5

# split into samples
train_ts, train_value = u.split_sequence(train_data[["Close"]].values, n_steps, signal_base_on_next_N_candles)
validate_ts, validate_value = u.split_sequence(validation_data[["Close"]].values, n_steps,
                                               signal_base_on_next_N_candles)

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1

train_ts = train_ts.reshape((train_ts.shape[0], train_ts.shape[1], n_features))
validate_ts = validate_ts.reshape((validate_ts.shape[0], validate_ts.shape[1], n_features))

# print(len(train_value[train_value==0.5]))

patience = 6

# Config hyperparameters
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([32, 64]))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([
    # int(len(train_ts) / 5000),
    # int(len(train_ts) / 1000),
    int(len(train_ts) / 50),
    int(len(train_ts) / 100),
]))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.0, 0.1, 0.2]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['Adam']))
HP_EPOCHS = hp.HParam('epochs', hp.Discrete(range(100, 500, 100)))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.0001, 0.001]))
HP_DENSE_LAYERS = hp.HParam('dense_layers', hp.Discrete([0, 1, 2]))

METRIC_ACCURACY = 'accuracy'
METRIC_LOSS = 'loss'
METRIC_VAL_ACCURACY = 'val_accuracy'
METRIC_VAL_LOSS = 'val_loss'

tuning_name = 'tuning_' + str(TIMESTAMP)

# config tensorboard logs
with tf.summary.create_file_writer(f"{logs_base_dir}/{tuning_name}").as_default():
    hp.hparams_config(
        hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER, HP_EPOCHS, HP_LEARNING_RATE, HP_DENSE_LAYERS, HP_BATCH_SIZE],
        metrics=[
            hp.Metric(METRIC_ACCURACY, display_name='Accuracy'),
            hp.Metric(METRIC_LOSS, display_name='Loss'),
            hp.Metric(METRIC_VAL_ACCURACY, display_name='VAL Accuracy'),
            hp.Metric(METRIC_VAL_LOSS, display_name='VAL Loss')
        ],
    )


#
# early_stopping = tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss',
#     patience=patience,
#     mode='min'
# )
#
# learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(
#     monitor='loss',
#     patience=patience,
#     verbose=1,
#     factor=0.5,
#     min_lr=0.0000001
# )

def run_train(log_dir, checkpoints_dir, h_params):
    model = Sequential()
    model.add(LSTM(h_params[HP_NUM_UNITS], activation='tanh', input_shape=(n_steps, n_features)))
    model.add(Dropout(h_params[HP_DROPOUT]))

    for i in range(h_params[HP_DENSE_LAYERS]):
        model.add(Dense(h_params[HP_NUM_UNITS], activation='relu'))
        model.add(Dropout(h_params[HP_DROPOUT]))

    model.add(Dense(3, activation='softmax'))
    model.compile(
        optimizer=getattr(tf.optimizers, h_params[HP_OPTIMIZER])(learning_rate=h_params[HP_LEARNING_RATE]),
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


def run(run_name, h_params):
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
total_sessions = len(HP_EPOCHS.domain.values) \
                 * len(HP_LEARNING_RATE.domain.values) \
                 * len(HP_BATCH_SIZE.domain.values) \
                 * len(HP_DROPOUT.domain.values) \
                 * len(HP_NUM_UNITS.domain.values) \
                 * len(HP_OPTIMIZER.domain.values) \
                 * len(HP_DENSE_LAYERS.domain.values)

for epoch in HP_EPOCHS.domain.values:
    for learning_rate in HP_LEARNING_RATE.domain.values:
        for batch_size in HP_BATCH_SIZE.domain.values:
            for dropout_rate in HP_DROPOUT.domain.values:
                for num_units in HP_NUM_UNITS.domain.values:
                    for optimizer in HP_OPTIMIZER.domain.values:
                        for dense_layers in HP_DENSE_LAYERS.domain.values:
                            hparams = {
                                HP_NUM_UNITS: num_units,
                                HP_DROPOUT: dropout_rate,
                                HP_OPTIMIZER: optimizer,
                                HP_EPOCHS: epoch,
                                HP_LEARNING_RATE: learning_rate,
                                HP_DENSE_LAYERS: dense_layers,
                                HP_BATCH_SIZE: batch_size,
                            }
                            RUN_NAME = "run-%d" % session_num
                            print(f"--- Starting trial: {RUN_NAME}")
                            print(f"--- Session {session_num} of {total_sessions}")
                            print({h.name: hparams[h] for h in hparams})
                            run(RUN_NAME, hparams)
                            session_num += 1
