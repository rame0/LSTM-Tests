import pandas as pd
import numpy as np
import os
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from WindowGeneratorClass import WindowGenerator
from utils import get_scaled_stock_data

transformers = {}

scaler, scaled_data, cols = get_scaled_stock_data(
    'USD000000TOD_200701_220730_5m',
    columns=['Open', 'High', 'Low', 'Close']
)

df = pd.DataFrame(scaled_data, columns=cols)
train, val, test = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])


def compile_and_fit(model, window, epochs=20, batch_size=32, patience=6):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min'
    )

    learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        patience=patience,
        verbose=1,
        factor=0.5,
        min_lr=0.0000001
    )

    model.compile(
        loss=tf.losses.MeanSquaredError(),
        optimizer=tf.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.metrics.MeanAbsoluteError(), tf.metrics.MeanAbsolutePercentageError()]
    )

    history = model.fit(
        window.train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=window.val,
        callbacks=[early_stopping, learning_rate_reduction]
    )

    return history


OUT_STEPS = 4
CONV_WIDTH = 24
LABEL_WIDTH = 24
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)

NUMBER_OF_EPOCHS = 200
BATCH_SIZE = 34

# window = WindowGenerator(
#     input_width=INPUT_WIDTH,
#     label_width=LABEL_WIDTH,
#     shift=1,
#     label_columns=cols
# )

window_generator = WindowGenerator(
    train_df=train, val_df=val, test_df=test,
    input_width=INPUT_WIDTH,
    label_width=LABEL_WIDTH,
    shift=OUT_STEPS,
    label_columns=cols
)

print(window_generator.train)
exit()

example_window = tf.stack([np.array(train[:window_generator.total_window_size]),
                           np.array(train[100:100 + window_generator.total_window_size]),
                           np.array(train[200:200 + window_generator.total_window_size]), ])

example_inputs, example_labels = window_generator.split_window(example_window)

window_generator.example = example_inputs, example_labels

# window.plot(plot_col='Close')
# window.plot()
# exit()


lstm_model = tf.keras.checkpoints_base_dir.Sequential([
    # Shape [batch, time, features] => [batch, features, time]
    tf.keras.layers.Permute([2, 1]),
    tf.keras.layers.LSTM(256, return_sequences=True, use_bias=True, activation='relu'),
    tf.keras.layers.LSTM(64, return_sequences=True, use_bias=True, activation='relu'),
    # Shape => [batch, features, time] => [batch, time, features]
    tf.keras.layers.Dense(24),
    tf.keras.layers.Permute([2, 1]),
])

compile_and_fit(lstm_model, window_generator, epochs=NUMBER_OF_EPOCHS, batch_size=BATCH_SIZE)
# lstm_model.summary()
window_generator.plot(model=lstm_model)

# lstm_model_2 = tf.keras.models.Sequential([
#     # Shape [batch, time, features] => [batch, features, time]
#     tf.keras.layers.Permute([2, 1]),
#     tf.keras.layers.LSTM(256, return_sequences=True, use_bias=True, activation='relu'),
#     # Shape => [batch, features, time] => [batch, time, features]
#     tf.keras.layers.Dense(24),
#     tf.keras.layers.Permute([2, 1]),
# ])
#
# compile_and_fit(lstm_model_2, window, epochs=NUMBER_OF_EPOCHS, batch_size=BATCH_SIZE)

# lstm_model_2.summary()

# window.plot(model=lstm_model_2)
