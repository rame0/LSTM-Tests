import tensorflow as tf

input_user = "Adam"
getattr(tf.optimizers, input_user)()

exit()


for layers in range(3, 6):
    model = Sequential()
    if layers == 1:
        model.add(LSTM(32, activation=act.relu, input_shape=(n_steps, n_features)))
        model.add(Dropout(0.2))
    elif layers == 2:
        model.add(LSTM(32, activation=act.relu, input_shape=(n_steps, n_features), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(16, activation=act.relu))
    elif layers == 3:
        model.add(LSTM(32, activation=act.relu, input_shape=(n_steps, n_features), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(16, activation=act.relu, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(16, activation=act.relu))
    elif layers == 4:
        model.add(LSTM(32, activation=act.relu, input_shape=(n_steps, n_features), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(64, activation=act.relu, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(16, activation=act.relu))

    else:
        break

    model.add(Dense(3, activation=act.sigmoid))

    for batch_size in batch_sizes:
        mdl_dir = models_dir + "{}/layers:{}-batch_size:{}/".format(TIMESTAMP, layers, batch_size)
        logs_dir = logs_base_dir + "{}/layers:{}-batch_size:{}".format(TIMESTAMP, layers, batch_size)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=logs_dir,
            histogram_freq=1
        )
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=mdl_dir + "mdl_e:{epoch}-acc:{accuracy:.2f}.hdf5",
                # save_best_only=True,  # Only save a model if `val_loss` has improved.
                save_freq=50,
                monitor="accuracy",
                verbose=False,
            ),
            hp.KerasCallback(logdir, hparams),  # log hparams

            # early_stopping,
            # learning_rate_reduction,
            tensorboard_callback
        ]

        model.compile(
            # loss=tf.losses.MeanSquaredError(),
            loss=tf.losses.CategoricalCrossentropy(),
            optimizer=tf.optimizers.Adam(learning_rate=0.01),
            # optimizer=tf.optimizers.Adam(),
            metrics=[tf.metrics.Accuracy()],
        )

        model.fit(
            train_ts,
            train_value,
            epochs=epochs,
            batch_size=batch_size,
            # validation_data=validate_ts,
            callbacks=callbacks,
            verbose=True,
            validation_split=0.2,
        )