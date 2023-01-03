#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@File    :   learn_LSTM.py
@Author  :   Tom Jessel
@Contact :   jesselt@cardiff.ac.uk

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
10/10/2022 09:34   tomhj      1.0         None
"""
from collections import deque
import time

from keras.callbacks import TensorBoard
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import resources
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

TARGET = 'Mean radius'  # Target for model to predict
SEQ_LEN = 10    # Number of data points to store within the RNN
VAL_FRAC = 0.1  # Fraction of data for validation set
EPOCHS = 200
BATCH_SIZE = 3
NAME = f'LSTM-SEQ-{SEQ_LEN}-EPCH-{EPOCHS}-BTCH-{BATCH_SIZE}-{time.strftime("%Y%m%d-%H%M%S",time.gmtime())}'


def preprocess_df(df):

    scaler = MinMaxScaler()

    for col in df.columns:
        if col != TARGET:
            df[col] = scaler.fit_transform(df[[col]])
    df.dropna(inplace=True)

    seq_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            seq_data.append([np.array(prev_days), i[-1]])  # When queue is long enough save it with the target value

    X = []
    y = []

    for seq, target in seq_data:
        X.append(seq)
        y.append(target)
    return np.array(X), np.array(y)


if __name__ == '__main__':
    exp = resources.load('Test5')
    main_df = exp.features.drop(columns=['Runout', 'Form error'])

    ind_val = main_df.index.values[-int(VAL_FRAC * len(main_df.index))]

    val_df = main_df[(main_df.index >= ind_val)].copy()
    train_df = main_df[(main_df.index < ind_val)].copy()

    train_X, train_y = preprocess_df(train_df)
    val_X, val_y = preprocess_df(val_df)
    print(f"Train data shape:\t{train_X.shape}\nVal data shape:\t{val_X.shape}")

    model = Sequential()

    model.add(LSTM(128, input_shape=(train_X.shape[1:]), activation='relu', return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(128, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='linear'))

    opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)

    model.compile(
        loss='mse',
        optimizer=opt,
        metrics=['MSE', 'MAE']
    )

    logdir = f'ml-results/logs/LSTM/{NAME}'
    file_writer = tf.summary.create_file_writer(logdir)
    tensorboard = TensorBoard(log_dir=logdir)

    history = model.fit(
        train_X,
        train_y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(val_X, val_y),
        # callbacks=[tensorboard],
    )

    score = model.evaluate(val_X, val_y, verbose=0)
    print('Test MSE:', score[1])
    print('Test MAE:', score[2])

    with file_writer.as_default():
        tf.summary.text('Test Scores', f'Test MSE: {(score[1] * 1e6):.3f} Test MAE: {(score[2] * 1000):.3f}', step=EPOCHS)
    # # Save model
    # model.save("models/{}".format(NAME))