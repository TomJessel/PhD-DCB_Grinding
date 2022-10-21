#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@File    :   mlp_hpsearch.py
@Author  :   Tom Jessel
@Contact :   jesselt@cardiff.ac.uk
 
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
12/10/2022 13:18   tomhj      1.0         None
"""
import time

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Input
from tensorflow.python.keras.callbacks import TensorBoard
from tensorboard.plugins.hparams import api as hp

import resources

logdir = f'ml-results/logs/HPsearch/MLP/'
TARGET = 'Mean radius'  # Target for model to predict

EPOCHS = [500]
BATCH_SIZE = [5, 10, 15]
DROPOUT = [0.1, 0.2, 0.3]
NO_LAYERS = [1, 2, 3]
NO_NODES = [16, 32, 64, 128]
TEST_FRAC = 0.1

HP_NO_NODES = hp.HParam('no_nodes', hp.Discrete(NO_NODES))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete(DROPOUT))
HP_NO_LAYERS = hp.HParam('no_layers', hp.Discrete(NO_LAYERS))
HP_BTCH_SIZE = hp.HParam('batch_size', hp.Discrete(BATCH_SIZE))
HP_EPOCHS = hp.HParam('epochs', hp.Discrete(EPOCHS))

METRIC_MAE = hp.Metric('epoch_MAE', group='validation', display_name='mae')
METRIC_MSE = hp.Metric('epoch_MSE', group='validation', display_name='mse')

file_writer_hp = tf.summary.create_file_writer(logdir)
with file_writer_hp.as_default():
    hp.hparams_config(
        hparams=[HP_EPOCHS, HP_DROPOUT, HP_BTCH_SIZE, HP_NO_LAYERS, HP_NO_NODES],
        metrics=[METRIC_MSE, METRIC_MAE]
    )


def preprocess_df(df, val_frac):
    scaler = MinMaxScaler()
    for col in df.columns:
        if col != TARGET:
            df[col] = scaler.fit_transform(df[[col]])
    df.dropna(inplace=True)

    x_tr, x_te, y_tr, y_te = resources.split_dataset(df, val_frac)

    return x_tr, x_te, y_tr, y_te


def train_test_model(x_tr, x_te, y_tr, y_te, hparams, runname):
    model = Sequential()
    model.add(Dense(hparams[HP_NO_NODES], input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dropout(hparams[HP_DROPOUT]))

    for _ in range(hparams[HP_NO_LAYERS]-1):
        model.add(Dense(hparams[HP_NO_NODES], activation='relu'))
        model.add(Dropout(hparams[HP_DROPOUT]))
    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mae',
        metrics=['MAE', 'MSE']
    )
    logdir_i = logdir + runname
    model.fit(
        x_tr,
        y_tr,
        batch_size=hparams[HP_BTCH_SIZE],
        epochs=hparams[HP_EPOCHS],
        validation_data=(x_te, y_te),
        callbacks=[
            TensorBoard(log_dir=logdir_i),
            hp.KerasCallback(logdir_i, hparams, trial_id=runname)
        ]
    )

    _, mae, mse = model.evaluate(x_te, y_te)
    return mae, mse


exp = resources.load('Test 5')
main_df = exp.features.drop(columns=['Runout', 'Form error'])
X_train, X_test, y_train, y_test = preprocess_df(main_df, TEST_FRAC)

NAME = f'MLP-{time.strftime("%Y%m%d-%H%M%S",time.gmtime())}'

for no_nodes in HP_NO_NODES.domain.values:
    for dropout_rate in HP_DROPOUT.domain.values:
        for no_layers in HP_NO_LAYERS.domain.values:
            for batch_size in HP_BTCH_SIZE.domain.values:
                for epochs in HP_EPOCHS.domain.values:
                    hparams = {
                        HP_NO_NODES: no_nodes,
                        HP_DROPOUT: dropout_rate,
                        HP_NO_LAYERS: no_layers,
                        HP_BTCH_SIZE: batch_size,
                        HP_EPOCHS: epochs,
                    }
                    run_name = f'MLP-{time.strftime("%Y%m%d-%H%M%S",time.gmtime())}'
                    print('--- Starting trial: %s' % run_name)
                    print({h.name: hparams[h] for h in hparams})
                    train_test_model(X_train, X_test, y_train, y_test, hparams, run_name)
