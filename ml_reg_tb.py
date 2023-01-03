#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@File    :   ml_reg_tb.py
@Author  :   Tom Jessel
@Contact :   jesselt@cardiff.ac.uk

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
13/10/2022 14:25   tomhj      1.0         None
"""
import time
from typing import Union, Dict, Any

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from scikeras.wrappers import KerasRegressor
from textwrap import dedent

import resources

TARGET = 'Mean radius'


def preprocess_df(df, val_frac):
    scaler = MinMaxScaler()
    for col in df.columns:
        if col != TARGET:
            df[col] = scaler.fit_transform(df[[col]])
    df.dropna(inplace=True)

    x_tr, x_te, y_tr, y_te = resources.split_dataset(df, val_frac)

    return x_tr, x_te, y_tr, y_te


def create_mlp(
        dropout=0.2,
        no_layers=2,
        no_nodes=32,
        activation='relu',
        init_mode='glorot_normal',
):
    model = Sequential(name='MLP_reg')
    model.add(Dense(
        units=no_nodes,
        activation=activation,
        input_shape=(7,),  # todo CHANGE TO WORK DYNAMICALLY
        kernel_initializer=init_mode,
        name='dense1',
        use_bias=True,
    ))
    model.add(Dropout(rate=dropout, name='dropout1'))
    for i, _ in enumerate(range(no_layers - 1)):
        model.add(Dense(
            units=no_nodes,
            activation=activation,
            kernel_initializer=init_mode,
            name=f'dense{i + 2}',
            use_bias=True,
        ))
        model.add(Dropout(rate=dropout, name=f'dropout{i + 2}'))
    model.add(Dense(1, name='output'))
    return model


def create_lstm(
        dropout=0.2,
        no_layers=2,
        no_dense=1,
        no_nodes=64,
        activation='relu',
):
    model = Sequential(name='LSTM_reg')
    model.add(LSTM(
        units=no_nodes,
        activation=activation,
        input_shape=(X_train[1:]),
        return_sequences=True if no_dense == 0 else False,
        name='lstm1',
    ))
    model.add(Dropout(rate=dropout, name='dropout1'))

    i = 0
    for i, _ in enumerate(range(no_layers - 1)):
        model.add(LSTM(
            units=no_nodes,
            activation=activation,
            return_sequences=True if no_dense == 0 else False,
            name=f'lstm{i + 2}',
        ))
        model.add(Dropout(rate=dropout, name=f'dropout{i + 2}'))

    for j, _ in enumerate(range(no_dense)):
        model.add(Dense(
            units=no_nodes,
            activation=activation,
            name=f'dense{j}',
        ))
        model.add(Dropout(rate=dropout, name=f'dropout{j + i}'))
    model.add(Dense(units=1, name='output', activation='linear'))
    return model


def score_test(
        model: Union[KerasRegressor, Pipeline],
        Xtest: np.ndarray,
        ytest: np.ndarray,
        tb_wr: Any = None,
        plot_fig: bool = True,
) -> Dict[str, float]:
    """
    Score a fitted model/pipeline on test set data.

    Args:
        model: Model/Pipeline for scoring.
        Xtest: Test set input data.
        ytest: Corresponding test set output data.
        tb_wr: Tensorboard writer function.

    Returns:
        Dict containing the scores for the fitted model.
    """
    y_pred = model.predict(Xtest, verbose=0)

    _test_score = {
        'MAE': mean_absolute_error(ytest, y_pred),
        'MSE': mean_squared_error(ytest, y_pred),
        'r2': r2_score(ytest, y_pred),
    }

    print('-' * 65)
    print('Model Test Scores:')
    print('-' * 65)
    print(f'MAE = {np.abs(_test_score["MAE"]) * 1000:.3f} um')
    print(f'MSE = {np.abs(_test_score["MSE"]) * 1_000_000:.3f} um^2')
    print(f'R^2 = {np.mean(_test_score["r2"]):.3f}')
    print('-' * 65)
    if tb_wr is not None:
        md_scores = dedent(f'''
        ### Scores - Test Data

        | MAE | MSE |  R2  |
        | ---- | ---- | ---- |
        | {_test_score['MAE'] * 1e3:.3f} | {_test_score['MSE'] * 1e6:.3f} | \
        {_test_score['r2']:.3f} |

        ''')

        with tb_wr.as_default():
            tf.summary.text('Test Scores', md_scores, step=0)

    if plot_fig:
        fig, ax = plt.subplots()
        ax.plot(ytest, color='red', label='Real data')
        ax.plot(y_pred, color='blue', ls='--', label='Predicted data')
        ax.set_title('Model Predictions - Test Set')
        ax.set_ylabel('Mean Radius (mm)')
        ax.set_xlabel('Data Points')
        ax.legend()
        plt.show()
        print('=' * 65)
    return _test_score


def tb_model_desc(model, tb_wr):
    # Model.summary()
    lines = []
    model.model_.summary(print_fn=lines.append)

    dropout = model.model_.layers[1].get_config()['rate']
    layers = model.model_.get_config()['layers']
    nodes = [layer['config']['units'] for layer in layers
             if layer['class_name'] in ('Dense', 'LSTM')]
    no_layers = len(nodes) - 1
    activation = layers[1]['config']['activation']
    opt = model.model_.optimizer.get_config()
    optimiser = opt['name']
    learning_rate = opt['learning_rate']
    decay = opt['decay']

    hp = dedent(f"""
        ### Parameters:
        ___

        | Epochs | Batch Size | No Layers | No Neurons | Init Mode | \
        Activation | Dropout | Loss | Optimiser | Learning rate | \
        Decay |
        |--------|------------|-----------|------------|-----------|\
        ------------|---------|------|-----------|---------------|-------|
        |{model.epochs}|{model.batch_size}|{no_layers}|{nodes[:-1]}|\
        {model.model__init_mode}|{activation}|{dropout:.3f}|{model.loss}|\
        {optimiser}|{learning_rate:.3f}|{decay:.3f}|

        """)

    lines = '    ' + '\n    '.join(lines)

    with tb_wr.as_default():
        tf.summary.text('Model Config', lines, step=0)
        tf.summary.text('Hyper-parameters', hp, step=0)


if __name__ == '__main__':
    EPOCHS = 500
    BATCH_SIZE = 10
    DROPOUT = 0.1
    NO_LAYERS = 2
    NO_NODES = 32
    TEST_FRAC = 0.2
    TARGET = 'Mean radius'

    exp = resources.load('Test 5')
    main_df = exp.features.drop(columns=['Runout', 'Form error'])\
        .drop([0, 1, 23, 24])
    X_train, X_test, y_train, y_test = preprocess_df(main_df, TEST_FRAC)
    print(main_df.keys())

    logdir = 'ml-results/logs/MLP/Features/'
    run_name = f'{logdir}MLP-E-{EPOCHS}-B-{BATCH_SIZE}-' + \
               f'L{np.full(NO_LAYERS, NO_NODES)}-D-{DROPOUT}' + \
               f'-{time.strftime("%Y%m%d-%H%M%S", time.localtime())}'
    tb_writer = tf.summary.create_file_writer(run_name)

    mlp_reg: Any = KerasRegressor(
        model=create_mlp,
        model__dropout=DROPOUT,
        model__activation='relu',
        model__no_nodes=NO_NODES,
        model__no_layers=NO_LAYERS,
        model__init_mode='glorot_normal',
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        loss='mae',
        metrics=['MSE', 'MAE', KerasRegressor.r_squared],
        optimizer='adam',
        optimizer__learning_rate=0.001,
        optimizer__decay=1e-6,
        verbose=1,
        callbacks=[tf.keras.callbacks.TensorBoard(
            log_dir=run_name, histogram_freq=1)],
    )

    # lstm_reg = KerasRegressor(
    #     model=create_lstm,
    #     model__dropout=DROPOUT,
    #     model__activation='relu',
    #     model__no_nodes=NO_NODES,
    #     model__no_layers=NO_LAYERS,
    #     model__no_dense=1,
    #     epochs=EPOCHS,
    #     batch_size=BATCH_SIZE,
    #     loss='mae',
    #     metrics=['MSE', 'MAE', KerasRegressor.r_squared],
    #     optimizer='adam',
    #     # optimizer__learning_rate=0.001,
    #     # optimizer__decay=1e-6,
    #     verbose=1,
    #     # callbacks=[tf.keras.callbacks.TensorBoard(
    #        log_dir=run_name, histogram_freq=1)],
    # )

    mlp_reg.fit(X_train, y_train, validation_split=0.2)

    test_scores = score_test(mlp_reg, X_test, y_test, tb_writer)
    tb_model_desc(mlp_reg, tb_writer)
    with tb_writer.as_default():
        tf.summary.text('Features', list(main_df.keys()), step=0)
