#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@File    :   main.py
@Author  :   Tom Jessel
@Contact :   jesselt@cardiff.ac.uk

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
22/08/2022 13:46   tomhj      1.0         Main file
"""
import os
import time

import matplotlib.pyplot as plt
import mplcursors
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

import resources


def corr_matrix(df: pd.DataFrame, save_fig: bool = True):
    cols = df.columns
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    sns.heatmap(df.corr(),
                ax=ax,
                annot=True,
                xticklabels=cols,
                yticklabels=cols,
                vmin=-1,
                vmax=1) \
        .set(title='Correlation Matrix')
    plt.tight_layout()
    plt.show()
    if save_fig:
        path = f'{exp.test_info.dataloc}/Figures'
        png_name = f'{path}/Test {exp.test_info.testno} - Correlation Matrix.png'
        if not os.path.isdir(path) or not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(png_name, dpi=200)
    return df.corr()


def corr_pairplot(df: pd.DataFrame, save_fig: bool = True):
    sns.pairplot(df, height=1.05, aspect=1.85)
    if save_fig:
        path = f'{exp.test_info.dataloc}/Figures'
        png_name = f'{path}/Test {exp.test_info.testno} - Pair Plot.png'
        plt.savefig(png_name, dpi=300)
    plt.show()


def ae_hits(exp: object, s: np.array):
    """
    Convert AE signal into hit data
    :param exp: object: experiement object
    :param s: ndarray: AE signal
    """

    def dbtov(db):
        v_ref = 1E-4
        v = [(10 ** (d / 20)) * v_ref for d in db]
        return v

    # THRESHOLD
    th = 91
    # hit time definition parameters from AEwin
    HDT = 800E-6
    PDT = 200E-6
    HLT = 1000E-6
    MAXD = 1000E-3
    fs = exp.test_info.acquisition[0]
    th_v = dbtov([th])

    abs_sig = np.abs(s)
    th_ind = abs_sig > th_v

    fig, ax = plt.subplots()
    ax.plot(np.arange(0, len(s)) * (1 / fs), s)
    ax.axhline(y=th_v[0], color='r')
    ax.axhline(y=(-1 * th_v[0]), color='r')
    mplcursors.cursor(multiple=True)
    plt.show()
    # todo finish ae hits attempt
    pass


if __name__ == '__main__':
    exp = resources.load(file='Test 5')
    dataframe = exp.features.drop(columns=['Runout', 'Form error']).drop([0, 1, 23, 24])

    logdir = 'ml-results/logs/MLP/CV/'
    run_name = f'{logdir}MLP-{time.strftime("%Y%m%d-%H%M%S", time.localtime())}'
    tb_writer = tf.summary.create_file_writer(run_name)

    pipe = resources.create_pipeline(
        model=resources.get_regression,
        model__init_mode='glorot_normal',
        model__dropout=0.1,
        model__hidden_layer_sizes=(30, 30),
        optimizer='adam',
        optimizer__learning_rate=0.001,
        loss='mae',
        metrics=['MAE', 'MSE'],
        batch_size=10,
        epochs=500,
        verbose=0,
        callbacks=[tf.keras.callbacks.TensorBoard(log_dir=run_name, histogram_freq=1)]
    )


    X_train, X_test, y_train, y_test = resources.split_dataset(dataframe)

    pipe, train_scores = resources.score_train(model=pipe, Xdata=X_train, ydata=y_train)

    pipe.fit(X_train, y_train, mlp_reg__validation_split=0.2)
    resources.train_history(pipe)

    test_score = resources.score_test(pipe, X_test, y_test)
    # c = corr_matrix(feat.drop([0, 1, 23, 24]), save_fig=False)
    # corr_pairplot(feat.drop([0, 1, 23, 24]), save_fig=False)
