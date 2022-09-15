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

import matplotlib.pyplot as plt
import mplcursors
import numpy as np
import pandas as pd
import seaborn as sns

from Experiment import load


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


def plot_triggers(ex_obj, fno):
    sig = ex_obj.ae.readAE(fno)
    ts = 1 / ex_obj.test_info.acquisition[0]
    n = sig.size
    t = np.arange(0, n) * ts
    filename = f'Test {fno:03d}'

    # en_sig = AE.envelope_hilbert(sig)
    # fil_sig = AE.low_pass(en_sig, 10, 2000000, 3)

    trigs = exp.ae.trig_points.loc[fno]

    plt.close()
    plt.figure()
    plt.plot(t, sig, linewidth=1)
    plt.axvline(trigs['trig st'] * ts, color='r')
    plt.axvline(trigs['trig end'] * ts, color='r')
    # plt.axhline(trigs['trig y-val'], color='r')
    plt.title(filename)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.show()


if __name__ == '__main__':
    exp = load(file='Test 5')
    plot_triggers(exp, 150)

    # envelope_hilbert(sig)

    # ae_hits(exp, sig[0:3000000])
    # feat = exp.features.copy()
    # feat.drop([0, 1, 23, 24])
    # c = corr_matrix(feat.drop([0, 1, 23, 24]), save_fig=False)
    # corr_pairplot(feat.drop([0, 1, 23, 24]), save_fig=False)
