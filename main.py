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
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
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


if __name__ == '__main__':
    exp = load(file='Test 5')
    feat = exp.features.copy()
    feat.drop([0, 1, 23, 24])
    c = corr_matrix(feat.drop([0, 1, 23, 24]), save_fig=False)
    corr_pairplot(feat.drop([0, 1, 23, 24]), save_fig=False)
