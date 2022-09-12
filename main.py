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
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from Experiment import load


def corr_test():
    rms = exp.ae.rms[:-1]
    kurt = exp.ae.kurt[:-1]
    amp = exp.ae.amplitude[:-1]

    f = np.array(exp.ae.fft[1000])
    f = f.T
    f_35 = f[35][:-1]
    f_10 = f[10][:-1]
    f_134 = f[134][:-1]

    mean_rad = exp.nc4.mean_radius[1:]
    run_out = exp.nc4.runout[1:]
    form_err = exp.nc4.form_error[1:]

    cols = ["RMS", 'Kurtosis', 'Amplitude', 'Freq 10 kHz', 'Freq 35 kHz', 'Freq 134 kHz',
            'Mean radius', 'Runout', 'Form error']

    m = np.stack((rms, kurt, amp, f_10, f_35, f_134, mean_rad, run_out, form_err), axis=0)
    df = pd.DataFrame(m.T, columns=cols)
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

    sns.pairplot(df, height=1)
    fig.show()
    return df.corr()


if __name__ == '__main__':
    exp = load(file='Test 5')
    table = corr_test()
