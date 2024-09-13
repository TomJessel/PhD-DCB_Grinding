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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from src import load


if __name__ == '__main__':
    exp = load('Test 21')
    # rms = ae.RMS('Test 9')
    
    # fig, ax = exp.nc4.plot_surf()
    # fig = exp.nc4.plot_att()
    # fig, ax = plt.subplots(2, 1)
    # ax[0] = exp.ae.plotAE(150, ax=ax[0])
    # ax[1] = rms.plot_rms(150, ax=ax[1])
    
    for i in range(0, len(exp.features)):
        rms = exp.ae.rolling_rms(i, plot_fig=False)
        x = np.arange(len(rms)) * 1 / exp.ae._fs
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        ax.plot(x, rms)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('RMS (V)')
        ax.set_xlim(0, x[-1])
        ax.set_ylim(0, 1.5)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.set_title(f'Test 8 - Pass {i}')
        plt.savefig(f'GIF/Test_21/Pass_{i}.png')
        plt.close()
    # plt.show()
