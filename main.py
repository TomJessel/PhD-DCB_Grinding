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
from src import load


if __name__ == '__main__':
    exp = load('Test 11')
    # rms = ae.RMS('Test 9')
    
    fig, ax = exp.nc4.plot_surf()
    fig = exp.nc4.plot_att()
    # fig, ax = plt.subplots(2, 1)
    # ax[0] = exp.ae.plotAE(150, ax=ax[0])
    # ax[1] = rms.plot_rms(150, ax=ax[1])

    plt.show()
