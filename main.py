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
from resources import load, ae


if __name__ == '__main__':
    exp = load('Test 9')
    exp.ae.plotAE(157)

    rms = ae.RMS('Test 9')
    rms.plot_rms(157)
    plt.show()
