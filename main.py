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
from matplotlib import pyplot as plt

from Experiment import load

if __name__ == '__main__':
    exp = load(file='Test 2')

    # f = np.array(exp.ae.fft[1000])
    # f = f.T
    # mean_rad = exp.nc4.mean_radius[1:]
    #
    # fig, ax = plt.subplots(2, 1)
    # ax[0].scatter(f[4][:-1], mean_rad)
    # ax[1].scatter(f[973][:-1], mean_rad)
    # c = np.corrcoef(f[:, :-1], mean_rad)
    #

    # rms = exp.ae.rms[:-1]
    # kurt = exp.ae.kurt[:-1]
    # mean_rad = exp.nc4.mean_radius[1:]
    # runout = exp.nc4.runout[1:]
    # m = np.stack((rms, kurt, mean_rad, runout), axis=0)
    # c = np.corrcoef(m)
    #
    # num_vectors = len(m)
    # fig, ax = plt.subplots(num_vectors, num_vectors)
    # labels = ['rms', 'kurtosis', 'mean radius', 'runout']
    #
    # for i in range(num_vectors):
    #     for j in range(num_vectors):
    #
    #         # Scatter column_j on the x-axis vs. column_i on the y-axis
    #         if i != j:
    #             ax[i][j].scatter(m[j], m[i])
    #
    #         # unless i == j, in which case show the series name
    #         else:
    #             ax[i][j].annotate(labels[i], (0.5, 0.5),
    #                               xycoords='axes fraction',
    #                               ha="center", va="center")
    #
    #         # Then hide axis labels except left and bottom charts
    #         if i < num_vectors - 1: ax[i][j].xaxis.set_visible(False)
    #         if j > 0: ax[i][j].yaxis.set_visible(False)
    #
    # # Fix the bottom-right and top-left axis labels, which are wrong because
    # # their charts only have text in them
    # ax[-1][-1].set_xlim(ax[0][-1].get_xlim())
    # ax[0][0].set_ylim(ax[0][1].get_ylim())
    #
    # plt.show()
