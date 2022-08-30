#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@File    :   AE.py
@Author  :   Tom Jessel
@Contact :   jesselt@cardiff.ac.uk

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
22/08/2022 13:46   tomhj      1.0         File which handles AE operations within experiment object
"""

import multiprocessing
from functools import partial
import matplotlib as mpl
import matplotlib.pyplot as plt
import moviepy.editor as mp
import mplcursors
import numpy as np
import pandas as pd
from matplotlib.animation import PillowWriter
from nptdms import TdmsFile
from scipy.stats import kurtosis


def rms(x):
    r = np.sqrt(np.mean(x ** 2))
    return r


class AE:
    def __init__(self, ae_files, pre_amp, fs, testinfo):
        self._files = ae_files
        self.kurt = []
        self.rms = []
        self._pre_amp = pre_amp
        self._fs = fs
        self._testinfo = testinfo
        self.fft = {}

    @staticmethod
    def volt2db(v):
        v_ref = 1E-4
        db = [20 * (np.log10(vin / v_ref)) for vin in v]
        return db

    def fftcalc(self, fno, freqres):
        length = int(self._fs / freqres)
        data = self.readAE(fno)
        if len(data) % length == 0:
            temp = np.reshape(data, (length, -1), order='F')
        else:
            leftover = int(length - np.fmod(len(data), length))
            temp = np.pad(data, (0, leftover), 'constant', constant_values=0)
            temp = np.reshape(temp, (length, -1), order='F')

        win = np.hanning(length)
        win = np.expand_dims(win, axis=-1)
        temp = np.multiply(win, temp)
        sc = len(win) / sum(win)

        fft = np.fft.fft(temp, n=length, axis=0)
        p2 = abs(fft / length)
        p = p2[0:int(np.floor(length / 2)), :]
        p[1:] = p[1:] * 2
        p = np.array(p * sc)
        fft_mean = np.mean(p, axis=1)
        print(f'Calc FFT - File {fno}... ')
        return fft_mean

    def readAE(self, fno):
        test = TdmsFile.read(self._files[fno])
        prop = test.properties
        data = []
        for group in test.groups():
            for channel in group.channels():
                data = channel[:]
        if not data.dtype == float:
            data = (data.astype(np.float) * prop.get('Gain')) + prop.get('Offset')
        if not self._pre_amp.gain == 40:
            if self._pre_amp.gain == 20:
                data = data * 10
            elif self._pre_amp == 60:
                data = data / 10
        return data

    def plotAE(self, fno):
        signal = self.readAE(fno)
        ts = 1 / self._fs
        n = signal.size
        t = np.arange(0, n) * ts
        filename = self._files[fno].partition('_202')[0]
        filename = filename[-8:]
        mpl.use("Qt5Agg")
        plt.figure()
        plt.plot(t, signal, linewidth=1)
        plt.title(filename)
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        mplcursors.cursor(multiple=True)
        plt.show()

    def plotfft(self, fno, freqres=1000):
        if freqres in self.fft:
            p = self.fft[freqres][fno]
        else:
            p = self.fftcalc(fno, freqres)
        f = np.arange(0, self._fs / 2, freqres, dtype=int)

        filename = self._files[fno].partition('_202')[0]
        filename = filename[-8:]
        mpl.use('Qt5Agg')
        plt.figure()
        plt.plot(f / 1000, p, linewidth=0.75)
        plt.title(f'Test No: {self._testinfo.testno} - FFT File {filename[-3:]}')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.xlabel('Frequency (kHz)')
        plt.ylabel('Amplitude (dB)')
        plt.grid()
        mplcursors.cursor(hover=True)
        plt.show()

    def process(self):
        with multiprocessing.Pool() as pool:
            print('Calculating results...')
            results = np.array(pool.map(self._calc, range(len(self._files))))
            if 1000 not in self.fft:
                print('Calculating FFT with 1kHz bins...')
                fft = pool.map(partial(self.fftcalc, freqres=1000), range(len(self._files)))
                p = self.volt2db(np.array(fft))
                self.fft[1000] = p
        pool.close()

        self.kurt = results[:, 0]
        self.rms = results[:, 1]

    def _calc(self, fno):
        data = self.readAE(fno)
        r = rms(data)
        k = kurtosis(data, fisher=False)
        print(f'Completed File {fno}...')
        return k, r

    def fftsurf(self, freqres=1000, freqlim=None):
        if freqres in self.fft:
            p = self.fft[freqres]
        else:
            with multiprocessing.Pool() as pool:
                results = pool.map(partial(self.fftcalc, freqres=freqres), range(len(self._files)))
            p = self.volt2db(np.array(results))
            self.fft[freqres] = p

        if freqlim is None:
            freqlim = {'lowlim': int(0 / freqres), 'uplim': int(self._fs / (2 * freqres))}
        else:
            freqlim = {'lowlim': int(freqlim[0] / freqres), 'uplim': int(freqlim[1] / freqres)}
        f = np.arange(0, self._fs / 2, freqres, dtype=int)
        n = np.arange(0, len(self._files))
        p = np.array(p)
        f = f[freqlim['lowlim']:freqlim['uplim']]
        p = p[:, freqlim['lowlim']:freqlim['uplim']]

        x, y = np.meshgrid(f, n)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(x, y, p, cmap='jet')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Measurement Number')
        ax.set_zlabel('Amplitude (dB)')
        ax.set_title(f'Test No: {self._testinfo.testno} - FFT')
        fig.show()

    def rolling_rms(self, fno):
        """
        Plot either rolling RMS of single AE file or creates animation of all the files in certain range

        :param fno: either int or tuple: int - single graph with return, tuple - range of animation plot
        :return: for single plot: V_rms
        """
        def calc_rms(i):
            data = self.readAE(i)
            data = pd.DataFrame(data)
            v = data.pow(2).rolling(500000).mean().apply(np.sqrt, raw=True)
            v = v[1_000_000:41_000_000].to_numpy()
            return v

        def mp4_conv(gifname):
            clip = mp.VideoFileClip(gifname)
            mp4name = gifname[:-4] + '.mp4'
            clip.write_videofile(mp4name)

        ts = 1 / self._fs
        if type(fno) is int:
            v_rms = calc_rms(fno)
            n = v_rms.size
            t = np.arange(0, n) * ts
            mpl.use('Qt5Agg')
            fig, ax = plt.subplots()
            ax.plot(t, v_rms, linewidth=0.75)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('RMS (s)')
            ax.set_title(f'File {fno} - Rolling RMS')
            ax.autoscale(enable=True, axis='x', tight=True)
            mplcursors.cursor(multiple=True)
            fig.show()
            return v_rms
        elif type(fno) is tuple:
            # Code to create a gif of the RMS plots
            fig, ax = plt.subplots()
            l, = ax.plot([], [])
            ax.set_xlim(0, 20)
            ax.set_ylim(0, 5)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('RMS (V)')
            writer = PillowWriter(fps=1)
            n = 40_000_000  # number of points to plot
            name = f'{self._testinfo.dataloc}\\Test {self._testinfo.testno} - Rolling RMS.gif'
            with writer.saving(fig, name, 100):
                for no in range(fno[0], fno[1]):
                    x = np.arange(0, n) * ts
                    y = calc_rms(no)
                    l.set_data(x, y)
                    ax.set_title(f'File - {no:03d}')
                    writer.grab_frame()
            mp4_conv(name)
