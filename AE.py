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
from tqdm import tqdm
from scipy.signal import hilbert, butter, filtfilt
import time


def rms(x):
    r = np.sqrt(np.mean(x ** 2))
    return r


def low_pass(data, cutoff, fs, order):
    data = np.pad(data, pad_width=10_000)
    norm_cutoff = cutoff / (0.5 * fs)
    b, a = butter(N=order, Wn=norm_cutoff, btype='lowpass', analog=False)
    y = filtfilt(b, a, data)
    return y[10_000:-10_000]


def envelope_hilbert(s):
    z = hilbert(s)
    inst_amp = np.abs(z)
    return inst_amp


def trigger_st(d):
    chunk_size = 100_000
    diff_change = 1.75E-6

    n_chunks = len(d) / chunk_size
    t = []

    chunks = np.array_split(d, n_chunks)
    grad = [(chunki[-1] - chunki[0]) / chunk_size for chunki in chunks]
    if np.max(grad) < diff_change:
        t = None
        t_y = None
    else:
        ind = [(i * chunk_size) + (chunk_size / 2) for i in range(len(grad))]
        zipped = tuple(zip(ind, np.abs(grad)))
        trig = sorted([z for z in zipped if z[1] >= diff_change], key=lambda x: x[0], reverse=False)

        t = (trig[0][0])
        t_y = d[int(t)]
        # t[1] = int(t[0]) + np.argmax(d[int(t[0]):] < t_y)
        # assert t[1] - t[0] > 35_000_000, 'Trigger point selection incorrect'
    return t, t_y


class AE:
    def __init__(self, ae_files, pre_amp, fs, testinfo):
        self._files = ae_files
        self.kurt = []
        self.rms = []
        self.amplitude = []
        self._pre_amp = pre_amp
        self._fs = fs
        self._testinfo = testinfo
        self.fft = {}
        self.trig_points = pd.DataFrame()

    @staticmethod
    def volt2db(v):
        v_ref = 1E-4
        db = [20 * (np.log10(vin / v_ref)) for vin in v]
        return db

    def fftcalc(self, fno, freqres):
        length = int(self._fs / freqres)
        data = self.readAE(fno)
        trig = self.trig_points.loc[fno]
        data = data[int(trig['trig st']):int(trig['trig end'])]
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
        # print(f'Calc FFT - File {fno}... ')
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

    def process(self, trigger=False, FFT=False):
        with multiprocessing.Pool() as pool:
            if self.trig_points.empty or trigger:
                trigs = list(tqdm(pool.imap(self.triggers, range(len(self._files))),
                                  total=len(self._files),
                                  desc='Triggers'))
                self.trig_points = pd.DataFrame(trigs,
                                                columns=['trig st', 'trig end', 'trig y-val'],
                                                )
            # print('Calculating results...')
            results = list(tqdm(pool.imap(self._calc, range(len(self._files))),
                                total=len(self._files),
                                desc='AE features'))

            results = np.array(results)
            if 1000 not in self.fft or FFT:
                # print('Calculating FFT with 1kHz bins...')
                fft = list(tqdm(pool.imap(partial(self.fftcalc, freqres=1000), range(len(self._files))),
                                total=len(self._files),
                                desc='Calc FFT 1 kHz'))
                p = self.volt2db(np.array(fft))
                self.fft[1000] = p
        pool.close()

        self.kurt = results[:, 0]
        self.rms = results[:, 1]
        self.amplitude = results[:, 2]

    def _calc(self, fno):
        data = self.readAE(fno)
        trig = self.trig_points.loc[fno]
        data = data[int(trig['trig st']):int(trig['trig end'])]
        r = rms(data)
        k = kurtosis(data, fisher=False)
        a = data.max()
        # print(f'Completed File {fno}...')
        return k, r, a

    def triggers(self, i):
        sig = self.readAE(i)
        e_sig = envelope_hilbert(sig[:6_000_000])
        f_sig = low_pass(e_sig, 10, self._fs, 3)
        trig, trig_y_val = trigger_st(f_sig[100_000:])
        if trig is None:
            trig_st = 0
            trig_end = len(sig)
            trig_y_val = 0
        else:
            trig_st = trig + 100_000
            en_trig2 = envelope_hilbert(sig[-6_000_000:])
            fil_trig2 = low_pass(en_trig2, 10, self._fs, 3)
            trig_end = (5_900_000 - np.argmax(fil_trig2[100_000:] < trig_y_val))
            if trig_end == 5_900_000:
                en_trig2 = envelope_hilbert(sig[6_000_000:-6_000_000])
                fil_trig2 = low_pass(en_trig2, 10, self._fs, 3)
                trig_end = 6_100_000 + (np.argmax(fil_trig2[100_000:] < trig_y_val))
            else:
                trig_end = len(sig) - trig_end
        return trig_st, trig_end, trig_y_val

    def fftsurf(self, freqres=1000, freqlim=None):
        if freqres in self.fft:
            p = self.fft[freqres]
        else:
            with multiprocessing.Pool() as pool:
                fft = list(tqdm(pool.imap(partial(self.fftcalc, freqres=freqres), range(len(self._files))),
                                total=len(self._files),
                                desc=f'Calc FFT  {freqres / 1000} kHz'))
                p = self.volt2db(np.array(fft))
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
            with writer.saving(fig, name, 200):
                for no in range(fno[0], fno[1]):
                    x = np.arange(0, n) * ts
                    y = calc_rms(no)
                    l.set_data(x, y)
                    ax.set_title(f'File - {no:03d}')
                    writer.grab_frame()
            mp4_conv(name)

    def update(self, files):
        self._files = files
