#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@File    :   ae.py
@Author  :   Tom Jessel
@Contact :   jesselt@cardiff.ac.uk

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
22/08/2022 13:46   tomhj      1.0        File which handles AE within exp obj.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import multiprocessing
import os
from functools import partial
from pathlib import PurePosixPath as Path
from typing import List, Tuple, Any, Union

import matplotlib.pyplot as plt
import mplcursors
import numpy as np
import pandas as pd
from nptdms import TdmsFile
from scipy.stats import kurtosis
from scipy.stats import skew
from tqdm import tqdm
from scipy.signal import hilbert, butter, filtfilt
import tkinter as tk
import glob
import re

from .. import config


HOME_DIR, BASE_DIR, CODE_DIR, TB_DIR, RMS_DATA_DIR = config.config_paths()


def butter_filter(
        data: np.ndarray,
        ftype: str = 'low',
        fs: float = 2_000_000,
        order: int = 3,
) -> np.ndarray:
    """
        Apply butterworth filter to input data.

    Args:
        data: Input signal for filtering
        ftype: Type of filter, (low, high, band)
        fs: Sample freq for the input signal
        order: Filter order

    Returns:
        Filtered signal

    """
    data = np.pad(data, pad_width=10_000)
    b, a = butter(N=order,
                  Wn=10,
                  fs=fs,
                  btype=ftype,
                  analog=False,
                  output='ba'
                  )
    y = filtfilt(b, a, data)
    return y[10_000:-10_000]


def rms(x: np.ndarray) -> np.ndarray:
    """
    Calculate root-mean squared of a np.array.

    Args:
        x: Input array to calculate for

    Returns:
        RMS of inputted array
    """
    r = np.sqrt(np.mean(x ** 2))
    return r


def envelope_hilbert(s: Union[np.ndarray, list]) -> np.ndarray:
    """
    Envelope a signal with the hilbert transform and return the
    instantaneous amplitude.

    Args:
        s: Input signal to be enveloped

    Returns:
        Instantaneous amplitude of the hilbert enveloped input signal
    """
    z = hilbert(s)
    inst_amp = np.abs(z)
    return inst_amp


def trigger_st(
        d: Union[np.ndarray, list],
        chunk_size: int = 100_000,
        diff_change: float = 1.75E-6,
) -> List[Union[int, float]]:
    """
    Find the index of the first trigger point based on change in gradient over\
     a chunk size.

    Args:
        d: Data to find trigger of.
        chunk_size: Size of chunks to calculate gradient over.
        diff_change: Threshold for change in gradient to find trigger.

    Returns:
        A tuple comprimising of both the trigger index location and data value\
         at that point.
        [t, t_y]: Trigger index, Data value at trigger point

    """
    n_chunks = len(d) / chunk_size

    chunks = np.array_split(d, n_chunks)
    grad = [(chunki[-1] - chunki[0]) / chunk_size for chunki in chunks]
    if np.max(grad) < diff_change:
        t = None
        t_y = None
    else:
        ind = [(i * chunk_size) + (chunk_size / 2) for i in range(len(grad))]
        zipped = tuple(zip(ind, np.abs(grad)))
        trig = sorted([z for z in zipped if z[1] >= diff_change],
                      key=lambda x: x[0],
                      reverse=False
                      )

        t = (trig[0][0])
        t_y = d[int(t)]
        # t[1] = int(t[0]) + np.argmax(d[int(t[0]):] < t_y)
        # assert t[1] - t[0] > 35_000_000, 'Trigger point selection incorrect'
    return t, t_y


class AE:
    def __init__(
            self,
            ae_files: Tuple[str],
            pre_amp: Any,
            testinfo: Any,
            fs: float,
    ) -> None:
        """
        AE class.

        Args:
            ae_files: File locations for each AE TDMS file.
            pre_amp: Pre-Amp object, containing pre-amp info for the\
                experiment.
            testinfo: Test Info object, containing testing info for the\
                experiment.
            fs: Sample rate for the AE acquisition during the test.
        """
        self._files = ae_files
        self.kurt = []
        self.rms = []
        self.amplitude = []
        self.skewness = []
        self._pre_amp = pre_amp
        self._fs = fs
        self._testinfo = testinfo
        self.fft = {}
        self.trig_points = pd.DataFrame()

    @staticmethod
    def volt2db(v: np.ndarray) -> list:
        """Converts array from volts to dB.

        Calculates the equivalent amplitude dB for a given input array,\
            based on typical AE values of; V_ref = 1E-4.

        Args:
            v: Voltage array for converting.

        Returns:
            The equivalent array converted to decibels.
        """
        v_ref = 1E-4
        db = [20 * (np.log10(vin / v_ref)) for vin in v]
        return db

    def _fftcalc(self, fno: int, freqres: float) -> np.ndarray:
        """
        Function to calculate the fft of an AE signal within the experiment.

        Args:
            fno: File number to calculate the fft for.
            freqres: Resolution of the fft, i.e. the size of averaging\
                for the fft.

        Returns:
            fft_mean: FFT of the signal averaged over bands specified by\
                freqres.

        """
        length = int(self._fs / freqres)
        data = self.readAE(fno)
        trig = self.trig_points.loc[fno]
        data = data[int(trig['trig st']):int(trig['trig end'])]
        # data = envelope_hilbert(data)
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

    def readAE(self, fno: int) -> list:
        """
        Read AE data from TDMS file and scale.

        Args:
            fno: TDMS file number to read into memory.

        Returns:
            data: AE data from the TDMS file.

        """
        filepath = CODE_DIR.joinpath(self._files[fno])
        test = TdmsFile.read(filepath)
        prop = test.properties
        data = []
        for group in test.groups():
            for channel in group.channels():
                data = channel[:]
        if not data.dtype == float:
            data = (data.astype(np.float64) * prop.get('Gain')) + prop.get(
                'Offset'
            )
        if not self._pre_amp.gain == 40:
            if self._pre_amp.gain == 20:
                data = data * 10
            elif self._pre_amp.gain == 60:
                data = data / 10
        return data

    def plotAE(self, fno: int, ax: plt.axes = None) -> Any:
        """
        Plot AE from file number.

        Args:
            fno: File number to plot the AE of.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        signal = self.readAE(fno)
        ts = 1 / self._fs
        n = len(signal)
        t = np.arange(0, n) * ts
        filename = self._files[fno].partition('_202')[0]
        filename = filename[-8:]
        ax.plot(t, signal, linewidth=1)
        ax.set_title(filename)
        ax.autoscale(enable=True, axis='x', tight=True)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage (V)')
        mplcursors.cursor(multiple=True)
        # fig.show()
        return fig, ax

    def plotfft(self, fno: int, freqres: float = 1000) -> None:
        """
        Plot fft of AE signal for given file number.

        Args:
            fno: File number to plot fft of.
            freqres: Resolution of the fft, through averaging.

        """
        if freqres in self.fft:
            p = self.fft[freqres][fno]
        else:
            p = self._fftcalc(fno, freqres)
        f = np.arange(0, self._fs / 2, freqres, dtype=int)

        filename = self._files[fno].partition('_202')[0]
        filename = filename[-8:]
        fig, ax = plt.subplots()
        ax.plot(f / 1000, p)
        ax.set_title(
            f'Test No: {self._testinfo.testno} - FFT File {filename[-3:]}'
        )
        ax.autoscale(enable=True, axis='x', tight=True)
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('Amplitude (dB)')
        ax.grid()
        mplcursors.cursor(hover=True)
        fig.show()
        return fig, ax

    def process(self, trigger: bool = True, FFT: bool = False) -> None:
        """
        Process the AE data calculating the crucial features.

        Multiprocessing function to process AE data saving the common features\
                within the AE object, with the options
        to calculate between the trigger points and calculate the 1kHz fft.

        Args:
            trigger: Option to calculate features within the trigger points.
            FFT: Option to calculate the 1kHz fft for each signal.

        """
        with multiprocessing.Pool(processes=20) as pool:
            if self.trig_points.empty and trigger:
                trigs = list(tqdm(pool.imap(self._triggers,
                                            range(len(self._files))
                                            ),
                                  total=len(self._files),
                                  desc='Triggers'))
                self.trig_points = pd.DataFrame(trigs,
                                                columns=['trig st',
                                                         'trig end',
                                                         'trig y-val'
                                                         ],
                                                )
            # print('Calculating results...')
            results = list(tqdm(pool.imap(self._calc, range(len(self._files))),
                                total=len(self._files),
                                desc='AE features'))

            results = np.array(results)
        pool.close()

        self.kurt = results[:, 0]
        self.rms = results[:, 1]
        self.amplitude = results[:, 2]
        self.skewness = results[:, 3]

        if 1000 not in self.fft or FFT:
            with multiprocessing.Pool(processes=20) as pool:
                # print('Calculating FFT with 1kHz bins...')
                fft = list(tqdm(pool.imap(partial(self._fftcalc, freqres=1000),
                                          range(len(self._files))
                                          ),
                                total=len(self._files),
                                desc='Calc FFT 1 kHz'))
                p = self.volt2db(np.array(fft))
                self.fft[1000] = p
        pool.close()

    def _calc(self, fno: int) -> List[np.ndarray]:
        """
        Function for multiprocessing to calculate all the time driven AE\
            results of an AE signal.

        Args:
            fno: File number to calculate features for.

        Returns:
            A tuple containg the kurtosis, rms, amplitude and skewness of the\
            signal (k, r, a, sk)
        """
        data = self.readAE(fno)
        trig = self.trig_points.loc[fno]
        data = data[int(trig['trig st']):int(trig['trig end'])]
        r = rms(data)
        k = kurtosis(data, fisher=False)
        a = data.max()
        sk = skew(data)
        # print(f'Completed File {fno}...')
        return k, r, a, sk

    def _triggers(self, fno: int) -> List[Union[int, float]]:
        """
        Compute the start and end trigger indicies, as well as the y-value at\
            those points.

        Args:
            fno: File number to find the triggers of.

        Returns:
            A tuple containing the start index, end index, and y value\
            (trig_st, trig_end, trig_y_val).

        """
        sig = self.readAE(fno)
        e_sig = envelope_hilbert(sig[:6_000_000])
        f_sig = butter_filter(data=e_sig, fs=self._fs, order=3, ftype='low')
        trig, trig_y_val = trigger_st(f_sig[100_000:])
        if trig is None:
            trig_st = 0
            trig_end = len(sig)
            trig_y_val = 0
        else:
            trig_st = trig + 100_000
            en_trig2 = envelope_hilbert(sig[-6_000_000:])
            fil_trig2 = butter_filter(data=en_trig2,
                                      fs=self._fs,
                                      order=3,
                                      ftype='low'
                                      )
            trig_end = (
                5_900_000 - np.argmax(fil_trig2[100_000:] < trig_y_val)
            )
            if trig_end == 5_900_000:
                en_trig2 = envelope_hilbert(sig[6_000_000:-6_000_000])
                fil_trig2 = butter_filter(data=en_trig2,
                                          fs=self._fs,
                                          order=3,
                                          ftype='low'
                                          )
                trig_end = 6_100_000 + (np.argmax(
                    fil_trig2[100_000:] < trig_y_val
                ))
            else:
                trig_end = len(sig) - trig_end
        return trig_st, trig_end, trig_y_val

    def fftsurf(
            self,
            freqres: float = 1000,
            freqlim: Union[None, list] = None
    ) -> None:
        """
        Plot a 3D surface of the fft of each AE signal in the experiment.

        Args:
            freqres: Resolution of the ffts to plot the surface with.
            freqlim: Limits of the freq axis for the surface.

        """
        if freqres in self.fft:
            p = self.fft[freqres]
        else:
            with multiprocessing.Pool() as pool:
                fft = list(tqdm(
                    pool.imap(partial(self._fftcalc, freqres=freqres),
                              range(len(self._files))),
                    total=len(self._files),
                    desc=f'Calc FFT  {freqres / 1000} kHz'))
                p = self.volt2db(np.array(fft))
            self.fft[freqres] = p

        if freqlim is None:
            freqlim = {'lowlim': int(0 / freqres),
                       'uplim': int(self._fs / (2 * freqres))
                       }
        else:
            freqlim = {'lowlim': int(freqlim[0] / freqres),
                       'uplim': int(freqlim[1] / freqres)
                       }
        f = np.arange(0, self._fs / 2, freqres, dtype=int)
        n = np.arange(0, len(self._files))
        p = np.array(p)
        f = f[freqlim['lowlim']:freqlim['uplim']]
        p = p[:, freqlim['lowlim']:freqlim['uplim']]

        x, y = np.meshgrid(f, n)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(x, y, p, cmap='jet')
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('Measurement Number')
        ax.set_zlabel('Amplitude (dB)')
        ax.set_title(f'Test No: {self._testinfo.testno} - FFT')
        fig.show()
    
    def fft_2d_surf(self, freqlim: Union[None, list] = None):
        p = np.array(self.fft[1000])
        if freqlim is None:
            freqlim = {'lowlim': int(0 / 1000),
                       'uplim': int(self._fs / (2 * 1000))
                       }
        else:
            freqlim = {'lowlim': int(freqlim[0] / 1000),
                       'uplim': int(freqlim[1] / 1000)
                       }
        p = p[:, freqlim['lowlim']:freqlim['uplim']]

        fig, ax = plt.subplots()
        im = ax.imshow(p,
                       origin='lower',
                       interpolation='bilinear',
                       aspect='auto',
                       cmap='jet'
                       )
        fig.colorbar(im, ax=ax, label='Amplitude (dB)')
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('Measurement No')
        xticks = ax.get_xticks()
        xticks_labels = [(xt + freqlim['lowlim']) for xt in xticks]
        # ax.set_xticks(xticks)
        ax.set_xticklabels(xticks_labels)
        return fig

    def rolling_rms(self, fno: Union[int, tuple], plot_fig: bool = True) -> \
            Union[np.ndarray, None]:
        """
        Produces either a figure or mp4/gif of the specified files.

        Args:
            fno: File numeber to produce the rolling rms for. For figure\
            single int input.
                For gif/mp4 tuple of start and end file.

        Returns:
            Array of the rolling rms of the selected file.\
            Only with single input file.
        """
        import moviepy.editor as mp
        from matplotlib.animation import PillowWriter

        def calc_roll_rms(i: int, win_size: float = 500_000) -> np.ndarray:
            """
            Calculate the rolling rms of the AE file

            Args:
                i: File number of the AE file to calculate for.
                win_size: the size of the rolling window size.

            Returns:
                An array of the calculated rolling rms for the specified file.
            """
            data = self.readAE(i)
            data = pd.DataFrame(data)
            v = data.pow(2).rolling(win_size).mean().apply(np.sqrt, raw=True)
            # v = v[499_999:].to_numpy()
            return v

        def mp4_conv(gifname: str) -> None:
            """
            Convert a gif file to mp4 format.

            Args:
                gifname: Full gif file name.

            """
            clip = mp.VideoFileClip(gifname)
            mp4name = gifname[:-4] + '.mp4'
            clip.write_videofile(mp4name)

        ts = 1 / self._fs
        if type(fno) is int:
            v_rms = calc_roll_rms(fno)
            n = v_rms.size
            t = np.arange(0, n) * ts
            if plot_fig:
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
            dataloc = CODE_DIR.joinpath(self._testinfo.dataloc)
            path = dataloc.joinpath('Figures/')
            name = f'{path}Test {self._testinfo.testno} - Rolling RMS.gif'
            with writer.saving(fig, name, 200):
                for no in range(fno[0], fno[1]):
                    x = np.arange(0, n) * ts
                    y = calc_roll_rms(no)
                    l.set_data(x, y)
                    ax.set_title(f'File - {no:03d}')
                    writer.grab_frame()
            mp4_conv(name)

    def update(self, files: Tuple[str]) -> None:
        """
        Update the file location list stored in this object.

        Args:
            files: Tuple of strings containing the paths to each AE file

        """
        self._files = files

    def plot_triggers(self, fno: int, ax: plt.axes = None) -> None:
        """
        Plot calculated trigger points of the file on the hibert enveloped\
            and lowpass filtered AE signal

        Args:
            fno: File number to show triggers

        """
        try:
            triggers = self.trig_points.loc[fno]
        except KeyError:
            print('No Triggers calculated for this test')
            print('Process AE data with triggers then retry')
            return

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        sig = self.readAE(fno)
        ts = 1 / self._fs
        n = len(sig)
        t = np.arange(0, n) * ts
        filename = f'Test {fno:03d} - Triggers of enveloped & filtered' \
                   f'AE signal'

        en_sig = envelope_hilbert(sig)
        sig = butter_filter(data=en_sig, fs=self._fs, order=3, ftype='low')

        ax.plot(t, sig, linewidth=1)
        ax.axhline(triggers['trig y-val'],
                   color='r', linewidth=1, alpha=0.8, linestyle='--')
        ax.axvline(triggers['trig st'] * ts,
                   color='r', linewidth=1, alpha=0.8, linestyle='--')
        ax.axvline(triggers['trig end'] * ts,
                   color='r', linewidth=1, alpha=0.8, linestyle='--')
        ax.set_title(filename)
        ax.autoscale(enable=True, axis='x', tight=True)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage (V)')
        return fig, ax


class RMS:
    def __init__(self,
                 exp_obj: str = None,
                 pre_amp_gain: int = 20,
                 ) -> None:
        
        self._data = None
        self._folder = None
        self._pre_amp_gain = pre_amp_gain
        self.exp_name = 'Test_99'

        if type(exp_obj) is str:
            ref_test_loc = CODE_DIR.joinpath(
                'src/reference/Test obj locations.txt'
            )
            f_locs = pd.read_csv(ref_test_loc, sep=',', index_col=0)
            f_locs = f_locs.to_dict()['Obj location']
            f_locs = {k: BASE_DIR.joinpath(r'AE/Testing', v)
                      for k, v in f_locs.items()}
            try:
                self._folder = f_locs[exp_obj.lower().replace(' ', '')].parent
            except KeyError:
                print('Test object not found in reference file')
                exp_obj = None

        if exp_obj is None:
            folder = tk.filedialog.askdirectory(
                title="Select the experiment's folder",
                initialdir=CODE_DIR.parents[1].joinpath('Testing'),
                mustexist=True,
            )
            self._folder = Path(folder)
        
        files = [f for f in list(os.walk(self._folder))[0][2]]
        for f in files:
            if re.search(r'Test \d.pickle', f):
                self.exp_name = f.split('.pickle')[0]

        self.no_files = self.data.shape[1]

        print('-' * 50)
        print(f'Loaded RMS data for "{self._folder.parts[-1]}"')
        print(f'Experiemnt No: {self.exp_name}')
        print(f'Number of files: {self.no_files}')
        print('-' * 50)

    @property
    def data(self) -> pd.DataFrame:
        if self._data is not None:
            return self._data
        else:
            self._data = self._get_data()
            return self._data
        
    def _get_data(self) -> pd.DataFrame:
        try:
            file = self._folder.joinpath('AE_RMS.csv')
            return pd.read_csv(file)
        except FileNotFoundError:
            print(f'Processing RMS data for {self._folder.parts[-1]}')
            return self._process_rms()

    def _process_rms(self) -> pd.DataFrame:
        assert os.path.exists(self._folder.joinpath('AE'))
        self._aefiles = glob.glob(os.path.join(self._folder, 'AE/TDMS/*.tdms'))
        no_files = len(self._aefiles)
        fnos = range(no_files)

        with multiprocessing.Pool() as pool:
            rms = list(tqdm(pool.imap(self._calc_rms, fnos),
                            total=no_files,
                            desc='Calculating RMS',
                            ))
        pool.close()
        m = min([r.shape[0] for r in rms])
        rms = [r[:m] for r in rms]
        rms = np.array(rms)

        data = pd.DataFrame(rms.T)
        data.to_csv(self._folder.joinpath('AE_RMS.csv'),
                    index=False,
                    encoding='utf-8',
                    )
        print(f'AE RMS data saved to {self._folder.joinpath("AE_RMS.csv")}')
        return data

    def _readAE(self, fno: int) -> np.ndarray:
        """
        Read the AE data from the specified file

        Args:
            fno: File number to read

        Returns:
            AE data as a numpy array

        """
        file = self._aefiles[fno]
        test = TdmsFile(file)
        prop = test.properties
        data = []
        for group in test.groups():
            for channel in group.channels():
                data = channel[:]
        if not data.dtype == float:
            data = (data.astype(np.float64) * prop.get('Gain')) + prop.get(
                'Offset'
            )
        if not self._pre_amp_gain == 40:
            if self._pre_amp_gain == 20:
                data = data * 10
            elif self._pre_amp_gain == 60:
                data = data / 10
        return data

    def _calc_rms(self, fno: int) -> np.ndarray:
        sig = self._readAE(fno)
        avg_size = 100000
        sig = pd.DataFrame(sig)
        sig = sig.pow(2).rolling(500000).mean().apply(np.sqrt, raw=True)
        sig = np.array(sig)[500000 - 1:]
        avg_sig = np.nanmean(np.pad(sig.astype(float),
                                    ((0, avg_size - sig.size % avg_size),
                                     (0, 0)
                                     ),
                                    mode='constant',
                                    constant_values=np.NaN
                                    )
                             .reshape(-1, avg_size), axis=1)
        return avg_sig

    def plot_rms(self,
                 fno: Union[int, list, tuple],
                 ax: plt.axes = None
                 ) -> Any:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if type(fno) is tuple or type(fno) is list:
            fno = range(*fno)

        fno = [fno] if type(fno) is int else fno

        for n in fno:
            ax.plot(self.data.iloc[:, n],
                    linewidth=0.75,
                    label=f'RMS {n}'
                    )
        ax.set_ylabel('RMS (V)')
        ax.autoscale(enable=True, axis='x', tight=True)
        if len(fno) > 1:
            ax.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
        else:
            ax.legend()
        fig.tight_layout()
        return fig, ax
