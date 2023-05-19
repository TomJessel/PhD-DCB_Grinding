#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@File    :   ae.py
@Author  :   Tom Jessel
@Contact :   jesselt@cardiff.ac.uk

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
22/08/2022 13:46   tomhj      1.0         File which handles NC4 operations
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from typing import Any, Union
from pathlib import PurePosixPath as Path
from nptdms import TdmsFile
from numpy import ndarray
from tqdm import tqdm
import numpy as np
import multiprocessing
import math
from scipy.ndimage.filters import uniform_filter1d
from scipy import signal
import circle_fit
import matplotlib.pyplot as plt
import mplcursors
import pickle
import pandas as pd

PLATFORM = os.name
if PLATFORM == 'posix':
    ONEDRIVE_PATH = Path(
        r'/mnt/c/Users/tomje/OneDrive - Cardiff University/Documents/PHD/'
    )
    ONEDRIVE_PATH = ONEDRIVE_PATH.joinpath('AE/PYTHON/Acoustic-Emission')
elif PLATFORM == 'nt':
    ONEDRIVE_PATH = Path(
        r'C:\Users\tomje\OneDrive - Cardiff University\Documents\PHD\AE'
    )
    ONEDRIVE_PATH = ONEDRIVE_PATH.joinpath('PYTHON/Acoustic-Emission')


def compute_shift(zipped: tuple[Any, Any]) -> int:
    """
    Use fft correlation to compute the shift between two signals.

    Args:
        zipped: Zip tuple containing the signals to compare between each other.

    Returns:
        Int representing the number of samples of shift between the signals.
    """
    x = zipped[0]
    y = zipped[1]
    assert len(x) == len(y)
    c = signal.correlate(x, y, mode='same', method='fft')
    assert len(c) == len(x)
    zero_index = int(len(x) / 2) - 1
    shift = zero_index - np.argmax(c)
    return shift


class NC4:
    def __init__(
            self,
            files: Union[tuple[str], list[str]],
            testinfo: Any,
            dcb: Any,
            fs: float,
    ) -> None:
        """
        NC4 class.

        Args:
            files: File loactions for each NC4 TDMS file.
            testinfo: Test info obj, containing test info for the experiment.
            dcb: DCB obj, containing info about the DCB used for the test.
            fs: Sample rate for the NC4 acquisition during the test.
        """
        self.theta = None
        self.radius = pd.Series(np.nan, index=np.arange(len(files)))
        # todo change to None and cond for updating to if none or [0] is none
        self.form_error = pd.Series(np.nan, index=np.arange(len(files)))
        self.runout = pd.Series(np.nan, index=np.arange(len(files)))
        self.peak_radius = pd.Series(np.nan, index=np.arange(len(files)))
        self.mean_radius = pd.Series(np.nan, index=np.arange(len(files)))
        self._files = files
        self._dcb = dcb
        self._fs = fs
        self._datano = np.arange(0, len(files))
        self._testinfo = testinfo

    def readNC4(self, fno: int) -> list[float]:
        """
        Read NC4 data from TDMS file into memory.

        Args:
            fno: TDMS file number to read into memory

        Returns:
            NC4 data from the file.
        """
        filepath = ONEDRIVE_PATH.joinpath(self._files[fno])
        test = TdmsFile.read(filepath)
        prop = test.properties
        data = []
        for group in test.groups():
            for channel in group.channels():
                data = channel[:]
        if not data.dtype == float:
            data = (
                data.astype(np.float64) * prop.get('Gain')
            ) + prop.get('Offset')
        return data

    def process(self) -> None:
        """
        Function to process the NC4 data from a voltage to radius and \
            compute features.
        """
        # print('Processing NC4 data...')
        # st1 = time.time()
        with multiprocessing.Pool() as pool:
            results = list(tqdm(pool.imap(
                self._sampleandpos,
                range(len(self._files)),
                chunksize=10),
                total=len(self._files),
                desc='NC4- Sampling'))
        pool.close()
        # en = time.time()
        # print(f'Sampling done {en - st1:.1f} s...')

        psample = [tple[0] for tple in results]
        posy = ([tple[1] for tple in results])
        nsample = [tple[2] for tple in results]
        negy = ([tple[3] for tple in results])

        p = zip(psample, posy)
        n = zip(nsample, negy)

        # st = time.time()
        prad, nrad = self.sigtorad(p, n)
        # en = time.time()
        # print(f'Converting done {en - st:.1f} s...')

        # st = time.time()
        radii = self._alignposneg(prad, nrad)
        radius = self._alignsigs(radii)
        self.radius = radius
        # en = time.time()
        # print(f'Aligning done {en - st:.1f} s...')

        # st = time.time()
        mean_radius, peak_radius, runout, form_error = self._fitcircles(radius)
        self.mean_radius = pd.Series(mean_radius)
        self.peak_radius = pd.Series(peak_radius)
        self.runout = pd.Series(runout)
        self.form_error = pd.Series(form_error)
        # en = time.time()
        # print(f'Calc results done {en - st:.1f} s...')
        # print(f'Total time: {en - st1:.1f} s')

    def check_last(self):
        def _compute_nc4(fno):
            psample, posy, nsample, negy = self._sampleandpos(fno=fno)
            prad = self.polyvalradius((psample, posy))
            nrad = self.polyvalradius((nsample, negy))
            prad = np.transpose(np.reshape(prad, (-1, 1)))
            nrad = np.transpose(np.reshape(nrad, (-1, 1)))
            radii = self._alignposneg(prad, nrad)
            st = np.argmin(radii[0, 0:int(self._fs)])
            rpy = 4
            clip = 0.5
            radius = radii[
                :, np.arange(
                    st, st + (radii.shape[1]) / (rpy - (2 * clip)), dtype=int
                )
            ]
            mean_radius, peak_radius, runout, form_error = self._fitcircles(
                radius
            )
            return mean_radius, peak_radius, runout, form_error

        self.theta = 2 * np.pi * np.arange(0, 1, 1 / self._fs)

        if np.isnan(self.mean_radius[0]):
            (self.mean_radius[0],
             self.peak_radius[0],
             self.runout[0],
             self.form_error[0]
             ) = _compute_nc4(fno=0)

        index = len(self._files) - 1
        if index > 0:
            mean_rad, peak_rad, runout, form_error = _compute_nc4(fno=index)
            self.mean_radius[index] = mean_rad[0]
            self.peak_radius[index] = peak_rad[0]
            self.runout[index] = runout[0]
            self.form_error[index] = form_error[0]

        wear = (self.mean_radius.iloc[-1] - self.mean_radius.iloc[0])
        wear = wear / self.mean_radius.iloc[0] * 100

        print('-' * 60)
        print(f'NC4 - File {index}:')
        print(f'\tMean radius = {self.mean_radius.iloc[-1]:.6f} mm')
        print(f'\tRunout = {self.runout.iloc[-1] * 1000:.3f} um')
        print(f'\tWear = {wear:.3f} %')
        print('-' * 60)
        fig = self.plot_att()
        return fig

    def plot_att(self) -> None:
        """
        Plot NC4 features for each measurement.
        """
        # mpl.use("TkAgg")
        dataloc = ONEDRIVE_PATH.joinpath(self._testinfo.dataloc)
        path = dataloc.joinpath('Figures')
        png_name = f'{path}/Test {self._testinfo.testno} - NC4 Attributes.png'
        pic_name = f'{path}/Test {self._testinfo.testno} ' \
                   f'- NC4 Attributes.pickle'
        if not os.path.isdir(path) or not os.path.exists(path):
            os.makedirs(path)

        fig, ax_r = plt.subplots()
        self.mean_radius.dropna().plot(color='C0',
                                       label='Mean Radius'
                                       )
        self.peak_radius.dropna().plot(color='C1',
                                       label='Peak Radius'
                                       )
        ax_e = ax_r.twinx()
        self.runout.dropna().multiply(1000).plot(color='C2',
                                                 label='Runout'
                                                 )
        self.form_error.dropna().multiply(1000).plot(color='C3',
                                                     label='Form Error'
                                                     )

        ax_r.set_title(f'Test No: {self._testinfo.testno} - NC4 Attributes')
        ax_r.autoscale(enable=True, axis='x', tight=True)
        ax_r.set_xlabel('Measurement No')
        ax_e.set_ylabel('Errors (\u03BCm)')
        ax_r.set_ylabel('Radius (mm)')
        ax_r.grid()
        l1, lab1 = ax_r.get_legend_handles_labels()
        l2, lab2 = ax_e.get_legend_handles_labels()
        ax_e.legend(l1 + l2, lab1 + lab2, loc='upper right', fontsize=9)

        try:
            open(png_name)
        except IOError:
            fig.savefig(png_name, dpi=300)
        try:
            open(pic_name)
        except IOError:
            with open(pic_name, 'wb') as f:
                pickle.dump(fig, f)
        mplcursors.cursor(hover=2)
        return fig
    
    def plot_xy(self, fno: tuple = None, step: int = 1) -> None:
        """
        Plot full radius measurement around tool circumference, \
            for a slice or all measurements.

        Args:
            fno: Tuple of start and stop indices for slice of files to plot.
            step: Step between files of slice.

        """
        dataloc = ONEDRIVE_PATH.joinpath(self._testinfo.dataloc)
        path = dataloc.joinpath('Figures')
        png_name = f'{path}/Test {self._testinfo.testno} - NC4 XY Plot.png'
        pic_name = f'{path}/Test {self._testinfo.testno} - NC4 XY Plot.pickle'

        savefig = False

        if not os.path.isdir(path) or not os.path.exists(path):
            os.makedirs(path)

        if fno is None:
            radius = self.radius
            fig, ax = plt.subplots()
            savefig = True
            n = 0
            for r in radius:
                ax.plot(self.theta, r, label=f'File {n:03.0f}', linewidth=0.5)
                n += 1
            ax.set_xlabel('Angle (rad)')
            ax.set_ylabel('Radius (mm)')
            ax.set_title(f'Test No: {self._testinfo.testno} - NC4 Radius Plot')
            ax.autoscale(enable=True, axis='x', tight=True)
        else:
            fig, ax = plt.subplots()
            slice_n = slice(fno[0], fno[1], step)
            radius = self.radius[slice_n]
            lbl = range(len(self._files))[slice_n]
            n = 0
            for r in radius:
                ax.plot(self.theta,
                        r,
                        label=f'File {lbl[n]:03.0f}',
                        linewidth=0.5
                        )
                n += 1
            ax.set_xlabel('Angle (rad)')
            ax.set_ylabel('Radius (mm)')
            ax.set_title(f'Test No: {self._testinfo.testno} - NC4 Radius Plot')
            ax.autoscale(enable=True, axis='x', tight=True)
        if savefig:
            try:
                open(png_name)
            except IOError:
                fig.savefig(png_name, dpi=300)
            try:
                open(pic_name)
            except IOError:
                with open(pic_name, 'wb') as f:
                    pickle.dump(fig, f)
        mplcursors.cursor(multiple=True)
        return fig

    def plot_surf(self) -> None:
        """
        Plot surface of DCB radius over measurements in time.

        """
        dataloc = ONEDRIVE_PATH.joinpath(self._testinfo.dataloc)
        path = dataloc.joinpath('Figures')
        png_name = f'{path}/Test {self._testinfo.testno} - NC4 Radius Surf.png'
        pic_name = f'{path}/Test {self._testinfo.testno} '\
                   f'- NC4 Radius Surf.pickle'
        if not os.path.isdir(path) or not os.path.exists(path):
            os.makedirs(path)
        try:
            with open(pic_name, 'rb') as f:
                fig = pickle.load(f)
        except IOError:
            fig = plt.figure()
            ax = fig.add_subplot()
            r = np.array(self.radius, dtype=float)
            x = np.array(self.theta, dtype=float)
            y = np.array(self._datano, dtype=float)
            surf = ax.pcolormesh(x,
                                 y,
                                 r,
                                 cmap='jet',
                                 rasterized=True,
                                 shading='nearest'
                                 )
            fig.colorbar(surf, label='Radius (mm)')
            ax.set_title(
                f'Test No: {self._testinfo.testno} - NC4 Radius Surface'
            )
            ax.set_ylabel('Measurement Number')
            ax.set_xlabel('Angle (rad)')
            try:
                open(png_name)
            except IOError:
                fig.savefig(png_name, dpi=300)
            try:
                open(pic_name)
            except IOError:
                with open(pic_name, 'wb') as f:
                    pickle.dump(fig, f)
        return fig

    def _sampleandpos(
            self,
            fno: int
    ) -> tuple[list, list, list, list]:
        """
        Load in NC4 voltage data and select most appropriate section of \
            the signal to carry forward.

        Args:
            fno: File number to sample from.

        Returns:
            A tuple containing the signal sample and y position for both the \
                positive and negative signal.
        """
        data = self.readNC4(fno)
        filt = 50
        scale = 1
        ysteps = np.around(np.arange(0.04, -0.02, -0.01), 2)
        rpy = 4
        spr = 1
        clip = 0.5
        gap = 0.4
        ts = 1 / self._fs
        gapsamples = int(gap / ts)
        nosections = int(2 * len(ysteps))
        lentime = float(nosections) * rpy * spr + gap
        lensamples = int(lentime / ts)
        seclensamples = math.ceil(rpy * spr / ts)
        vs = math.ceil((clip * spr) / ts) - 1
        ve = int(seclensamples - ((clip * spr) / ts)) - 1

        vfilter = uniform_filter1d(data, size=filt)
        if scale == 1:
            datarange = (np.amax(vfilter), np.amin(vfilter))
            voltage = 5 * (
                (vfilter - datarange[1]) / (datarange[0] - datarange[1])
            )
        else:
            voltage = vfilter

        voltage = voltage[-(lensamples + 1):]
        vsec = np.empty(shape=(nosections, seclensamples), dtype=object)
        for sno in range(nosections):
            if sno <= (nosections - 1) / 2:
                vsec[sno][:] = voltage[
                    (sno * seclensamples):((sno + 1) * seclensamples)
                ]
            else:
                vsec[sno][:] = voltage[
                    ((sno * seclensamples) + gapsamples):
                    (((sno + 1) * seclensamples) + gapsamples)
                ]

        vsample = vsec[:, vs:ve]
        voff = np.sum((vsample - 2.5) ** 2, axis=1)

        psec = np.argmin(voff[0:math.ceil((nosections - 1) / 2)])
        psample = vsample[:][psec]
        posy = ysteps[psec]

        nsec = np.argmin(voff[math.ceil((nosections - 1) / 2):])
        nsample = vsample[:][nsec + math.ceil((nosections - 1) / 2)]
        negy = ysteps[nsec]
        # print(f'Completed File {fno}...')
        return psample, posy, nsample, negy

    def polyvalradius(self, x: tuple[np.ndarray, float]) -> list[float]:
        """
        Convert NC4 voltage signal to radius, using NC4 calibration constants.

        Args:
            x: Tuple containing the signal sample and its y position.

        Returns:
            List of converting values to radius.
        """
        pval = [-0.000341717477186167,
                0.00459433449011791,
                -0.0237307202784755,
                0.0585315537400639,
                -0.0766338436136931,
                5.15045955887124
                ]

        d = self._dcb.diameter
        rad = np.polyval(pval, x[0]) - 5.1 + (d / 2) + x[1]
        return rad

    def sigtorad(self, p: Any, n: Any) -> tuple[list[float], list[float]]:
        """
        Multiprocessing function to convert signals to radius.

        Args:
            p: Zip containing postive NC4 signal and the y position.
            n: Zip containing negative NC4 signal and the y position.

        Returns:
            Tuple containing a list of the converted postive signal and \
                converted negative signal.
        """
        # Converting to Radii rather then Voltage
        with multiprocessing.Pool() as pool:
            prad = list(tqdm(pool.imap(self.polyvalradius, p, chunksize=10),
                             total=len(self._files),
                             desc='NC4- Conv pos'))
            nrad = list(tqdm(pool.imap(self.polyvalradius, n, chunksize=10),
                             total=len(self._files),
                             desc='NC4- Conv neg'))
        pool.close()
        return prad, nrad

    @staticmethod
    def _alignposneg(prad: Union[list, np.ndarray],
                     nrad: Union[list, np.ndarray]
                     ) -> np.ndarray:
        """
        Combine the pos and neg halfs of the signal together.

        Args:
            prad: Array of radius values for positive half of signal.
            nrad: Array of radius values for negative half of signal.

        Returns:
            Array of combined radius signal
        """
        pradzero = np.subtract(np.transpose(prad), np.mean(prad, axis=1))
        nradzero = np.subtract(np.transpose(nrad), np.mean(nrad, axis=1))
        # print('Working out Lags')
        radzeros = list(zip(np.transpose(pradzero), np.transpose(nradzero)))
        if len(radzeros) == 1:
            lag = [compute_shift(radzeros[0])]
        else:
            with multiprocessing.Pool() as pool:
                lag = list(tqdm(pool.imap(
                    compute_shift,
                    radzeros,
                    chunksize=10),
                    total=len(radzeros),
                    desc='NC4- Merge pn'))
            pool.close()
        # print('Finished lags')
        nrad = np.array([np.roll(row, -x) for row, x in zip(nrad, lag)])
        radii = np.array([(p + n) / 2 for p, n in zip(prad, nrad)])
        # print('Calculated radii')
        return radii

    def _alignsigs(self, radii: np.ndarray) -> np.ndarray:
        """
        Shift radius signals so that they align with each other.

        Args:
            radii: Array containing radius vector for each measurement.

        Returns:
            Aligned radius array.
        """
        radzero = radii - radii.mean(axis=1, keepdims=True)
        radzeros = zip(radzero, np.roll(radzero, -1, axis=0))
        with multiprocessing.Pool() as pool:
            lags = list(tqdm(pool.imap(compute_shift, radzeros, chunksize=10),
                             total=len(self._files),
                             desc='NC4- Aln sigs'))
        pool.close()

        dly = np.cumsum(lags)
        dly = np.roll(dly, 1)
        dly[0] = 0
        radii = np.array([np.roll(row, -x) for row, x in zip(radii, dly)])
        self.theta = 2 * np.pi * np.arange(0, 1, 1 / self._fs)
        st = np.argmin(radii[0, 0:int(self._fs)])
        rpy = 4
        clip = 0.5
        radius = radii[
            :,
            np.arange(
                st, st + (radii.shape[1]) / (rpy - (2 * clip)), dtype=int
            )
        ]
        # self.radius = radius
        return radius

    def _fitcircles(self,
                    radius: np.ndarray
                    ) -> tuple[ndarray, ndarray, ndarray, ndarray]:
        """
        Fit a circle to each NC4 measurements and return its properties.

        Args:
            radius: Array of radius to calc attributes for

        Returns:
            List of tuples containing the x and y coords, the radius of the \
                circle and the variance.
        """
        theta = self.theta
        x = np.array([np.multiply(r, np.sin(theta)) for r in radius])
        y = np.array([np.multiply(r, np.cos(theta)) for r in radius])
        xy = np.array(list(zip(x, y))).transpose([0, 2, 1])
        if len(x) == 1:
            circle = [circle_fit.hyper_fit(xy[0])]
        else:
            with multiprocessing.Pool() as pool:
                circle = list(tqdm(pool.imap(
                    circle_fit.hyper_fit,
                    xy,
                    chunksize=10),
                    total=len(xy),
                    desc='NC4- Calc att'))
            pool.close()
        runout = np.array(
            [2 * (np.sqrt(x[0] ** 2 + x[1] ** 2)) for x in circle]
        )
        mean_radius = np.array(
            [x[2] for x in circle]
        )
        peak_radius = np.array(
            [np.max(rad) for rad in radius]
        )
        form_error = np.array(
            [(np.max(rad) - np.min(rad)) for rad in radius]
        )
        return mean_radius, peak_radius, runout, form_error

    def update(self, files: Union[list[str], tuple[str]]) -> None:
        """
        Update the file location list stored in this object.

        Args:
            files: List of strings containing the paths to each NC4 file.

        """
        self._files = files
        self._datano = np.arange(0, len(self._files))
