#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@File    :   Experiment.py
@Author  :   Tom Jessel
@Contact :   jesselt@cardiff.ac.uk

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
22/08/2022 13:46   tomhj      1.0         File containing code to produce and operate experiment objects
"""

import datetime
import fnmatch
import glob
import os
import sys
import tkinter.filedialog as tkfiledialog
from tkinter.filedialog import askdirectory
import pickle
import numpy as np
import mplcursors
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import TestInfo
import AE
import NC4


def _move_files(src: str, dst: str, ext: str = '*'):
    for f in fnmatch.filter(os.listdir(src), ext):
        os.rename(os.path.join(src, f), os.path.join(dst, f))


def _sort_rename(files, path):
    def substring_after(s, delim):
        return s.partition(delim)[2]

    t = []
    for name in files:
        temp = datetime.datetime.fromtimestamp(os.path.getmtime(name))
        t.append(temp)
    sort_ind = np.argsort(np.argsort(t))

    zipfiles = zip(sort_ind, files, t)
    sdfiles = sorted(zipfiles)

    for fno in sdfiles:
        number = str('%03.f' % fno[0])
        newfilename = 'File_' + number + '_202' + substring_after(fno[1], '202')
        if not newfilename == fno[1]:
            os.rename(fno[1], os.path.join(path, newfilename))


def load(file: str = None, process=False):
    """
    Load in a saved exp pickle file, option to process data

    :param file: str (default: None) choose file to load
    :param process: bool (default:False) option to process data when loading
    :rtype: object
    """

    f_locs = {
        'test5': r'F:\OneDrive - Cardiff University\Documents\PHD\AE\Testing\22_08_03_grit1000\Test 5.pickle',
        'test2': r"F:\OneDrive - Cardiff University\Documents\PHD\AE\Testing\TEST2Combined\Test 2.pickle",
        'test1': r"F:\OneDrive - Cardiff University\Documents\PHD\AE\Testing\28_2_22_grit1000\Test 1.pickle"
    }

    if file is None:
        try:
            file_path = tkfiledialog.askopenfilename(defaultextension='pickle')
            if not file_path:
                raise NotADirectoryError
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        except NotADirectoryError:
            print('No file selected.')
            quit()
    else:
        try:
            file_path = f_locs[file.lower().replace(' ', '')]
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        except KeyError:
            print(f'File location of {file} not saved')
            quit()
    if process:
        try:
            getattr(data.nc4, 'radius')
        except AttributeError:
            data.nc4.process()

        if not data.ae.kurt.all():
            data.ae.process()
    return data


def create_obj(process=False):
    """
    Creates object for individual test

    :param process: bool (default:False) decides if results should be processed
    :return: experiment obj
    """

    def folder_exist(path):
        # if folder doesn't exist create it
        if not os.path.isdir(path) or not os.path.exists(path):
            os.makedirs(path)

    def getdate(AE_f, NC4_f):
        if AE_f:
            d = datetime.date.fromtimestamp(os.path.getmtime(AE_f[0]))
        else:
            d = datetime.date.fromtimestamp(os.path.getmtime(NC4_f[0]))
        return d

    # import file names and directories of AE and NC4
    try:
        folder_path = askdirectory(title='Select test folder:')
        if not folder_path:
            raise NotADirectoryError
    except NotADirectoryError:
        print('No Folder selected!')
        sys.exit()

    # setup file paths and create folder if needed
    folder_path = os.path.normpath(folder_path)
    folder_path = os.path.relpath(folder_path)
    ae_path = os.path.join(folder_path, 'AE', 'TDMS')
    nc4_path = os.path.join(folder_path, 'NC4', 'TDMS')
    folder_exist(ae_path)
    folder_exist(nc4_path)

    if glob.glob(os.path.join(folder_path, "*MHz.tdms")):
        print("Moving AE files...")
        _move_files(folder_path, ae_path, '*MHz.tdms')
    ae_files = glob.glob(os.path.join(ae_path, "*.tdms"))
    _sort_rename(ae_files, ae_path)

    if glob.glob(os.path.join(folder_path, "*kHz.tdms")):
        print("Moving NC4 files...")
        _move_files(folder_path, nc4_path, '*kHz.tdms')
    nc4_files = glob.glob(os.path.join(nc4_path, "*.tdms"))
    _sort_rename(nc4_files, nc4_path)
    # collect data for exp obj
    ae_files = tuple(glob.glob(os.path.join(ae_path, "*.tdms")))
    nc4_files = tuple(glob.glob(os.path.join(nc4_path, "*.tdms")))
    date = getdate(ae_files, nc4_files)
    obj = Experiment(folder_path, date, ae_files, nc4_files)
    if process:
        obj.nc4.process()
        obj.ae.process()
    return obj


class Experiment:
    def __init__(self, dataloc, date, ae_files, nc4_files):
        self.test_info = TestInfo.TestInfo(dataloc)
        self.date = date
        self.dataloc = dataloc
        self.ae = AE.AE(ae_files, self.test_info.pre_amp, self.test_info.acquisition[0], self.test_info)
        self.nc4 = NC4.nc4(nc4_files, self.test_info, self.test_info.dcb, self.test_info.acquisition[1])
        self.features = pd.DataFrame

    def __repr__(self):
        rep = f'Test No: {self.test_info.testno} \nDate: {self.date} \nData: {self.dataloc}'
        return rep

    def save(self):
        """
        Save obj to the data location folder as a '.pickle' file
        """
        with open(f'{self.dataloc}/Test {self.test_info.testno}.pickle', 'wb') as f:
            pickle.dump(self, f)

    def update(self):
        dataloc: str = self.test_info.dataloc
        print(f'Updating experiemnt obj - {datetime.datetime.now()}')
        ae_path = os.path.join(dataloc, 'AE', 'TDMS')
        nc4_path = os.path.join(dataloc, 'NC4', 'TDMS')
        if glob.glob(os.path.join(dataloc, "*MHz.tdms")):
            _move_files(dataloc, ae_path, '*MHz.tdms')
            print('Moving new AE files...')
            ae_files = glob.glob(os.path.join(ae_path, "*.tdms"))
            _sort_rename(ae_files, ae_path)
            self.ae.update(tuple(ae_files))
        else:
            print('No new AE files')

        if glob.glob(os.path.join(dataloc, "*kHz.tdms")):
            print("Moving new NC4 files...")
            _move_files(dataloc, nc4_path, '*kHz.tdms')
            nc4_files = glob.glob(os.path.join(nc4_path, "*.tdms"))
            _sort_rename(nc4_files, nc4_path)
            self.nc4.update(tuple(nc4_files))
        else:
            print('No new NC4 files.')

    # todo finish update func and print test update

    def correlation(self, plotfig=True):
        """
        Corerlation between each freq bin and NC4 measurements

        Produces correlation coeffs for each freq bin compared to mean radius

        :param self: obj that is being used
        :param plotfig: boolean for plotting the figure
        :return: correlation coeffs
        """

        def plot(fre, c):
            figure, axs = plt.subplots(2, 1, sharex='all')
            axs[0].plot(fre, c[:, 0], 'C0', label='Mean Radius', linewidth=1)
            axs[0].plot(fre, c[:, 1], 'C1', label='Peak Radius', linewidth=1)
            axs[1].plot(fre, c[:, 2], 'C2', label='Runout', linewidth=1)
            axs[1].plot(fre, c[:, 3], 'C3', label='Form Error', linewidth=1)
            axs[0].set_xlim(0, 1000)
            axs[1].set_xlim(0, 1000)
            axs[0].xaxis.set_major_locator(mpl.ticker.MultipleLocator(100))
            axs[1].xaxis.set_major_locator(mpl.ticker.MultipleLocator(100))
            axs[0].set_ylim(-1, 1)
            axs[1].set_ylim(-1, 1)
            axs[0].set_xlabel('Freq Bin (kHz)')
            axs[1].set_xlabel('Freq Bin (kHz)')
            axs[0].set_ylabel('Pearson Correlation Coeff')
            axs[1].set_ylabel('Pearson Correlation Coeff')
            axs[0].legend(['Mean Radius', 'Peak Radius'])
            axs[1].legend(['Runout', 'Form Error'])
            axs[0].grid()
            axs[1].grid()
            axs[0].set_title('Correlation of AE FFT bins and NC4 measurements')
            return figure

        mpl.use('Qt5Agg')
        path = f'{self.test_info.dataloc}/Figures'
        png_name = f'{path}/Test {self.test_info.testno} - FFT_NC4 Correlation.png'
        pic_name = f'{path}/Test {self.test_info.testno} - FFT_NC4 Correlation.pickle'
        if not os.path.isdir(path) or not os.path.exists(path):
            os.makedirs(path)
        r = np.stack([self.nc4.mean_radius, self.nc4.peak_radius, self.nc4.runout, self.nc4.form_error])
        f = np.array(self.ae.fft[1000])
        f = f.transpose()
        coeff = np.corrcoef(f, r[-np.shape(f)[1]:])[:-4, -4:]
        if plotfig:
            try:
                with open(pic_name, 'rb') as f:
                    fig = pickle.load(f)
            except IOError:
                freq = np.arange(0, self.test_info.acquisition[0] / 2, 1000, dtype=int) / 1000
                fig = plot(freq, coeff)
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
            fig.show()
        return coeff

    # todo same thing but for AE features other than FFT
    def create_feat_df(self: object):
        """
        Creates dataframe of useful calc features from obj.

        :param self: object: experiment obj containing data for dataframe
        :return: df: DataFrame: features of experiment
        """
        cols = ["RMS", 'Kurtosis', 'Amplitude', 'Freq 10 kHz', 'Freq 35 kHz', 'Freq 134 kHz',
                'Mean radius', 'Runout', 'Form error']

        rms: np.array = self.ae.rms[:-1]
        kurt: np.array  = self.ae.kurt[:-1]
        amp: np.array  = self.ae.amplitude[:-1]

        f = np.array(self.ae.fft[1000])
        f = f.T
        f_35: np.array  = f[35][:-1]
        f_10: np.array  = f[10][:-1]
        f_134: np.array  = f[134][:-1]

        mean_rad: np.array  = self.nc4.mean_radius[1:]
        run_out: np.array  = self.nc4.runout[1:]
        form_err: np.array  = self.nc4.form_error[1:]

        m: np.array  = np.stack((rms, kurt, amp, f_10, f_35, f_134, mean_rad, run_out, form_err), axis=0)
        df: pd.DataFrame = pd.DataFrame(m.T, columns=cols)
        print(f'Feature DF of Test {self.test_info.testno}:')
        print(df.head())
        self.features = df
        return df
