#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@File    :   experiment.py
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
from tkinter.filedialog import askdirectory, askopenfilename
import tkinter as tk
import pickle
from typing import Union

import numpy as np
import mplcursors
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from .ae import AE
from .nc4 import NC4


def _move_files(src: str, dst: str, ext: str = '*') -> None:
    """
    Move filtered files depending on extension from one folder to another.

    Args:
        src: Source directory path to move files from.
        dst: Destination directory path to move files to.
        ext: Filter option, to allow only certain files to be moved.

    """
    for f in fnmatch.filter(os.listdir(src), ext):
        os.rename(os.path.join(src, f), os.path.join(dst, f))


def _sort_rename(files: list[str], path: str) -> None:
    """
    Rename all files in the list based on date and time stamp on the file.

    Args:
        files: List of file paths to sort and rename.
        path:  Folder path of the file locations.

    """
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


class TestInfo:
    def __init__(self, dataloc: str) -> None:
        """
        Class to store data about the experiment from the experiment "TESTING INFO.txt" file.

        Args:
            dataloc: Location of the experiment folder, to find the "TESTING INFO.txt" file.
        """
        self.dataloc = dataloc
        infofile = os.path.join(dataloc, 'TESTING INFO.txt')
        try:
            data = pd.read_table(infofile, delimiter='\t', header=None, skiprows=(0, 1))
        except FileNotFoundError as err:
            print(f'{err} \nNo "TESTING INFO.txt" file found!!')
        else:
            self.testno = int(data.iloc[0][1])
            self.date = datetime.datetime.strptime(data.iloc[1][1], '%d %b %Y')
            self.pre_amp = PreAmp(float(data.iloc[5][1]), data.iloc[6][1], data.iloc[7][1])
            self.sensor = data.iloc[9][1]
            self.acquisition = (float(data.iloc[11][1]) * 1E6, float(data.iloc[12][1]) * 1E3)
            self.dcb = DCB(float(data.iloc[14][1]), float(data.iloc[15][1]), data.iloc[16][1])
            self.grindprop = GrindProp(float(data.iloc[18][1]), float(data.iloc[19][1]), float(data.iloc[20][1]),
                                       float(data.iloc[21][1]))


class GrindProp:
    def __init__(self, feedrate: float, doc_ax: float, doc_rad: float, v_spindle: float) -> None:
        """
        Class to store data about the grinding properties.

        Args:
            feedrate: Feedrate of the grinding during test. (mm/min)
            doc_ax: Axial depth of cut during test. (mm)
            doc_rad: Radial depth of cut during test. (mm)
            v_spindle: Spindle speed during the test. (RPM)
        """
        self.feedrate = feedrate
        self.doc_ax = doc_ax
        self.doc_rad = doc_rad
        self.v_spindle = v_spindle


class PreAmp:
    def __init__(self, gain: float, spec: str, filt: str) -> None:
        """
        Class to store data about the pre-amp during the test.

        Args:
            gain: Gain setting of the pre-amp. (dB)
            spec: Pre-amp specification/type.
            filt: Pre-amp built-in filter. (kHz)
        """
        self.gain = gain
        self.spec = spec
        self.filter = filt


class DCB:
    def __init__(self, d: float, grit: float, form: str) -> None:
        """
        Class to store data about the DCB used in the test.

        Args:
            d: Diameter of the DCB used in the test. (mm)
            grit: Mesh size of the DCB used in the test. (#)
            form: Form of the DCB used during the test.
        """
        self.grainsize = None
        self.diameter = d
        self.grit = grit
        self.form = form
        self.gritsizeset()

    def gritsizeset(self):
        grainsizes = pd.read_csv('reference/grainsizes.csv')
        self.grainsize = float(grainsizes.iloc[np.where(grainsizes['Mesh'] == self.grit)]['AvgGrainSize'])


class Experiment:
    def __init__(self, dataloc: str, date: datetime.date, ae_files: tuple[str], nc4_files: tuple[str]) -> None:
        """
        Experiment Class, including AE class, NC4 class, test_info and features.

        Args:
            dataloc: Folder path of the experiment folder.
            date: Date the test was carried out.
            ae_files: Tuple of file paths for AE TDMS files.
            nc4_files: Tuple of file paths for NC4 TDMS files.
        """
        self.test_info = TestInfo(dataloc)
        self.date = date
        self.dataloc = dataloc
        self.ae = AE(ae_files, self.test_info.pre_amp, self.test_info,  self.test_info.acquisition[0])
        self.nc4 = NC4(nc4_files, self.test_info, self.test_info.dcb, self.test_info.acquisition[1])
        self.features = pd.DataFrame

    def __repr__(self):
        no_nc4 = len(self.nc4._files)
        no_ae = len(self.ae._files)
        rep = f'Test No: {self.test_info.testno}\nDate: {self.date}\nData: {self.dataloc}\n' \
              f'No. Files: AE-{no_ae} NC4-{no_nc4}'
        return rep

    def save(self) -> None:
        """
        Save Experiment obj to the data location folder as a '.pickle' file.
        """
        with open(f'{self.dataloc}/Test {self.test_info.testno}.pickle', 'wb') as f:
            pickle.dump(self, f)

    def update(self) -> None:
        """
        Update the Experiment class with new files.
        """
        dataloc = self.test_info.dataloc
        if glob.glob(os.path.join(dataloc, "*.tdms")):
            print(f'Updating experiemnt obj - {datetime.datetime.now()}')
            ae_path = os.path.join(dataloc, 'AE', 'TDMS')
            nc4_path = os.path.join(dataloc, 'NC4', 'TDMS')
            if glob.glob(os.path.join(dataloc, "*MHz.tdms")):
                _move_files(dataloc, ae_path, '*MHz.tdms')
                print('Moving new AE files...')
                ae_files = glob.glob(os.path.join(ae_path, "*.tdms"))
                _sort_rename(ae_files, ae_path)
                ae_files = glob.glob(os.path.join(ae_path, "*.tdms"))
                self.ae.update(tuple(ae_files))
            # else:
            #     print('No new AE files')

            if glob.glob(os.path.join(dataloc, "*kHz.tdms")):
                print("Moving new NC4 files...")
                _move_files(dataloc, nc4_path, '*kHz.tdms')
                nc4_files = glob.glob(os.path.join(nc4_path, "*.tdms"))
                _sort_rename(nc4_files, nc4_path)
                nc4_files = glob.glob(os.path.join(nc4_path, "*.tdms"))
                self.nc4.update(tuple(nc4_files))
            # else:
            #     print('No new NC4 files.')

            no_nc4 = len(self.nc4._files)
            no_ae = len(self.ae._files)
            print(f'No. Files: AE-{no_ae} NC4-{no_nc4}')
            self.save()


    # todo finish update func and print test update

    def correlation(self, plotfig: bool = True) -> None:
        """
        Corerlation between each freq bin and NC4 measurements


        Produces correlation coeffs for each freq bin compared to mean radius
        Args:
            plotfig: Option to plot a figure.

        Returns:
            Correlation coefficients for each freq bin compared to NC4 measurements.

        """

        def plot(fre, c):
            figure, axs = plt.subplots(2, 1, sharex='all')
            axs[0].plot(fre, c[:, 0], 'C0', label='Mean Radius', linewidth=1)
            axs[0].plot(fre, c[:, 1], 'C1', label='Peak Radius', linewidth=1)
            axs[1].plot(fre, c[:, 2], 'C2', label='Runout', linewidth=1)
            axs[1].plot(fre, c[:, 3], 'C3', label='Form Error', linewidth=1)
            axs[0].axhline(0.65, color='r', linewidth=0.5)
            axs[0].axhline(-0.65, color='r', linewidth=0.5)
            axs[1].axhline(0.65, color='r', linewidth=0.5)
            axs[1].axhline(-0.65, color='r', linewidth=0.5)
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

        freq = 1000

        mpl.use('Qt5Agg')
        path = f'{self.test_info.dataloc}/Figures'
        png_name = f'{path}/Test {self.test_info.testno} - FFT_NC4 Correlation.png'
        pic_name = f'{path}/Test {self.test_info.testno} - FFT_NC4 Correlation.pickle'
        if not os.path.isdir(path) or not os.path.exists(path):
            os.makedirs(path)
        r = np.stack([self.nc4.mean_radius, self.nc4.peak_radius, self.nc4.runout, self.nc4.form_error])
        f = np.array(self.ae.fft[freq])
        f = f.transpose()
        coeff = np.corrcoef(f, r[-np.shape(f)[1]:])[:-4, -4:]
        if plotfig:
            freq = np.arange(0, self.test_info.acquisition[0] / 2, freq, dtype=int) / 1000
            fig = plot(freq, coeff)
            mplcursors.cursor(multiple=True)
            fig.show()
            try:
                open(png_name)
            except IOError:
                fig.savefig(png_name, dpi=300)
            try:
                open(pic_name)
            except IOError:
                with open(pic_name, 'wb') as f:
                    pickle.dump(fig, f)
        return coeff

    # todo same thing but for AE features other than FFT
    def create_feat_df(self) -> pd.DataFrame:
        """
        Creates dataframe of useful calc features from test for ML.

        Returns:
            DataFrame: Features of experiment
        """
        cols = ["RMS", 'Kurtosis', 'Amplitude', 'Skewness', 'Freq 10 kHz', 'Freq 35 kHz', 'Freq 134 kHz',
                'Mean radius', 'Peak radius', 'Radius diff', 'Runout', 'Form error']

        rms = self.ae.rms[:-1]
        kurt = self.ae.kurt[:-1]
        amp = self.ae.amplitude[:-1]
        skew = self.ae.skewness[:-1]

        f = np.array(self.ae.fft[1000])
        f = f.T
        f_35: np.array = f[35][:-1]
        f_10: np.array = f[10][:-1]
        f_134: np.array = f[134][:-1]

        mean_rad: np.array = self.nc4.mean_radius[1:]
        peak_rad = self.nc4.peak_radius[1:]
        rad_diff = np.diff(self.nc4.mean_radius)
        run_out: np.array = self.nc4.runout[1:]
        form_err: np.array = self.nc4.form_error[1:]

        m: np.array = np.stack((rms, kurt, amp, skew, f_10, f_35, f_134, mean_rad, peak_rad, rad_diff, run_out, form_err), axis=0)
        df: pd.DataFrame = pd.DataFrame(m.T, columns=cols)
        print(f'Feature DF of Test {self.test_info.testno}:')
        print(df.head())
        self.features = df
        return df


def load(file: str = None, process: bool = False) -> Union[Experiment, None]:
    """
    Load in a saved exp pickle file, option to process data.

    Args:
        file: Name of test to load.
        process: Option to process data in experiment object

    Returns:
        Saved experiment obj containing AE, NC4 data.
    """

    f_locs = {
        'test5': r'..\..\Testing\22_08_03_grit1000\Test 5.pickle',
        'test2': r"..\..\Testing\TEST2Combined\Test 2.pickle",
        'test1': r"..\..\Testing\28_2_22_grit1000\Test 1.pickle"
    }

    if file is None:
        try:
            root = tk.Tk()
            file_path = askopenfilename(defaultextension='pickle')
            root.withdraw()
            if not file_path:
                raise NotADirectoryError
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        except NotADirectoryError:
            print('No existing exp file selected!')
            raise NotADirectoryError('No existing exp file selected to load!')
    else:
        try:
            file_path = f_locs[file.lower().replace(' ', '')]
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        except KeyError:
            # print(f'File location of {file} not saved in load function.')
            # print(f'Known file locations are : {f_locs}')
            raise NotADirectoryError(f'File location of {file} not saved in load function. \
Known file locations are : {f_locs.keys()}')
    if process:
        try:
            getattr(data.NC4, 'radius')
        except AttributeError:
            data.NC4.process()

        if not data.ae.kurt.all():
            data.ae.process()
    return data


def create_obj(folder: str = None, process: bool = False) -> Union[Experiment, None]:
    """
    Creates experiment obj for test from test folder, selected either by GUI or path input.

    Args:
        folder: Test file path to create obj from, if blank will open GUI to select folder.
        process: Option to process AE and NC4 data on creation of obj.

    Returns:
        Experiment obj containing storage for AE & NC4 data, and relevant features.
    """

    def folder_exist_create(path: str) -> None:
        """
        Check if a folder exists, if not create it.

        Args:
            path: Folder path to check and create.

        """
        # if folder doesn't exist create it
        if not os.path.isdir(path) or not os.path.exists(path):
            os.makedirs(path)

    def getdate(AE_f: Union[list[str], tuple[str]], NC4_f: Union[list[str], tuple[str]]) -> datetime.date:
        """
        Find the date of the first AE file, or first NC4 file if no AE data.

        Args:
            AE_f: List of AE files in the test folder.
            NC4_f: List of NC4 files in the test folder.

        Returns:
            A date class of the first AE file.

        """
        if AE_f:
            d = datetime.date.fromtimestamp(os.path.getmtime(AE_f[0]))
        else:
            d = datetime.date.fromtimestamp(os.path.getmtime(NC4_f[0]))
        return d

    # import file names and directories of AE and NC4
    if folder is None:
        try:
            root = tk.Tk()
            folder_path = askdirectory(title='Select test folder:')
            root.withdraw()
            if not folder_path:
                raise NotADirectoryError
        except NotADirectoryError:
            print('No Folder selected!')
            return
    else:
        try:
            if os.path.isdir(folder):
                folder_path = folder
            else:
                raise NotADirectoryError
        except NotADirectoryError:
            print('Not a valid folder path.')
            return

    # setup file paths and create folder if needed
    folder_path = os.path.normpath(folder_path)
    folder_path = os.path.relpath(folder_path)
    ae_path = os.path.join(folder_path, 'AE', 'TDMS')
    nc4_path = os.path.join(folder_path, 'NC4', 'TDMS')
    folder_exist_create(ae_path)
    folder_exist_create(nc4_path)

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
