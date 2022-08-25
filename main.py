#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import mplcursors

import NC4
import datetime
import fnmatch
import glob
import os
import shutil
import tkinter.filedialog as tkfiledialog
from tkinter.filedialog import askdirectory
import numpy as np
import Experiment
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import pandas as pd

mpl.use("Qt5Agg")
# from importlib import reload
# import AE


# function that moves _files from src folder to dst folder with optional ext selector
def move_files(src: str, dst: str, ext: str = '*'):
    for f in fnmatch.filter(os.listdir(src), ext):
        shutil.move(os.path.join(src, f), os.path.join(dst, f))


# function to extract substring after pattern
def substring_after(s, delim):
    return s.partition(delim)[2]


# function that creates a folder if one does not already exist based on path
def folder_exist(path):
    if not os.path.isdir(path) or not os.path.exists(path):
        os.makedirs(path)


# function that sorts _files based off date modified and then renames based on position
def sort_rename(files, path):
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


# function that gets date of test from first AE or NC4 file
def getdate(AE_f, NC4_f):
    if AE_f:
        d = datetime.date.fromtimestamp(os.path.getmtime(AE_f[0]))
    else:
        d = datetime.date.fromtimestamp(os.path.getmtime(NC4_f[0]))
    return d


def createobj():
    # import file names and directories of AE and NC4
    folder_path = None
    try:
        folder_path = askdirectory(title='Select test folder:')
        if not folder_path:
            raise NotADirectoryError
    except NotADirectoryError:
        print('No Folder selected!')
        sys.exit()

    folder_path = os.path.normpath(folder_path)
    
    # create two folder paths for AE and NC4
    ae_path = os.path.join(folder_path, 'AE', 'TDMS')

    nc4_path = os.path.join(folder_path, 'NC4', 'TDMS')

    folder_exist(ae_path)
    folder_exist(nc4_path)

    if glob.glob(os.path.join(folder_path, "*MHz.tdms")):
        print("Moving AE _files...")
        move_files(folder_path, ae_path, '*MHz.tdms')
        ae_files = glob.glob(os.path.join(ae_path, "*.tdms"))
        sort_rename(ae_files, ae_path)

    if glob.glob(os.path.join(folder_path, "*kHz.tdms")):
        print("Moving NC4 _files...")
        move_files(folder_path, nc4_path, '*kHz.tdms')
        nc4_files = glob.glob(os.path.join(nc4_path, "*.tdms"))
        sort_rename(nc4_files, nc4_path)

    ae_files = tuple(glob.glob(os.path.join(ae_path, "*.tdms")))
    nc4_files = tuple(glob.glob(os.path.join(nc4_path, "*.tdms")))

    date = getdate(ae_files, nc4_files)

    obj = Experiment.Experiment(folder_path, date, ae_files, nc4_files)

    return obj


def load():
    try:
        file_path = tkfiledialog.askopenfilename(defaultextension='pickle')
        if not file_path:
            raise NotADirectoryError
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except NotADirectoryError:
        print('No file selected.')
        quit()
    return data


if __name__ == '__main__':
    exp = load()
    try:
        getattr(exp.nc4, 'radius')
    except AttributeError:
        exp.nc4.process()

    if not exp.ae.kurt:  # todo broken if filled in
        exp.ae.process()
    exp.save()

# todo add methods to update objects and also print progress of tests
