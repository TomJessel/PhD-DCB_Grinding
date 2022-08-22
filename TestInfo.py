# Test Info class

import os

import numpy as np
import pandas as pd
from datetime import datetime


class TestInfo:
    def __init__(self, dataloc):
        self.dataloc = dataloc
        infofile = os.path.join(dataloc, 'TESTING INFO.txt')
        try:
            data = pd.read_table(infofile, delimiter='\t', header=None, skiprows=(0, 1))
        except FileNotFoundError as err:
            print(f'{err} \nNo "TESTING INFO.txt" file found!!')
        else:
            self.testno = int(data.iloc[0][1])
            self.date = datetime.strptime(data.iloc[1][1], '%d %b %Y')
            self.pre_amp = PreAmp(float(data.iloc[5][1]), data.iloc[6][1], data.iloc[7][1])
            self.sensor = data.iloc[9][1]
            self.acquisition = (float(data.iloc[11][1]) * 1E6, float(data.iloc[12][1]) * 1E3)
            self.dcb = DCB(float(data.iloc[14][1]), float(data.iloc[15][1]), data.iloc[16][1])
            self.grindprop = GrindProp(float(data.iloc[18][1]), float(data.iloc[19][1]), float(data.iloc[20][1]),
                                       float(data.iloc[21][1]))


class GrindProp:
    def __init__(self, feedrate, doc_ax, doc_rad, v_spindle):
        self.feedrate = feedrate
        self.doc_ax = doc_ax
        self.doc_rad = doc_rad
        self.v_spindle = v_spindle


class PreAmp:
    def __init__(self, gain, spec, filter):
        self.gain = gain
        self.spec = spec
        self.filter = filter


class DCB:
    def __init__(self, d, grit, form):
        self.grainsize = None
        self.diameter = d
        self.grit = grit
        self.form = form
        self.gritsizeset()

    def gritsizeset(self):
        grainsizes = pd.read_csv('Reference/grainsizes.csv')
        self.grainsize = float(grainsizes.iloc[np.where(grainsizes['Mesh'] == self.grit)]['AvgGrainSize'])


def main():
    obj = TestInfo('F:\\OneDrive - Cardiff University\\Documents\\PHD\\AE\\Testing\\22_08_03_grit1000')
    return obj


if __name__ == '__main__':
    exp = main()
