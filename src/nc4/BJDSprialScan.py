import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
from nptdms import TdmsFile
from tqdm import tqdm
from datetime import datetime

from src import config

HOME_DIR, BASE_DIR, CODE_DIR, TB_DIR, RMS_DATA_DIR = config.config_paths()


def _smooth(sig, win=11):
    """
    Smooth signal using a moving average filter.

    Replicates MATLAB's smooth function. (http://tinyurl.com/374kd3ny)

    Args:
        sig (np.array): Signal to smooth.
        win (int, optional): Window size. Defaults to 11.

    Returns:
        np.array: Smoothed signal.
    """
    out = np.convolve(sig, np.ones(win, dtype=int), 'valid') / win
    r = np.arange(1, win - 1, 2)
    start = np.cumsum(sig[:win - 1])[::2] / r
    stop = (np.cumsum(sig[:-win:-1])[::2] / r)[::-1]
    return np.concatenate((start, out, stop))


class NC4SpiralScan:
    def __init__(self,
                 scanPath,
                 sCurvePath,
                 fs=50_000,
                 spindleSpeed=60,
                 zSpiralFeedrate=4,
                 toolNomDia=1.3,
                 yOffset=0.,
                 scFeedrate=60,
                 ):

        # Initialize class variables
        self._scanPath = scanPath
        self._sCurvePath = sCurvePath
        self._fs = fs
        self._spindleSpeed = spindleSpeed
        self._zSpiralFeedrate = zSpiralFeedrate
        self._toolNomDia = toolNomDia
        self._yOffset = yOffset
        self._scFeedrate = scFeedrate

        self.__calBounds = None
        self._scanMat = None

        # Setup for conversion from signal to radius
        self._sig2rad = self.SCurveCalibration(self._sCurvePath)

    @property
    def scanMat(self):
        if self._scanMat is None:
            self._scanMat = self.processSpiralScan(self._scanPath)
        return self._scanMat

    @staticmethod
    def readTDMS(path):
        test = TdmsFile.read(path)
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

    def _getSCurve(self, path):
        sCurve = self.readTDMS(path)
        time = np.arange(0, len(sCurve) / self._fs, 1 / self._fs)
        return time, sCurve
    
    def plotSCurve(self, path):
        time, sCurve = self._getSCurve(path)
        sCurve = _smooth(sCurve, win=51)

        fig, ax = plt.subplots()
        ax.plot(time, sCurve, 'C0', label='NC4')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('NC4 Voltage (V)')
        ax.set_title(f'S-Curve Calibration - {path.stem}')
        return fig, ax

    def SCurveCalibration(self, path):
        time, sCurve = self._getSCurve(path)

        scHi = 0.95 * (sCurve.max() - sCurve.min()) + sCurve.min()
        scLo = 0.05 * (sCurve.max() - sCurve.min()) + sCurve.min()
        self.__calBounds = {'scLo': scLo, 'scHi': scHi}

        s1 = np.argmin(np.abs(sCurve - scHi))
        s2 = np.argmin(np.abs(sCurve - scLo))

        distanceMoved = (time[s2] - time[s1]) * self._scFeedrate / 60
        yaxis = np.linspace(0, distanceMoved, (s2 - s1))
        yaxis = yaxis - yaxis.mean()
        yaxis = -yaxis * np.sign(yaxis[0])

        sCurve = _smooth(sCurve, win=51)

        sig2rad = interp1d(sCurve[s1:s2], yaxis, fill_value='extrapolate')
        return sig2rad

    def _scanToMat(self, data):
        samplesPerRev = int(self._fs * (self._spindleSpeed / 60))
        n_pad = int(samplesPerRev - np.mod(len(data), samplesPerRev))
        dataMat = np.pad(data,
                         (0, n_pad),
                         mode='constant',
                         constant_values=np.NaN,
                         )
        dataMat = dataMat.reshape(-1, samplesPerRev)
        return dataMat

    def processSpiralScan(self, path):
        scanData = self.readTDMS(path)
        scanData = _smooth(scanData, win=51)

        # trim data to within cal bounds
        st_ix = np.argmax(scanData > self.__calBounds['scHi'])
        scanData = scanData[st_ix:]

        scanRad = self._sig2rad(scanData)
        # account for positioning offset
        scanRad = scanRad + (self._toolNomDia / 2) + self._yOffset

        scanMat = self._scanToMat(scanRad)
        return scanMat

    def plotSpiralScan(self, path=None, Zsection=1.0, saveFig=False):
        if path is None:
            path = self._scanPath

        scanMat = self.scanMat

        z = np.arange(0, len(scanMat)) * self._zSpiralFeedrate / 60
        theta = np.linspace(0, 360, scanMat.shape[1])

        # Only look show 1mm upwards to get rid of coolant droplet
        z_ix = np.argmin(np.abs(z - Zsection))
        scanMat = scanMat[z_ix:, :]
        z = z[z_ix:]

        fig, ax = plt.subplots(1, 2,
                               width_ratios=[0.5, 1],
                               sharey=True,
                               layout='constrained',
                               figsize=(10, 6),
                               )
        fig.suptitle(f'{path.stem}')
        im = ax[1].imshow(scanMat[:, :],
                          aspect='auto',
                          cmap='jet',
                          origin='lower',
                          interpolation='none',
                          extent=[theta[0], theta[-1],
                                  z[0], z[-1]
                                  ],
                          vmin=0.61,
                          vmax=0.68,
                          )
        plt.colorbar(im, ax=ax[1], label='Radius (mm)')
        ax[1].set_xlabel('Angle (deg)')

        ax[0].plot(np.median(scanMat, axis=1), z, 'C0', label='Median')
        # ax[0].plot(scanMat.mean(axis=1), z, 'C0', label='Mean')
        ax[0].plot(scanMat.min(axis=1), z, 'r', label='Min')
        ax[0].plot(scanMat.max(axis=1), z, 'g', label='Max')
        ax[0].set_xlabel('Radius (mm)')
        ax[0].set_ylabel('Z (mm)')
        ax[0].autoscale(enable=True, axis='y')
        ax[0].grid()
        ax[0].legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.1))
        if saveFig:
            fig.savefig(f'{path.parent}/{path.stem}.png',
                        dpi=100,
                        bbox_inches='tight',
                        )
        return fig, ax


def renameSpiralScans(exp, spiralScanDir, spiralScanFiles):
    # find closest NC4 file for each spiral scan, via timestamps
    timeStamp_Sprial = []
    timeStamp_NC4 = []
    for f in spiralScanFiles:
        timeStamp_Sprial.append(datetime.strptime(f.name[:19],
                                                  "%Y_%m_%d_%H_%M_%S",
                                                  ))
    for f in exp.nc4._files:
        timeStamp_NC4.append(datetime.strptime(Path(f).name[9:28],
                                               "%Y_%m_%d_%H_%M_%S",
                                               ))

    for i, (ts_spiral, f) in enumerate(zip(timeStamp_Sprial, spiralScanFiles)):
        # find the nearest NC4 file
        idx = np.argmin([abs(ts_spiral - t0) for t0 in timeStamp_NC4])
        fileName = f'Cut_{Path(exp.nc4._files[idx]).name[5:8]}_{f[i].name}'
        spiralScanFiles[i].rename(spiralScanDir.joinpath(fileName))


def processExpSprialScans(exp,
                          SCPath,
                          nomDia=1.3,
                          feedrate=2,
                          rpm=60,
                          fs=50_000,
                          yOffset=0.03,
                          calFeedrate=60,
                          ):
    spiralScanDir = Path(exp.dataloc).joinpath('Spiral Scans')
    assert spiralScanDir.exists(), "Spiral Scan directory not found."
    spiralScanFiles = list(spiralScanDir.glob("*.tdms"))
    
    # if files not already renamed
    if not all([f.name.startswith('Cut') for f in spiralScanFiles]):
        renameSpiralScans(exp, spiralScanDir, spiralScanFiles)
    
    # process the renamed files
    nc = []
    for f in tqdm(spiralScanFiles):
        sc = NC4SpiralScan(scanPath=f,
                           sCurvePath=SCPath,
                           toolNomDia=nomDia,
                           zSpiralFeedrate=feedrate,
                           spindleSpeed=rpm,
                           fs=fs,
                           yOffset=yOffset,
                           scFeedrate=calFeedrate,
                           )
        sc.scanMat
        nc.append(sc)
    return nc


if __name__ == "__main__":

    NOM_DIA = 1.3
    FEEDRATE = 2
    RPM = 60
    FS = 50_000
    YOFFSET = 0.03
    CALFEEDRATE = 60

    SCPath = CODE_DIR.joinpath(
        r'src/reference/NC4_BJD_SCurve_2024_05_30_14_17_12-Ch0-50kHz.tdms'
    )
    assert SCPath.exists(), "SCurve calibration file not found."

    # AE testing Folder
    TEST_FOLDER = BASE_DIR.joinpath(r'AE/Testing')

    assert TEST_FOLDER.exists(), "Testing folder not found."

    dataFiles = list(
        TEST_FOLDER.joinpath(
            r'24_07_03_weartest_D1.3_#1000/Spiral Scans'
        ).glob('*.tdms')
    )

    for f in tqdm(dataFiles[:]):
        nc = NC4SpiralScan(
            scanPath=f,
            sCurvePath=SCPath,
            fs=FS,
            spindleSpeed=RPM,
            zSpiralFeedrate=FEEDRATE,
            toolNomDia=NOM_DIA,
            yOffset=YOFFSET,
        )
        fig, ax = nc.plotSpiralScan(saveFig=True,
                                    Zsection=1.0,
                                    )
        plt.close(fig)
    # plt.show()
