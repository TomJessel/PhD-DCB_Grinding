from nptdms import TdmsFile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import mplcursors
import multiprocessing
from functools import partial
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
        db = [20*(np.log10(vin/v_ref)) for vin in v]
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
        matplotlib.use("Qt5Agg")
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
        matplotlib.use('Qt5Agg')
        plt.figure()
        plt.plot(f / 1000, p, linewidth=0.75)
        plt.title(f'Test No: {self._testinfo.testno} - FFT File {filename[-3:]}')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.xlabel('Frequency (kHz)')
        plt.ylabel('Amplitude (dB)')
        plt.grid()
        mplcursors.cursor(multiple=True)
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
        # print(f'Completed File {fno}...')
        return k, r

    def fftsurf(self, res=1000, freqlim=None):
        if res in self.fft:
            p = self.fft[res]
        else:
            with multiprocessing.Pool() as pool:
                results = pool.map(partial(self.fftcalc, freqres=res), range(len(self._files)))
            p = self.volt2db(np.array(results))
            self.fft[res] = p

        if freqlim is None:
            freqlim = {'lowlim': int(0/res), 'uplim': int(self._fs / (2 * res))}
        else:
            freqlim = {'lowlim': int(freqlim[0]/res), 'uplim': int(freqlim[1]/res)}
        f = np.arange(0, self._fs / 2, res, dtype=int)
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
