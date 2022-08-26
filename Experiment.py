# Class for test objects with methods
import os
import TestInfo
import AE
import pickle
import NC4
import numpy as np
import mplcursors
import matplotlib as mpl
import matplotlib.pyplot as plt


class Experiment:
    def __init__(self, dataloc, date, ae_files, nc4_files):
        self.test_info = TestInfo.TestInfo(dataloc)
        self.date = date
        self.dataloc = dataloc
        self.ae = AE.AE(ae_files, self.test_info.pre_amp, self.test_info.acquisition[0], self.test_info)
        self.nc4 = NC4.nc4(nc4_files, self.test_info, self.test_info.dcb, self.test_info.acquisition[1])

    def __repr__(self):
        rep = f'Test No: {self.test_info.testno} \nDate: {self.date} \nData: {self.dataloc}'
        return rep

    def save(self):
        with open(f'{self.dataloc}/Test {self.test_info.testno}.pickle', 'wb') as f:
            pickle.dump(self, f)

    def correlation(self, plotfig=True):
        """
        Corerlation between each freq bin and NC4 measurements

        Produces correlation coeffs for each freq bin compared to mean radius

        :param self: obj that is being used
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
