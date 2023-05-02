"""
@File    :   surf_meas.py
@Author  :   Tom Jessel
@Contact :   jesselt@cardiff.ac.uk

@Creation Time    @Author    @Version    @Description
------------      -------    --------    -----------
31/01/2023 13:19   tomhj      1.0         N/A
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from typing import Union, Any
import multiprocessing as mp
# import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import skew
from tqdm.auto import tqdm

import resources


class SurfMeasurements:
    def __init__(self,
                 rad_mat: np.ndarray,
                 ls: int = 45,
                 lc: int = 1250,
                 ):
        """
        Class for all things surface measurements

        Args:
            rad_mat: Matrix of radius measurements from NC4 for whole test.
            ls: No of samples to filer over to remove micro-roughness
            lc: No of samples to filter over to remove roughness
        """
        self.rad = rad_mat.astype('float')
        self.ls = ls
        self.lc = lc

        self.p, self.w, self.r, self.meas_df = self._process_dataset()

    @staticmethod
    def _filt_rough(surf: Any, ls: int, lc: int):
        """
        Filer surface profile to extract raw, waviness and roughness profiles.

        Args:
            surf: Surface measurement
            ls: No of samples to filer over to remove micro-roughness
            lc: No of samples to filter over to remove roughness

        Returns:
            Extracted profiles of raw, waviness and roughness
        """
        p_f = gaussian_filter1d(surf, sigma=ls)
        w_f = gaussian_filter1d(p_f, sigma=lc)
        r_f = p_f - w_f
        return p_f, w_f, r_f

    @staticmethod
    def _sur_meas(p_f, w_f, r_f):
        """
        Extract surface measurement values from profiles.

        Args:
            p_f: Raw profile
            w_f: Waviness profile
            r_f: Roughness profile

        Returns:
            Dict with a, q and sk values for each profile.
        """
        p_dev = p_f - np.mean(p_f)
        w_dev = w_f - np.mean(w_f)
        r_dev = r_f - np.mean(r_f)

        p_a = (np.sum(np.abs(p_dev))) / len(p_f)
        w_a = (np.sum(np.abs(w_dev))) / len(w_f)
        r_a = (np.sum(np.abs(r_dev))) / len(r_f)

        p_q = np.sqrt(np.sum(p_dev**2)) / len(p_f)
        w_q = np.sqrt(np.sum(w_dev**2)) / len(w_f)
        r_q = np.sqrt(np.sum(r_dev**2)) / len(r_f)

        p_sk = skew(p_dev)
        w_sk = skew(w_dev)
        r_sk = skew(r_dev)

        results = {'p': [p_a, p_q, p_sk],
                   'w': [w_a, w_q, w_sk],
                   'r': [r_a, r_q, r_sk],
                   }
        return results

    def _mp_process(self, r_i: int):
        """
        Process profiling for one radius measurement row.

        Args:
            r_i: Row index of rad_mat to process

        Returns:
            Extracted profiles and values of calc measurements.
        """
        surf = self.rad[r_i, :]
        pt, wt, rt, = self._filt_rough(surf, self.ls, self.lc)
        meas = self._sur_meas(pt, wt, rt)
        meas = list(meas.values())
        m = [item for sublist in meas for item in sublist]
        return pt, wt, rt, m

    def _process_dataset(self):
        """
        Process all measurements with multiprocessing.

        Returns:
            Matrix of each profile and df of calc values.
        """
        no_files = np.shape(self.rad)[0]
        with mp.Pool() as pool:
            out = list(tqdm(pool.imap(self._mp_process, range(no_files)),
                            total=no_files,
                            desc='Profiling'))

        p_f, w_f, r_f, res = map(list, zip(*out))

        p_f = np.array(p_f)
        w_f = np.array(w_f)
        r_f = np.array(r_f)

        df = pd.DataFrame(res,
                          columns=['Pa', 'Pq', 'Psk',
                                   'Wa', 'Wq', 'Wsk',
                                   'Ra', 'Rq', 'Rsk',
                                   ])
        return p_f, w_f, r_f, df

    def surf_2d(self, profile: Union[None, str, list[str]] = None):
        """
        Plot 2d map of the whole test.

        Args:
            profile: Choice of which profile figures to plot.

        Returns:
            2d surface figures created.
        """

        def plot(pro, data):
            fig, ax = plt.subplots()
            im = ax.imshow(data, origin='lower', interpolation='bilinear',
                           aspect='auto', cmap='jet')
            fig.colorbar(im, ax=ax, label='')
            ax.set_xlabel('')
            ax.set_ylabel('')

            def onclick(event):
                if event.dblclick:
                    if event.button == 1:
                        y = round(event.ydata)
                        values = self.meas_df.iloc[y]
                        if pro == 'p':
                            start_i = 0
                        elif pro == 'w':
                            start_i = 3
                        else:
                            start_i = 6
                        a = values.iloc[start_i]
                        q = values.iloc[(start_i + 1)]
                        sk = values.iloc[(start_i + 2)]
                        print(f'{pro.upper()} - Meas {y:03d}:'
                              f'\t{pro.upper()}a = {a:.3e}'
                              f'\t{pro.upper()}q = {q:.3e}'
                              f'\t{pro.upper()}sk = {sk:.3e}')

            fig.canvas.mpl_connect('button_press_event', onclick)
            return fig

        key = {'p': self.p, 'w': self.w, 'r': self.r}

        if profile is None:
            figs = [plot(pro, data) for pro, data in key.items()]
        elif type(profile) is str:
            figs = plot(profile, key[profile])
        else:
            figs = [plot(pr, key[pr]) for pr in profile]

        plt.show()
        return figs

    def plot_profiles(self, r_no):
        """
        Plots all 3 profiles of one specific measurement number.

        Args:
            r_no: Row number to plot profiles of.

        Returns:
            Created profile figure.
        """
        p_i = self.p[r_no, :]
        w_i = self.w[r_no, :]
        r_i = np.multiply(self.r[r_no, :], 1000)
        p_mean = np.mean(p_i)

        fig, ax = plt.subplots(2, 1, sharex='col')
        fig.suptitle(f'Profiles of NC4 measurement No {r_no}')
        ax[0].plot(p_i, color='b')
        ax[0].plot(w_i, color='g')
        ax[0].hlines(y=p_mean, xmin=0, xmax=len(p_i), color='k',
                     linestyles='dashed', alpha=0.4)
        ax[1].plot(r_i, color='r')
        ax[1].hlines(y=0, xmin=0, xmax=len(p_i), color='gray', linewidth=0.5)
        ax[1].set_xlim(0, len(p_i))
        return fig

# print(f'P_a: {p_a}')
# print(f'W_a: {w_a}')
# print(f'R_a: {r_a}')
# print('\n')
#
# print(f'P_q: {p_q}')
# print(f'W_q: {w_q}')
# print(f'R_q: {r_q}')
# print('\n')
#
# print(f'P_sk: {p_sk}')
# print(f'W_sk: {w_sk}')
# print(f'R_sk: {r_sk}')
# print('\n')


if __name__ == "__main__":

    LS = 45
    LC = 1250

    exp5 = resources.load('Test 5')
    exp7 = resources.load('Test 7')
    exp8 = resources.load('Test 8')
    exp9 = resources.load('Test 9')

    surf5 = SurfMeasurements(exp5.nc4.radius, ls=LS, lc=LC)
    surf7 = SurfMeasurements(exp7.nc4.radius, ls=LS, lc=LC)
    surf8 = SurfMeasurements(exp8.nc4.radius, ls=LS, lc=LC)
    surf9 = SurfMeasurements(exp9.nc4.radius, ls=LS, lc=LC)
