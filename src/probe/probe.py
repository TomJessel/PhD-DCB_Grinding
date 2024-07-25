from typing import Any, Union
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from .. import config

HOME_DIR, BASE_DIR, CODE_DIR, TB_DIR, RMS_DATA_DIR = config.config_paths()


class PROBE:
    def __init__(
            self,
            filepath: Union[str, Path],
            doc: float = 0.03,
            tol: float = 0.02,
    ):
        """
        Probe class.

        Args:
            filepath (str, Path): Path to the probe data csv file.
            doc (float): Inputted radial depth of cut for grinding (mm).
            tol (float): desired tolerance for the depth of cut (mm).
        """

        self._filepath = filepath
        assert self.filepath.exists(), f"File not found: {self.filepath}"

        self.doc = doc
        self.tol = tol

        self.probeData = self.readProbeCSV(self.filepath)

    @property
    def filepath(self):
        if isinstance(self._filepath, str):
            self._filepath = Path(self._filepath)
        return self._filepath
    
    @staticmethod
    def readProbeCSV(filepath):
        df = pd.read_csv(filepath,
                         header=2,
                         )

        columns = ['Cut No', 'REFBORE-X', 'REFBORE-Y', 'AVGPROBE']
        assert all([col in df.columns for col in columns]), \
            f"Columns not found: {columns}"

        df['PROBEDIFF'] = df['AVGPROBE'].diff()
        df['PROBEDIFF'] = df['PROBEDIFF'].fillna(0)
        df['REFBORE-X'] = (df['REFBORE-X'] - df['REFBORE-X'].iloc[0]) * 1000
        df['REFBORE-Y'] = (df['REFBORE-Y'] - df['REFBORE-Y'].iloc[0]) * 1000

        df.set_index('Cut No', inplace=True)
        df['AVGPROBE'] = df['AVGPROBE'] - df['AVGPROBE'].iloc[0]
        return df
    
    def refreshProbeData(self):
        self.probeData = self.readProbeCSV(self.filepath)
    
    def plot_probe_DOC(self, plt_ax=None):
        if plt_ax is None:
            fig, ax = plt.subplots()
        else:
            ax = plt_ax
            fig = ax.get_figure()

        y_measured = self.probeData['AVGPROBE'].values

        ax.plot(y_measured, label='Cumulative DOC')

        x = np.arange(0, len(self.probeData))
        y_ideal = x * self.doc
        ax.plot(x, y_ideal, 'C0', linestyle='--', alpha=0.5)

        y_tol = (y_ideal + self.tol, y_ideal - self.tol)
        ax.fill_between(x, y_tol[0], y_tol[1],
                        color='C0', alpha=0.4,
                        )

        ax2 = ax.twinx()
        ax2.plot(y_ideal - y_measured,
                 color='C1', label='Cumulative Error',
                 alpha=0.75,
                 )

        ax.set_ylim(0,)
        ax2.set_ylim(0,)
        ax.set_xlim(0,)

        ax.set_ylabel('Radial DOC (mm)')
        ax2.set_ylabel('Error (mm)')
        ax.set_xlabel('Cut No.')

        fig.legend(loc='upper center',
                   ncol=2,
                   bbox_to_anchor=(0.5, -0.005),
                   )
        return fig, ax

    def plot_probe_refBore(self, plt_ax=None):
        if plt_ax is None:
            fig, ax = plt.subplots(1, 2, constrained_layout=True)
        else:
            ax = plt_ax
            fig = ax.get_figure()

        ax[0].scatter(self.probeData.index,
                      self.probeData['REFBORE-S'],
                      label='REFBORE-S',
                      s=10,
                      )
        ax[0].set_xlabel('Cut No')
        ax[0].set_ylabel('Reference Bore Diameter (mm)')

        sc = ax[1].scatter(self.probeData['REFBORE-X'],
                           self.probeData['REFBORE-Y'],
                           marker='x',
                           s=100,
                           c=np.arange(len(self.probeData)),
                           )
        fig.colorbar(sc, ax=ax[1], label='Cut No')
        ax[1].set_xlabel('Relative Bore X Position (um)')
        ax[1].set_ylabel('Relative Bore Y Position (um)')

        # fig, ax = plt.subplots(1, 3, figsize=(12, 4), layout='tight')
        # ax[0].boxplot(self.probeData['REFBORE-X'], labels=['REFBORE-X'])
        # ax[1].boxplot(self.probeData['REFBORE-Y'], labels=['REFBORE-Y'])
        # ax[2].boxplot(self.probeData['REFBORE-S'], labels=['REFBORE-S'])

        # ax[0].set_ylabel('Relative Bore X Position (um)')
        # ax[1].set_ylabel('Relative Bore Y Position (um)')
        # ax[2].set_ylabel('Bore Diameter (mm)')
        return fig, ax
