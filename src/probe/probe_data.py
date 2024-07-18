import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

TESTING_DIR = Path.home().joinpath(r'OneDrive - Cardiff University',
                                   r'Documents',
                                   r'PHD',
                                   r'AE',
                                   r'Testing',
                                   )


def readProbeCSV(filepath):
    df = pd.read_csv(filepath,
                     header=2,
                     )
    df['PROBEDIFF'] = df['AVGPROBE'].diff()
    df['PROBEDIFF'] = df['PROBEDIFF'].fillna(0)
    df['REFBORE-X'] = (df['REFBORE-X'] - df['REFBORE-X'].iloc[0]) * 1000
    df['REFBORE-Y'] = (df['REFBORE-Y'] - df['REFBORE-Y'].iloc[0]) * 1000

    df.set_index('Cut No', inplace=True)
    df['AVGPROBE'] = df['AVGPROBE'] - df['AVGPROBE'].iloc[0]
    return df


def probe_cumulative(df, doc=0.03, tol=0.02, plt_ax=None):
    if plt_ax is None:
        fig, ax = plt.subplots()
    else:
        ax = plt_ax
        fig = ax.get_figure()

    y_measured = df['AVGPROBE'].values - df['AVGPROBE'].iloc[0]
    ax.plot(y_measured, label='Measured DOC',)

    x = np.arange(0, len(df))
    y_ideal = x * doc
    ax.plot(x, y_ideal, 'C0', linestyle='--', alpha=0.5, label='Ideal DOC')

    y_tol = (y_ideal + tol, y_ideal - tol)
    ax.fill_between(x, y_tol[0], y_tol[1],
                    color='C0', alpha=0.4, label='Tolerance'
                    )

    ax.set_ylim(0,)
    ax.set_xlim(0,)
    ax.grid()

    ax2 = ax.twinx()
    ax2.plot(y_ideal - y_measured, label='Error', color='C1')

    ax.set_ylabel('Radial DOC (mm)')
    ax2.set_ylabel('Error (mm)')
    ax.set_xlabel('Cut No.')
    return fig, ax


def plotProbeDOC(df, doc=None, tolerance=None):
    assert 'PROBEDIFF' in df.columns, 'PROBE_DIFF not in DataFrame'

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.scatter(df.index, df['PROBEDIFF'], label='Measured DOC')
    ax.set_ylabel('Measured Radial DOC (mm)')
    ax.set_xlabel('Cut No')
    ax.grid()
    if doc is not None:
        ax.axhline(doc,
                   color='r',
                   alpha=0.8,
                   label='Input DOC',
                   )
        if tolerance is not None:
            tolerance = [doc - tolerance, doc + tolerance]
            for tol in tolerance:
                ax.axhline(tol,
                           color='r',
                           linestyle='--',
                           alpha=0.8,
                           label='Tolerance',
                           )
    return fig, ax


def refbore_probe(df):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].scatter(df.index, df['REFBORE-S'], label='REFBORE-S', s=10)
    ax[0].set_xlabel('Cut No')
    ax[0].set_ylabel('Reference Bore Diameter (mm)')

    sc = ax[1].scatter(df['REFBORE-X'], df['REFBORE-Y'],
                       marker='x', s=100, c=np.arange(len(df)),
                       )
    fig.colorbar(sc, ax=ax[1], label='Cut No')
    ax[1].set_xlabel('Relative Bore X Position (um)')
    ax[1].set_ylabel('Relative Bore Y Position (um)')

    fig, ax = plt.subplots(1, 3, figsize=(12, 4), layout='tight')
    ax[0].boxplot(df['REFBORE-X'], labels=['REFBORE-X'])
    ax[1].boxplot(df['REFBORE-Y'], labels=['REFBORE-Y'])
    ax[2].boxplot(df['REFBORE-S'], labels=['REFBORE-S'])

    ax[0].set_ylabel('Relative Bore X Position (um)')
    ax[1].set_ylabel('Relative Bore Y Position (um)')
    ax[2].set_ylabel('Bore Diameter (mm)')
    return df[['REFBORE-S', 'REFBORE-X', 'REFBORE-Y']].describe()


def nc4radiusOutput(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df.index, df['NC4 Radius'], label='NC4 Radius')
    ax.set_xlabel('Cut No')
    ax.set_ylabel('NC4 Radius (mm)')
    return fig, ax


if __name__ == "__main__":
    filePath = TESTING_DIR.joinpath(r'24_06_10_weartest_D1.3_#1000',
                                    r'240610-WEARTEST.csv',
                                    )
    
    probe_data = readProbeCSV(filePath)
    fig, ax = plotProbeDOC(probe_data, doc=0.03, tolerance=0.01)

    plt.show()
