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
                     index_col='Cut No',
                     )
    df['PROBEDIFF'] = df['AVGPROBE'].diff()
    df['REFBORE-X'] = (df['REFBORE-X'] - df['REFBORE-X'].iloc[0]) * 1000
    df['REFBORE-Y'] = (df['REFBORE-Y'] - df['REFBORE-Y'].iloc[0]) * 1000
    return df


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


if __name__ == "__main__":
    filePath = TESTING_DIR.joinpath(r'24_06_10_weartest_D1.3_#1000',
                                    r'240610-WEARTEST.csv',
                                    )
    
    probe_data = readProbeCSV(filePath)
    fig, ax = plotProbeDOC(probe_data, doc=0.03, tolerance=0.01)

    plt.show()
