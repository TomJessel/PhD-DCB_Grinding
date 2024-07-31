import sys, os # noqa
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from nptdms import TdmsFile
from scipy.ndimage import uniform_filter1d
from scipy import signal
import circle_fit

from src import config_paths

HOME_DIR, BASE_DIR, _, _, _ = config_paths()
TESTING_DIR = BASE_DIR / 'AE/Testing'


def readNC4(filepath) -> list[float]:
    """
    Read NC4 data from TDMS file into memory.

    Args:
        fno: TDMS file number to read into memory

    Returns:
        NC4 data from the file.
    """
    test = TdmsFile.read(filepath)
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


def sampleandpos(filepath) -> tuple[list, list, list, list]:
    """
    Load in NC4 voltage data and select most appropriate section of \
        the signal to carry forward.

    Args:
        fno: File number to sample from.

    Returns:
        A tuple containing the signal sample and y position for both the \
            positive and negative signal.
    """
    data = readNC4(filepath)
    filt = 50
    scale = 1
    ysteps = np.around(np.arange(0.04, -0.02, -0.01), 2)
    rpy = 4
    spr = 1
    clip = 0.5
    gap = 0.4
    ts = 1 / _fs
    gapsamples = int(gap / ts)
    nosections = int(2 * len(ysteps))
    lentime = float(nosections) * rpy * spr + gap
    lensamples = int(lentime / ts)
    seclensamples = math.ceil(rpy * spr / ts)
    vs = math.ceil((clip * spr) / ts) - 1
    ve = int(seclensamples - ((clip * spr) / ts)) - 1

    vfilter = uniform_filter1d(data, size=filt)
    if scale == 1:
        datarange = (np.amax(vfilter), np.amin(vfilter))
        voltage = 5 * (
            (vfilter - datarange[1]) / (datarange[0] - datarange[1])
        )
    else:
        voltage = vfilter

    voltage = voltage[-(lensamples + 1):]
    vsec = np.empty(shape=(nosections, seclensamples), dtype=object)
    for sno in range(nosections):
        if sno <= (nosections - 1) / 2:
            vsec[sno][:] = voltage[
                (sno * seclensamples):((sno + 1) * seclensamples)
            ]
        else:
            vsec[sno][:] = voltage[
                ((sno * seclensamples) + gapsamples):
                (((sno + 1) * seclensamples) + gapsamples)
            ]

    vsample = vsec[:, vs:ve]
    voff = np.sum((vsample - 2.5) ** 2, axis=1)

    psec = np.argmin(voff[0:math.ceil((nosections - 1) / 2)])
    psample = vsample[:][psec]
    posy = ysteps[psec]

    nsec = np.argmin(voff[math.ceil((nosections - 1) / 2):])
    nsample = vsample[:][nsec + math.ceil((nosections - 1) / 2)]
    negy = ysteps[nsec]
    return psample, posy, nsample, negy


def polyvalradius(x: tuple[np.ndarray, float]) -> list[float]:
    """
    Convert NC4 voltage signal to radius, using NC4 calibration constants.

    Args:
        x: Tuple containing the signal sample and its y position.

    Returns:
        List of converting values to radius.
    """
    pval = [-0.000341717477186167,
            0.00459433449011791,
            -0.0237307202784755,
            0.0585315537400639,
            -0.0766338436136931,
            5.15045955887124
            ]

    d = _dcb_diameter
    rad = np.polyval(pval, x[0]) - 5.1 + (d / 2) + x[1]
    return rad


def compute_shift(zipped) -> int:
    """
    Use fft correlation to compute the shift between two signals.

    Args:
        zipped: Zip tuple containing the signals to compare between each other.

    Returns:
        Int representing the number of samples of shift between the signals.
    """
    x = zipped[0]
    y = zipped[1]
    assert len(x) == len(y)
    c = signal.correlate(x, y, mode='same', method='fft')
    assert len(c) == len(x)
    zero_index = int(len(x) / 2) - 1
    shift = zero_index - np.argmax(c)
    return shift


def _fitcircles(radius: np.ndarray):
    """
    Fit a circle to each NC4 measurements and return its properties.

    Args:
        radius: Array of radius to calc attributes for

    Returns:
        List of tuples containing the x and y coords, the radius of the \
            circle and the variance.
    """
    x = np.array([np.multiply(r, np.sin(_theta)) for r in radius])
    y = np.array([np.multiply(r, np.cos(_theta)) for r in radius])
    xy = np.array(list(zip(x, y))).transpose([0, 2, 1])
    circle = [circle_fit.hyper_fit(xy[0])]
    runout = np.array(
        [2 * (np.sqrt(x[0] ** 2 + x[1] ** 2)) for x in circle]
    )
    mean_radius = np.array(
        [x[2] for x in circle]
    )
    peak_radius = np.array(
        [np.max(rad) for rad in radius]
    )
    form_error = np.array(
        [(np.max(rad) - np.min(rad)) for rad in radius]
    )
    return mean_radius, peak_radius, runout, form_error


def alignposneg(prad,
                nrad
                ) -> np.ndarray:
    """
    Combine the pos and neg halfs of the signal together.

    Args:
        prad: Array of radius values for positive half of signal.
        nrad: Array of radius values for negative half of signal.

    Returns:
        Array of combined radius signal
    """
    pradzero = np.subtract(np.transpose(prad), np.mean(prad, axis=1))
    nradzero = np.subtract(np.transpose(nrad), np.mean(nrad, axis=1))
    # print('Working out Lags')
    radzeros = list(zip(np.transpose(pradzero), np.transpose(nradzero)))
    lag = [compute_shift(radzeros[0])]
    nrad = np.array([np.roll(row, -x) for row, x in zip(nrad, lag)])
    radii = np.array([(p + n) / 2 for p, n in zip(prad, nrad)])
    return radii


def _compute_nc4(filepath):
    psample, posy, nsample, negy = sampleandpos(filepath)

    prad = polyvalradius((psample, posy))
    nrad = polyvalradius((nsample, negy))

    prad = np.transpose(np.reshape(prad, (-1, 1)))
    nrad = np.transpose(np.reshape(nrad, (-1, 1)))

    radii = alignposneg(prad, nrad)

    st = np.argmin(radii[0, 0:int(_fs)])
    rpy = 4
    clip = 0.5

    radius = radii[
        :, np.arange(
            st, st + (radii.shape[1]) / (rpy - (2 * clip)), dtype=int
        )
    ]
    mean_radius, peak_radius, runout, form_error = _fitcircles(
        radius
    )

    atts = {}
    atts['mean_radius'] = mean_radius[0]
    atts['peak_radius'] = peak_radius[0]
    atts['runout'] = runout[0]
    atts['form_error'] = form_error[0]

    return radii, atts


def disp_atts(att: dict):
    """
    Display the attributes of the NC4 data.

    Args:
        att: Dictionary of attributes to display.
    """
    for key, val in att.items():
        print(f' {key}: \t{val:6f}')


if __name__ == "__main__":

    # Common usage:
    # python testingUtils/nc4-runout.py -p "24_07_17_ToolSetupRunout"

    parser = argparse.ArgumentParser(
        description='Convert raw NC4 data to radius.'
    )

    parser.add_argument('filepath',
                        type=str,
                        help='Rel path to TDMS file to convert'
                        )
    parser.add_argument('-fs',
                        default=50_000,
                        type=int,
                        help='Sampling frequency of the NC4'
                        )
    parser.add_argument('-dia',
                        default=1.3,
                        type=float,
                        help='Diameter of the DCB'
                        )
    parser.add_argument('-p', '--plot',
                        action='store_true',
                        help='Plot converted the data'
                        )
    
    args = parser.parse_args()

    filepath = TESTING_DIR / args.filepath
    # if filepath is a folder convert last tdms file
    if filepath.is_dir():
        filepath = list(filepath.glob('*.tdms'))[-1]

    assert filepath.exists(), f'Filepath does not exist: {filepath}'

    _fs = args.fs
    _dcb_diameter = args.dia
    _theta = 2 * np.pi * np.arange(0, 1, 1 / _fs)

    rad, atts = _compute_nc4(filepath)
    print(f'NC4 Scan - {filepath.stem}:')
    disp_atts(atts)

    if args.plot:
        plt.figure(figsize=(8.5, 4.8))
        plt.plot(rad[0])
        plt.axhline(y=atts['mean_radius'], color='r', linestyle='--')
        plt.xlabel('Sample')
        plt.ylabel('Radius (mm)')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.title(f'NC4 Scan: {filepath.stem}')
        plt.subplots_adjust(right=0.8)
        txt = f"Mean Rad: {atts['mean_radius']:.3f} mm\n" \
              f"Peak Rad: {atts['peak_radius']:.3f} mm\n\n" \
              f"Runout: {atts['runout'] * 1000:.3f} um\n" \
              f"Form Error: {atts['form_error'] * 1000:.3f} um"
        plt.figtext(0.81, 0.5, txt, fontsize=10)
        plt.savefig(f'{filepath}.png', dpi=300, bbox_inches='tight')
        # plt.show()
