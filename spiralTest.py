from matplotlib import pyplot as plt

from src import config, load
from src.nc4.BJDSprialScan import processExpSprialScans, spiralPlotter

HOME_DIR, BASE_DIR, CODE_DIR, TB_DIR, RMS_DATA_DIR = config.config_paths()

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

    exp = load('Test 18')

    sprialScans = processExpSprialScans(exp,
                                        SCPath,
                                        nomDia=NOM_DIA,
                                        feedrate=FEEDRATE,
                                        rpm=RPM,
                                        fs=FS,
                                        yOffset=YOFFSET,
                                        calFeedrate=CALFEEDRATE,
                                        )
    
    scPlotter = spiralPlotter(sprialScans, 18).plot()

    plt.show()
