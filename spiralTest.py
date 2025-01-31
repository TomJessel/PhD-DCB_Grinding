from matplotlib import pyplot as plt

from src import config, load
from src.nc4.BJDSprialScan import processExpSprialScans, spiralPlotter
from src.nc4.BJDSprialScan import NC4SpiralScan

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

    # exp = load('Test 25')

    # sprialScans = processExpSprialScans(exp,
    #                                     SCPath,
    #                                     nomDia=NOM_DIA,
    #                                     feedrate=FEEDRATE,
    #                                     rpm=RPM,
    #                                     fs=FS,
    #                                     yOffset=YOFFSET,
    #                                     calFeedrate=CALFEEDRATE,
    #                                     )
    
    # scPlotter = spiralPlotter(sprialScans, exp.test_info.testno).plot()
    SCANpath = BASE_DIR.joinpath(
        r'AE',
        r'Testing',
        r'24_10_08_weartest_D1.3_#1000',
        r'Spiral Scans',
        r'Cut_058_2024_10_08_16_24_38-Ch0-50kHz.tdms'
    )
    assert SCANpath.exists(), "Spiral scan file not found."

    sc = NC4SpiralScan(scanPath=SCANpath,
                       sCurvePath=SCPath,
                       toolNomDia=NOM_DIA,
                       zSpiralFeedrate=FEEDRATE,
                       spindleSpeed=RPM,
                       fs=FS,
                       yOffset=YOFFSET,
                       scFeedrate=CALFEEDRATE)
    sc.plotSpiralScan(Zsection=0,
                      vmin=0.6,
                      vmax=0.67,)
    plt.show()
