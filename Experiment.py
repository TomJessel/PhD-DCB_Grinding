# Class for test objects with methods
import TestInfo
import AE
import pickle
import NC4


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

