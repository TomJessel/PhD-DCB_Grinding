#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@File    :   testing_main.py
@Author  :   Tom Jessel
@Contact :   jesselt@cardiff.ac.uk

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
09/11/2022 11:08   tomhj      1.0         File for quickly setting up console for checking testing data
"""

import resources


def check_ae(_exp: resources.experiment.Experiment):
    """
    Plots the most recent AE file in the Experiemnt obj.

    Args:
        _exp: Experiment object to use
    """
    print(f'Plotting most recent AE file...')
    _exp.ae.plotAE(-1)


def check_nc4(_exp: resources.experiment.Experiment):
    print('Checking most recent NC4 file...')
    _exp.nc4.check_last()
    _exp.save()
    # todo change so first one has always been calculated


def main() -> resources.experiment.Experiment:
    try:
        _exp = resources.load()
    except NotADirectoryError:
        _exp = resources.experiment.create_obj()

    print(f'{"-" * 22}TESTING EXP FILE{"-" * 22}')
    print(_exp)
    print('-' * 60)
    _exp.update()
    _exp.save()
    return _exp


if __name__ == "__main__":
    exp = main()
    # check_ae(exp)
