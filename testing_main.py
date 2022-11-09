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

try:
    exp = resources.load()
except NotADirectoryError:
    exp = resources.experiment.create_obj()

print(f'{"-"*22}TESTING EXP FILE{"-"*22}')
print(exp)
print('-'*60)
exp.update()
exp.save()
