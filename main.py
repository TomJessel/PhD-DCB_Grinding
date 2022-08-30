#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import Experiment

if __name__ == '__main__':
    # exp = Experiment.load()
    # exp.save()
    exp = Experiment.create_obj()

# todo .all() doesn't work when creating new obj  possibly inbuild this part into the load and createobj funcs)
# todo add methods to update objects and also print progress of tests
# todo possibly move functionality of creating and loading objects into Experiment.py file
# todo change file location saving to relative paths not absolute paths
