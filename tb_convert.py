"""
@File    :   tb_convert.py
@Author  :   Tom Jessel
@Contact :   jesselt@cardiff.ac.uk

@Creation Time    @Author    @Version    @Description
------------      -------    --------    -----------
09/01/2023 09:38   tomhj      1.0         N/A
"""
import os
import pandas as pd
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator


def parse_tensorboard(path, scalars):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={'tensors': 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    assert all(
        s in ea.Tags()["tensors"] for s in scalars
    ), "some scalars were not found in the event accumulator"

    results = {scalar: pd.DataFrame(
        [(s, tf.make_ndarray(t)) for _, s, t in ea.Tensors(
            scalar)], dtype='float32', columns=['step', scalar]).set_index(
        'step')
               for scalar in scalars}
    return pd.concat(results.values(), axis=1)


if __name__ == "__main__":
    os.chdir('../../ml/Tensorboard')
    # path = r"MLP_WIN/hparam_test/MLP_Win-WLEN-5-E-1500-B-10-L-[64 " \
    #        r"64]-D-0.1-20230106-165556/events.out.tfevents.1673025279" \
    #        r".DESKTOP-9434MAQ.8359.1.v2"
    scalars = ['cv_iter/mse', 'cv_iter/mae', 'cv_iter/r2']
    # cv_results = parse_tensorboard(path, scalars)


    folder = r'MLP_WIN/hparam_test/'
    dirs = os.listdir(folder)
    a = ['MLP_Win-', 'MLP-', 'LSTM-']
    dirs = [x for x in dirs if any(a in x for a in a)]
    dirs.sort()

    df = {x: parse_tensorboard(os.path.join(folder, x), scalars)
                            for x in dirs}
    df = pd.concat(df.values(), keys=df.keys())