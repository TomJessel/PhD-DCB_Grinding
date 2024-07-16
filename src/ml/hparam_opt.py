"""
@File    :   hparam_opt.py
@Author  :   Tom Jessel
@Contact :   jesselt@cardiff.ac.uk

@Creation Time    @Author    @Version    @Description
------------      -------    --------    -----------
16/01/2023 10:03   tomhj      1.0         N/A
"""

from src import MLP_Win_Model
import src

import gc
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import multiprocessing


def opt_model_score(mod):
    mod.cv(n_splits=10, n_repeats=10)
    mod.fit(validation_split=0.1, verbose=0)
    mod.score(plot_fig=False)


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    exp5 = src.load('Test 5')
    exp7 = src.load('Test 7')
    exp8 = src.load('Test 8')
    exp9 = src.load('Test 9')

    dfs = [exp5.features.drop([23, 24]),
           exp7.features,
           exp8.features,
           exp9.features]
    main_df = pd.concat(dfs)
    main_df = main_df.drop(columns=['Runout', 'Form error', 'Peak radius',
                                    'Radius diff'])  # .drop([0, 1, 2, 3])
    main_df.reset_index(drop=True, inplace=True)

    EPOCHS = [500, 1000, 1500]
    LOSS = ['mse']
    NO_NODES = [64, 128]
    DROPOUT = [0.01, 0.1, 0.2, 0.5]
    BATCH_SIZE = [5, 10, 20]
    SEQ_LEN = [5, 10, 15]
    NO_LAYERS = [1, 2, 3, 4]
    INIT_MODE = [
        'glorot_normal',
        'glorot_uniform',
        'he_normal',
        'he_uniform',
        'random_normal'
    ]

    for dropout in DROPOUT:
        for batch_size in BATCH_SIZE:
            for no_layers in NO_LAYERS:
                for init_mode in INIT_MODE:
                    hparams = {
                        'epochs': 1500,
                        'loss': 'mse',
                        'no_nodes': 128,
                        'seq_len': 5,
                        'dropout': dropout,
                        'batch_size': batch_size,
                        'no_layers': no_layers,
                        'init_mode': init_mode,
                    }
                    ml_win_reg = MLP_Win_Model(feature_df=main_df,
                                               target='Mean radius',
                                               tb=True,
                                               tb_logdir='hparam_test_2',
                                               params=hparams
                                               )
                    opt_model_score(ml_win_reg)
                    gc.collect()
