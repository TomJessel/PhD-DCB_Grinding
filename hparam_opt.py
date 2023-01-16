"""
@File    :   hparam_opt.py
@Author  :   Tom Jessel
@Contact :   jesselt@cardiff.ac.uk

@Creation Time    @Author    @Version    @Description
------------      -------    --------    -----------
16/01/2023 10:03   tomhj      1.0         N/A
"""

from ml_mlp import MLP_Model, MLP_Win_Model, LSTM_Model
import resources

import gc
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import multiprocessing
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass

def opt_model_score(mod):
    mod.cv(n_splits=10, n_repeats=10)
    mod.fit(validation_split=0.1, verbose=0)
    mod.score(plot_fig=False)
    del mod


if __name__ == "__main__":
    exp5 = resources.load('Test 5')
    exp7 = resources.load('Test 7')
    exp8 = resources.load('Test 8')
    exp9 = resources.load('Test 9')

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
    BATCH_SIZE = [15]
    SEQ_LEN = [5, 10, 15]
    NO_LAYERS = [1, 2, 3, 4]
    INIT_MODE = [
        'glorot_normal',
        'glorot_uniform',
        'he_normal',
        'he_uniform',
        'random_normal'
    ]

    for epoch in EPOCHS:
        for loss in LOSS:
            for no_nodes in NO_NODES:
                for seq_len in SEQ_LEN:
                    hparams = {
                        'epochs': epoch,
                        'loss': loss,
                        'no_nodes': no_nodes,
                        'seq_len': seq_len,
                    }
                    lstm_reg = LSTM_Model(feature_df=main_df,
                                          target='Mean radius',
                                          tb=True,
                                          tb_logdir='hparam_test_4',
                                          params=hparams
                                          )
                    opt_model_score(lstm_reg)
                    del lstm_reg
                    gc.collect()
