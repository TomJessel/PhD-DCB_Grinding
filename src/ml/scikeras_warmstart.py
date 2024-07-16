import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from autoencoder import LSTMAutoEncoder, load_model
import src
import numpy as np
import pandas as pd
from pathlib import PurePosixPath as Path


def remove_dc(sig):
    return sig - np.nanmean(sig)


class join_rms_obj:
    def __init__(self, data, exp_name):
        self.data = data
        self.exp_name = exp_name


if __name__ == '__main__':

    _, _, CODE_DIR, TB_DIR, RMS_DATA_DIR = src.config_paths()
    print(f'TB logdir: {TB_DIR}')

    exps = ['Test 5', 'Test 7', 'Test 8', 'Test 9']
    rms = {}

    for test in exps:
        rms[test] = src.ae.RMS(test)
        rms[test].data.drop(['0', '1', '2'], axis=1, inplace=True)
    
    try:
        rms['Test 5'].data.drop(['23', '24'], axis=1, inplace=True)
    except KeyError:
        pass

    for test in exps:
        rms[test]._data = rms[test].data.iloc[50:350, :].reset_index(drop=True)
        rms[test]._data = rms[test].data.apply(remove_dc, axis=0)

    for i, val_exp in enumerate(exps):
        print('\n')
        print('=' * 50)
        print(f'Experiment {val_exp} is the validation set.')
        print('=' * 50)

        print('Combining training data...')
        dfs = []
        for exp in exps:
            if exp != val_exp:
                dfs.append(rms[exp].data.iloc[:, :50].values)
        join_df = np.concatenate(dfs, axis=1)
        join_df = pd.DataFrame(join_df)
        print(f'No data files: \t{np.shape(join_df)[1]}')
        print('-' * 50)

        join_rms = join_rms_obj(join_df, f'!{val_exp}')

        # print('Create Autoencoder...')
        # autoe = LSTMAutoEncoder(join_rms,
        #                         join_rms.data,
        #                         tb=False,
        #                         tb_logdir='tmp/LSTMAE_combined_datasets',
        #                         train_slice=(0, 50),
        #                         val_frac=0.33,
        #                         params={'epochs': 100,
        #                                 'batch_size': 64,
        #                                 'n_size': [256, 128, 64],
        #                                 'seq_len': 100,
        #                                 'n_bottleneck': 32,
        #                                 'loss': 'mean_squared_error',
        #                                 'callbacks': [
        #                                     tf.keras.callbacks.EarlyStopping(
        #                                         monitor='val_loss',
        #                                         patience=10,
        #                                         mode='min',
        #                                         # start_from_epoch=75,
        #                                     ),
        #                                 ]
        #                                 }
        #                         )

        # name = autoe.run_name
        # model_folder = TB_DIR.joinpath(autoe._tb_logdir.joinpath(name))
        # if not os.path.exists(model_folder):
        #     os.makedirs(model_folder)
        # assert os.path.exists(model_folder)

        # autoe.model.callbacks.append(
        #     tf.keras.callbacks.ModelCheckpoint(
        #         filepath=model_folder.joinpath(f'{name}.h5'),
        #         monitor='val_loss',
        #         mode='min',
        #         save_best_only=True,
        #         save_weights_only=True,
        #     )
        # )
        # autoe.fit(x=autoe.train_data,
        #           val_data=autoe.val_data,
        #           verbose=1,
        #           )
        
        # autoe.model.model_.load_weights(
        #     TB_DIR.joinpath(model_folder.joinpath(f'{name}.h5'))
        # )

        # autoe.pred = None
        # autoe.scores = None

        # mod_pkl = autoe.save_model()

        # fig, ax = autoe.loss_plot()
        # fig_name = TB_DIR.joinpath(
        #     model_folder.joinpath(f'{name}_loss_pre.png')
        # )
        # fig.savefig(fig_name)
        # print(f'Saved loss plot to {fig_name}')

        pkl_files = [
            'LSTMAE-!Test_5-WIN-100-E-300-L-[256, 128, 64, 32, 64, 128, 256]-20230823-095028',
            'LSTMAE-!Test_7-WIN-100-E-300-L-[256, 128, 64, 32, 64, 128, 256]-20230824-101436',
            'LSTMAE-!Test_8-WIN-100-E-300-L-[256, 128, 64, 32, 64, 128, 256]-20230824-230832',
            'LSTMAE-!Test_9-WIN-100-E-300-L-[256, 128, 64, 32, 64, 128, 256]-20230825-060122',
        ]

        print('Loaded AUTOE: ')

        autoe_2 = load_model(f'LSTMAE_combined_datasets/{pkl_files[i]}')
        autoe_2.set_run_name(append='TL')
        print(autoe_2.run_name)

        name = autoe_2.run_name
        model_folder = TB_DIR.joinpath(autoe_2._tb_logdir.joinpath(name))
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        assert os.path.exists(model_folder)

        autoe_2.model.warm_start = True
        print(f'Warm start: {autoe_2.model.warm_start}')

        autoe_2.fit(
            x=autoe_2.train_data,
            val_data=autoe_2.val_data,
            epochs=10,
            verbose=1,
        )

        autoe_2.model.model_.load_weights(
            TB_DIR.joinpath(model_folder.joinpath(f'{name}.h5'))
        )

        autoe_2.pred = None
        autoe_2.scores = None

        fig, ax = autoe_2.loss_plot()
        fig_name = TB_DIR.joinpath(
            model_folder.joinpath(f'{name}_loss_post.png')
        )
        fig.savefig(fig_name)
        print(f'Saved loss plot to {fig_name}')
        del autoe_2
