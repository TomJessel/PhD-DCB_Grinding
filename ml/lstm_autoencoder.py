# Test file for testing LSTM autoencoder by training on 3 of the 4 datasets
# to then predict the 4th dataset.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from autoencoder import LSTMAutoEncoder
import resources
import matplotlib.pyplot as plt
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

    platform = os.name
    if platform == 'nt':
        onedrive = Path(r'C:\Users\tomje\OneDrive - Cardiff University')
        onedrive = onedrive.joinpath('Documents', 'PHD', 'AE')
        TB_DIR = onedrive.joinpath('Tensorboard')
    elif platform == 'posix':
        onedrive = Path(r'/mnt/c/Users/tomje/OneDrive - Cardiff University')
        onedrive = onedrive.joinpath('Documents', 'PHD', 'AE')
        TB_DIR = onedrive.joinpath('Tensorboard')
    print(f'TB logdir: {TB_DIR}')

    exps = ['Test 5', 'Test 7', 'Test 8', 'Test 9']
    # exps = ['Test 5']
    rms = {}

    for test in exps:
        rms[test] = resources.ae.RMS(test)
        rms[test].data.drop(['0', '1', '2'], axis=1, inplace=True)
    
    try:
        rms['Test 5'].data.drop(['23', '24'], axis=1, inplace=True)
    except KeyError:
        pass

    for test in exps:
        rms[test]._data = rms[test].data.iloc[50:350, :].reset_index(drop=True)
        rms[test]._data = rms[test].data.apply(remove_dc, axis=0)

    for i, val_exp in enumerate(exps):
        print(f'Experiment {val_exp} is the validation set.')

        print('\tCombining training data...')
        dfs = []
        for exp in exps:
            if exp != val_exp:
                dfs.append(rms[exp].data.iloc[:, :50].values)
        join_df = np.concatenate(dfs, axis=1)
        join_df = pd.DataFrame(join_df)
        print(f'\t{np.shape(join_df)}')

        join_rms = join_rms_obj(join_df, f'!{val_exp}')

        print('\tCreate Autoencoder...')
        autoe = LSTMAutoEncoder(join_rms,
                                join_rms.data,
                                tb=True,
                                tb_logdir='LSTMAE_combined_datasets',
                                train_slice=(0, 149),
                                val_frac=0.33,
                                params={'epochs': 300,
                                        'batch_size': 64,
                                        'n_size': [256, 128, 64],
                                        'seq_len': 100,
                                        'n_bottleneck': 32,
                                        'loss': 'mean_squared_error',
                                        'callbacks': [
                                            tf.keras.callbacks.EarlyStopping(
                                                monitor='val_loss',
                                                patience=10,
                                                mode='min',
                                                start_from_epoch=75,
                                            ),
                                        ]
                                        }
                                )
        
        name = autoe.run_name
        model_folder = TB_DIR.joinpath(autoe._tb_logdir.joinpath(name))
        if not os.path.exists:
            os.makedirs(model_folder)
        assert os.path.exists(model_folder)

        autoe.model.callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_folder.joinpath(f'{name}.h5'),
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                save_weights_only=True,
            )
        )

        print('\tTraining...')
        autoe.fit(x=autoe.train_data,
                  val_data=autoe.val_data,
                  verbose=0,
                  )

        print('\tLoading best weights...')
        autoe.model.model_.load_weights(
            TB_DIR.joinpath(model_folder.joinpath(f'{name}.h5'))
        )
        autoe.pred = None
        autoe.scores = None

        df_val = rms[val_exp].data
        jr_val = []
        for i in range(np.shape(df_val)[1]):
            jr_val.extend(df_val.iloc[:, i].values.T)
        jr_val = np.array(jr_val).reshape(-1, 1)
        print(f'\tNo of RMS samples in Val df: {np.shape(jr_val)}')
        assert ~np.isnan(jr_val).any(), 'NaN values in RMS data'

        jr_val = autoe.scaler.transform(jr_val)
        seq_val = autoe.seqence_inputs(jr_val, autoe.seq_len)
        print(f'\tSequenced data shape: {np.shape(seq_val)}')

        print('\tScoring...')
        print(f'\t!{val_exp}')
        (_, pred_val), scores = autoe['join_rms'].score(x=(seq_val))

        # Prediction plot
        print('\tPrediction Plot...')
        fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        fig.suptitle(f'{autoe.RMS.exp_name} - Resconstruction')

        ax = autoe.pred_plot(10000, input(seq_val, pred_val), plt_ax=ax)

        fig_name = TB_DIR.joinpath(model_folder.joinpath(f'{name}_pred.png'))
        fig.savefig(fig_name)
        print(f'\tSaved predicition fig to {fig_name}')

        # Scatter error plot
        print('\tScatter Error Plot...')
        metric = ['mse', 'mae']
        features = ['Runout', 'Form error']

        fig, ax = plt.subplots(len(metric), len(exps),
                               figsize=(10, 6),
                               constrained_layout=True,
                               sharex='col',
                               dpi=500,
                               )

        ax2 = []

        try:
            ax.ravel()
            for a in ax.ravel():
                ax2.append(a.twinx())
        except AttributeError:
            ax2.append(ax.twinx())

        for a, b in zip(ax2[1:], ax2[0:-1]):
            a.sharey(b)

        if len(metric) * len(exps) > 1:
            ax2 = np.reshape(ax2, ax.shape)

        axes = fig.axes
        axes2 = axes[-(len(axes) // 2):]
        axes = axes[0:(len(axes) // 2)]
        axes[i].set_title(val_exp)

        exp = resources.load(val_exp)
        for j, met in enumerate(metric):
            score = scores[met]
            axes[(i + j)].scatter(x=range(len(score)),
                                  y=score,
                                  s=2,
                                  label=met,
                                  )

            for feature in features:
                feat = exp.features[feature].drop([0, 1, 2])
                if test == 'Test 5':
                    feat = feat.drop([23, 24])
                axes2[(j) + i].plot(range(0, len(scores[met]), 300),
                                    feat,
                                    label=feature
                                    )

            axes[(j) + i].set_xlabel('')
            axes[(j) + i].set_ylabel('')
            if i == 0:
                axes[(j) + i].set_ylabel(f'{met.upper()}')
            if i == len(exps) - 1:
                axes2[(j) + i].set_ylabel('Errors')

        _ = fig.supxlabel('Cut Number')

        for i, a in enumerate(axes2):
            if (i + 1) % 1 != 0:
                plt.setp(a.get_yticklabels(), visible=False)

        l1, lab1 = axes[0].get_legend_handles_labels()
        l2, lab2 = axes2[0].get_legend_handles_labels()

        plt.figlegend(l1 + l2,
                      ['Metric'] + lab2,
                      loc='upper center',
                      bbox_to_anchor=(0.5, 0),
                      ncol=len(l1 + l2)
                      )

        fig_name = TB_DIR.joinpath(
            model_folder.joinpath(f'{name}_scatter.png')
        )
        fig.savefig(fig_name)
        print(f'\tSaved scatter fig to {fig_name}')
