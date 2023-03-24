"""
@File    :   autoencoder.py
@Author  :   Tom Jessel
@Contact :   jesselt@cardiff.ac.uk

@Creation Time    @Author    @Version    @Description
------------      -------    --------    -----------
27/02/2023 11:23   tomhj      1.0         N/A
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from textwrap import dedent
from typing import Any
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model
import tensorboard.plugins.hparams.api as hp
import tensorflow_addons as tfa
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import time
from scikeras.wrappers import KerasRegressor, BaseWrapper

import resources

DATA_DIR = Path.home().joinpath(r'Testing/RMS')
TB_DIR = Path.home().joinpath(r'ml/Tensorboard/AUTOE')


def _mp_rms_process(fno):
    avg_size = 100000
    sig = exp.ae.readAE(fno)
    sig = pd.DataFrame(sig)
    sig = sig.pow(2).rolling(500000).mean().apply(np.sqrt, raw=True)
    sig = np.array(sig)[500000 - 1:]
    avg_sig = np.nanmean(np.pad(sig.astype(float),
                                ((0, avg_size - sig.size % avg_size), (0, 0)),
                                mode='constant',
                                constant_values=np.NaN
                                )
                         .reshape(-1, avg_size), axis=1)
    return avg_sig


def mp_get_rms(fnos):
    with mp.Pool(processes=20, maxtasksperchild=1) as pool:
        rms = list(tqdm(pool.imap(
            _mp_rms_process,
            fnos),
            total=len(fnos),
            desc='RMS averaging'
        ))
        pool.close()
        pool.join()
    return rms


def pred_and_score(mod, input_data):
    pred = mod.predict(input_data, verbose=0)
    mae = mean_absolute_error(input_data.T, pred.T, multioutput='raw_values')
    mse = mean_squared_error(input_data.T, pred.T, multioutput='raw_values')
    r2 = r2_score(input_data.T, pred.T, multioutput='raw_values')

    print(f'\tMAE: {np.mean(mae):.5f}')
    print(f'\tMSE: {np.mean(mse):.5f}')
    print(f'\tR2: {np.mean(r2):.5f}')

    scores = {'mae': mae, 'mse': mse, 'r2': r2}
    return pred, scores


def pred_plot(mod, input, pred, no):
    pred_input = input[no, :].reshape(-1, mod.n_inputs)
    x_pred = pred[no, :].reshape(-1, mod.n_inputs)

    pred_input = mod.scaler.inverse_transform(pred_input)
    x_pred = mod.scaler.inverse_transform(x_pred)

    mse = mean_squared_error(pred_input, x_pred)
    mae = mean_absolute_error(pred_input, x_pred)

    fig, ax = plt.subplots()
    ax.plot(pred_input.T, label='Real')
    ax.plot(x_pred.T, label='Predicition')
    ax.legend()
    ax.set_title(f'MAE: {mae:.4f} MSE: {mse:.4f}')
    return fig, ax


def scatter_scores(scores):
    # presumes scores is a dict with (mae, mse, r2)
    fig, ax = plt.subplots(1, 3)
    ax[0].scatter(x=range(len(scores['mae'])),
                  y=scores['mae'],
                  color='b',
                  label='mae'
                  )
    ax[1].scatter(x=range(len(scores['mse'])),
                  y=scores['mse'],
                  color='g',
                  label='mse'
                  )
    ax[2].scatter(x=range(len(scores['r2'])),
                  y=scores['r2'],
                  color='r',
                  label='r2'
                  )
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    fig.suptitle('Scores')
    fig.tight_layout()
    return fig, ax


class RMS:
    def __init__(
            self,
            exp_name,
    ):
        self.exp_name = exp_name.upper().replace(" ", "_")

        print(f'\nLoaded {exp_name} RMS Data')

        # Read in data from file or compute
        self._get_data()

    def _process_exp(self, save_path=None):
        # load in exp for this obj
        global exp
        exp = resources.load(self.exp_name)

        # get no of AE files in exp dataset
        fnos = range(len(exp.ae._files))

        # process data, requires func outside class for mp
        rms = mp_get_rms(fnos)

        # find min length of signals and create array
        m = min([r.shape[0] for r in rms])
        rms = [r[:m] for r in rms]
        rms = np.array(rms)

        # create dataframe from array each column is a cut
        df = pd.DataFrame(rms.T)

        if save_path is not None:
            df.to_csv(save_path,
                      encoding='utf-8',
                      index=False)
            print(f'Data file saved to: {save_path}')
        del exp
        return df

    def _get_data(self):
        # get file name of .csv file if created
        file_name = f'RMS_{self.exp_name}.csv'

        # join path of home dir, data folder, and file name for reading
        path = DATA_DIR.joinpath(file_name)

        try:
            # try to read in .csv file
            self.data = pd.read_csv(path)
        except FileNotFoundError:
            print(f'RMS Data File Not Found for {self.exp_name}')
            # if no data file process data and save
            self.data = self._process_exp(path)


class AutoEncoder():
    def __init__(
        self,
        rms_obj: RMS,
        tb: bool = True,
        tb_logdir: str = '',
        params: dict = None,
        train_slice=(0, 100),
        random_state=None,
    ):
        print('AutoEncoder Model')
        self.RMS = rms_obj
        self.data = rms_obj.data
        self._train_slice = np.s_[train_slice[0]:train_slice[1]]
        self.random_state = random_state
        self._tb = tb
        self._tb_logdir = TB_DIR.joinpath(tb_logdir)

        if params is None:
            params = {}
        self.params = params

        self.pre_process()
        self.model = self.initialise_model(**self.params)
        print(f'\n{self.run_name}')
        self.model.initialize(X=self.train_data)
        self.model.model_.summary()
        print()

    def pre_process(
        self,
        val_frac: float = 0.1,
    ):
        print('Pre-Processing Data:')

        # First split off Test data based on slice from self._train_slice
        # to let the model only be trained ona  portion of the data.
        # i.e. first 100 cuts

        print(f'\tTraining Data: {self._train_slice}')
        data = self.data.values.T
        train_data = data[self._train_slice]

        self.scaler = MinMaxScaler(feature_range=(0, 1))

        if self.random_state is None:
            x_train, x_test = train_test_split(train_data,
                                               test_size=val_frac,)
        else:
            x_train, x_test = train_test_split(train_data,
                                               test_size=val_frac,
                                               random_state=self.random_state,)

        print(f'\tInput train shape: {x_train.shape}')
        print(f'\tInput val shape: {x_test.shape}')

        self.scaler.fit(x_train)
        x_train = self.scaler.transform(x_train)
        x_test = self.scaler.transform(x_test)

        self._n_inputs = x_train.shape[1]
        self.train_data = x_train
        self.val_data = x_test

    @staticmethod
    def get_autoencoder(
        n_inputs: int,
        n_bottleneck: int,
        n_size: list[int],
        activation: str,
    ):
        def get_encoder(n_inputs, n_bottleneck, n_size, activation):
            encoder_in = Input(shape=(n_inputs, ))
            e = encoder_in

            for dim in n_size:
                e = Dense(dim, activation=activation)(e)
                e = BatchNormalization()(e)

            encoder_out = Dense(n_bottleneck, activation='relu')(e)
            encoder = Model(encoder_in, encoder_out, name='Encoder')
            return encoder

        def get_decoder(n_inputs, n_bottleneck, n_size, activation):
            decoder_in = Input(shape=(n_bottleneck, ))
            d = decoder_in

            for dim in n_size[::-1]:
                d = Dense(dim, activation=activation)(d)
                d = BatchNormalization()(d)

            decoder_out = Dense(n_inputs, activation='relu')(d)
            decoder = Model(decoder_in, decoder_out, name='Decoder')
            return decoder

        encoder = get_encoder(n_inputs, n_bottleneck, n_size, activation)
        decoder = get_decoder(n_inputs, n_bottleneck, n_size, activation)

        autoencoder_in = Input(shape=(n_inputs, ), name='Input')
        encoded = encoder(autoencoder_in)
        decoded = decoder(encoded)
        autoencoder = Model(autoencoder_in, decoded, name='AutoEncoder')

        # self.encoder = encoder
        # self.decoder = decoder
        return autoencoder

    def initialise_model(
        self,
        n_bottleneck: int = 10,
        n_size: list = [64, 64],
        activation: str = 'relu',
        epochs: int = 500,
        loss: str = 'mse',
        metrics: list[str] = ['MSE',
                              'MAE',
                              KerasRegressor.r_squared
                              ],
        optimizer=tf.keras.optimizers.Adam,
        verbose: int = 1,
        callbacks: list[Any] = None,
    ):
        layers = n_size + [n_bottleneck] + n_size[::-1]
        t = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        self.run_name = f'AUTOE-{self.RMS.exp_name}-E-{epochs}-L-{layers}-{t}'

        if callbacks is None:
            callbacks = []
        callbacks.append(tfa.callbacks.TQDMProgressBar(
            show_epoch_progress=False))

        if self._tb:
            callbacks.append(tf.keras.callbacks.TensorBoard(
                log_dir=self._tb_logdir.joinpath(self.run_name),
                histogram_freq=1,
            ))
            tb_writer = tf.summary.create_file_writer(
                f'{self._tb_logdir.joinpath(self.run_name)}')

            with tb_writer.as_default():
                hp_params = self.params
                hp_params.pop('callbacks', None)
                if 'n_size' in hp_params.keys():
                    s = hp_params.pop('n_size')
                    hp_params['n_size'] = str(s)

                hp.hparams(
                    hp_params,
                    trial_id=f'{self._tb_logdir.joinpath(self.run_name)}'
                )

        model = BaseWrapper(
            model=self.get_autoencoder,
            model__n_inputs=self._n_inputs,
            model__n_bottleneck=n_bottleneck,
            model__n_size=n_size,
            model__activation=activation,
            epochs=epochs,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            verbose=verbose,
            callbacks=callbacks,
        )

        return model

    def fit(self, x, val_data, **kwargs):
        self.model.fit(
            X=x,
            y=x,
            validation_data=(val_data, val_data),
            **kwargs
        )

    def score(
            self,
            x: np.ndarray,
            print_score: bool = True,
    ) -> dict:
        pred = self.model.predict(x, verbose=0)

        mae = mean_absolute_error(x, pred, multioutput='raw_values')
        mse = mean_squared_error(x, pred, multioutput='raw_values')
        r2 = r2_score(x, pred, multioutput='raw_values')

        if print_score:
            print(f'\tMAE: {np.mean(mae):.5f}')
            print(f'\tMSE: {np.mean(mse):.5f}')
            print(f'\tR2: {np.mean(r2):.5f}')

        scores = {'mae': mae, 'mse': mse, 'r2': r2}

        if self._tb:
            tb_writer = tf.summary.create_file_writer(
                f'{self._tb_logdir.joinpath(self.run_name)}')

            md_scores = dedent(f'''
                    ### Scores - Validation Data

                     | MAE | MSE |  R2  |
                     | ---- | ---- | ---- |
                     | {np.mean(scores['mae']) * 1e3:.5f} |\
                         {np.mean(scores['mse']) * 1e6:.5f} |\
                             {np.mean(scores['r2']):.5f} |

                    ''')
            with tb_writer.as_default():
                tf.summary.text('Model Info', md_scores, step=2)
                tf.summary.scalar(
                    'Val MAE',
                    np.mean(scores['mae']),
                    step=1,
                )
                tf.summary.scalar(
                    'Val MSE',
                    np.mean(scores['mse']),
                    step=1,
                )
                tf.summary.scalar(
                    'Val R\u00B2',
                    np.mean(scores['r2']),
                    step=1,
                )
        return pred, scores


if __name__ == '__main__':

    # exps = ['Test 5', 'Test 7', 'Test 8', 'Test 9']
    exps = ['Test 5']

    rms = {}
    for test in exps:

        rms[test] = RMS(test)

    rms['Test 5'].data.drop(['23', '24'], axis=1, inplace=True)
    print()

    for test in exps:

        autoe = AutoEncoder(rms[test],
                            random_state=1,
                            train_slice=(0, 100),
                            tb=True,
                            tb_logdir=rms[test].exp_name,
                            params={'n_bottleneck': 10,
                                    'n_size': [64, 64],
                                    'epochs': 500,
                                    'loss': 'mse',
                                    }
                            )

        autoe.fit(
            x=autoe.train_data,
            val_data=autoe.val_data,
            verbose=0
        )

        print('\nValidation Scores:')
        autoe.score(autoe.val_data)

        # # plot hist of loss values from training
        # p = autoe.predict(autoe.train_data, verbose=0)
        # train_mse = mean_squared_error(autoe.train_data, p,
        #                                 multioutput='raw_values')

        # train_mae = mean_absolute_error(autoe.train_data, p,
        #                                multioutput='raw_values')

        # train_r2 = r2_score(autoe.train_data, p,
        #                    multioutput='raw_values')

        # # fig, ax = plt.subplots()
        # # ax.hist(train_mse, bins=50)
        # # ax.set_xlabel('Error')
        # # ax.set_ylabel('No of Occurences')
        # # ax.set_title(f'{test} Histogram of Training Dataset Prediction Error')

        # # calc thresholds from each metric based on mean and std
        # thr_mse = np.mean(train_mse) + np.std(train_mse)
        # thr_mae = np.mean(train_mae) + np.std(train_mae)
        # thr_r2 = np.mean(train_r2) - np.std(train_r2)

        # print(f'\nThreshold:')
        # print(f'\tMSE = {thr_mse}')
        # print(f'\tMAE = {thr_mae}')
        # print(f'\tR2 = {thr_r2}')

        # # prediction plot of cut 8
        # fig, ax = pred_plot(autoe, autoe.val_data, autoe_val_pred, 8)
        # ax.set_title(f'{test} Validation Predictions - {ax.get_title()}')

        # # Prediction on whole dataset
        # print(f'\nDataset Predictions:')
        # # first scale whole dataset
        # d = autoe.scaler.transform(autoe.data.values.T)
        # autoe_pred, autoe_scores = pred_and_score(autoe,
        #                                           d)

        # # plot scatter scores with the threshold from training scores
        # # TODO need to check anomalies within RMS data especially test 5, 9
        # fig, ax = scatter_scores(scores=autoe_scores)
        # fig.suptitle(f'{test} - {fig._suptitle.get_text()}')
        # ax[0].axhline(thr_mae, color='k', ls='--')
        # ax[1].axhline(thr_mse, color='k', ls='--')
        # ax[2].axhline(thr_r2, color='k', ls='--')

        # def plot_to_tf_im(figure):
        #     """Converts the matplotlib plot specified by 'figure' to a PNG
        #     image and
        #     returns it. The supplied figure is closed and inaccessible after
        #     this call."""
        #     # Save the plot to a PNG in memory.
        #     buf = io.BytesIO()
        #     figure.savefig(buf, format='png')
        #     buf.seek(0)
        #     # Convert PNG buffer to TF image
        #     image = tf.image.decode_png(buf.getvalue(), channels=4)
        #     # Add the batch dimension
        #     image = tf.expand_dims(image, 0)
        #     return image

        # tb_writer = tf.summary.create_file_writer(run_name)
        # with tb_writer.as_default():
        #     tf.summary.image("Scatter Predicitons", plot_to_tf_im(fig), step=0)
        #     for key, score in autoe_scores.items():
        #         step = 0
        #         for error in score:
        #             tf.summary.scalar(f'pred_scores/{key}', error, step=step)
        #             step += 1

        # # prediction plot of cut 110
        # fig, ax = pred_plot(autoe, d, autoe_pred, 110)
        # ax.set_title(f'{test} UNSEEN DATA - {ax.get_title()}')

    # plt.show(block=True)
