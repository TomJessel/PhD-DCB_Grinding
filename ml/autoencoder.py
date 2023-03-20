"""
@File    :   autoencoder.py
@Author  :   Tom Jessel
@Contact :   jesselt@cardiff.ac.uk

@Creation Time    @Author    @Version    @Description
------------      -------    --------    -----------
27/02/2023 11:23   tomhj      1.0         N/A
"""

import os
import io
import pathlib
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model, Sequential
import tensorflow_addons as tfa
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import time
from scikeras.wrappers import KerasRegressor

import resources

DATA_DIR = rf'Testing/RMS'

def _mp_rms_process(fno):
    avg_size = 100000
    sig = exp.ae.readAE(fno)
    sig = pd.DataFrame(sig)
    sig = sig.pow(2).rolling(500000).mean().apply(np.sqrt, raw=True)
    sig = np.array(sig)[500000-1:]
    avg_sig = np.nanmean(np.pad(sig.astype(float),
                                ((0, avg_size - sig.size%avg_size), (0,0)),
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
    return pred, (mae, mse, r2)

def pred_plot(mod, input, pred, no):
    pred_input =input[no,:].reshape(-1, mod.n_inputs)
    x_pred = pred[no,:].reshape(-1, mod.n_inputs)

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
    # presumes scores is a tuple with (mae, mse, r2)
    fig, ax = plt.subplots(1, 3)
    ax[0].scatter(x=range(len(scores[0])),
                  y=scores[0],
                  color='b',
                  label='mae'
                  )
    ax[1].scatter(x=range(len(scores[1])),
                  y=scores[1],
                  color='g',
                  label='mse'
                  )
    ax[2].scatter(x=range(len(scores[2])),
                  y=scores[2],
                  color='r',
                  label='r2'
                  )
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    fig.suptitle(f'Scores')
    fig.tight_layout()
    return fig, ax


class RMS:
    def __init__(
            self,
            exp_name,
    ):
        self.exp_name = exp_name

        print(f'\nLoaded {exp_name.upper().replace(" ", "_")} RMS Data')

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
        # get path to home directory
        dirs = Path.cwd().parts
        search_str = 'tomje'
        i = dirs.index(search_str)
        home_dir = os.path.join(*dirs[:i + 1])

        # get file name of .csv file if created
        file_name = f'RMS_{self.exp_name.upper().replace(" ", "_")}.csv'

        # join path of home dir, data folder, and file name for reading
        path = Path(os.path.join(home_dir, DATA_DIR, file_name))

        try:
            # try to read in .csv file
            self.data = pd.read_csv(path)
        except FileNotFoundError:
            print(f'RMS Data File Not Found for {self.exp_name}')
            # if no data file process data and save
            self.data = self._process_exp(path)


class AutoEncoder_Model(Model):
    def __init__(
            self,
            data,
            train_slice = (0,100),
            random_state = None,
    ):
        print('AutoEncoder Model')
        super(AutoEncoder_Model, self).__init__()
        self.data = data
        self._train_slice = np.s_[train_slice[0]:train_slice[1]]
        self.random_state = random_state

        self.pre_process()
        self.encoder = self.get_encoder(self.n_inputs, 10)
        self.decoder = self.get_decoder(self.n_inputs, 10)

    def pre_process(
            self,
            val_frac: float = 0.1,
    ):
        print(f'\tPre-Processing Data:')

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

        self.n_inputs = x_train.shape[1]
        self.train_data = x_train
        self.val_data = x_test

    @staticmethod
    def get_encoder(n_inputs, n_bottleneck):
        encoder = Sequential(name='Encoder')
        encoder.add(Input(shape=(n_inputs, )))
        encoder.add(Dense(64, activation='relu'))
        encoder.add(BatchNormalization())

        encoder.add(Dense(64, activation='relu'))
        encoder.add(BatchNormalization())

        encoder.add(Dense(n_bottleneck, activation='relu'))
        return encoder

    @staticmethod
    def get_decoder(n_inputs, n_bottleneck):
        decoder = Sequential(name='Decoder')
        decoder.add(Input(shape=(n_bottleneck, )))
        decoder.add(Dense(64, activation='relu'))
        decoder.add(BatchNormalization())

        decoder.add(Dense(64, activation='relu'))
        decoder.add(BatchNormalization())

        decoder.add(Dense(n_inputs, activation='relu'))
        return decoder

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

if __name__ == '__main__':

    exps = ['Test 5', 'Test 7', 'Test 8', 'Test 9']
    # exps = ['Test 5']

    rms= {}
    for test in exps:

        rms[test] = RMS(test)

    rms['Test 5'].data.drop(['23', '24'], axis=1, inplace=True)

    for test in exps:

        print(f'\n{test.upper().replace(" ", "_")}')

        t = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        base_folder = os.path.abspath(rf'{pathlib.Path.home()}/ml/Tensorboard')
        run_name = os.path.join(base_folder,
                                r'AUTOEN',
                                rf'{test.upper().replace(" ", "_")}-AE-{t}')

        autoe = AutoEncoder_Model(data=rms[test].data,
                                  random_state=1,
                                  train_slice=(0, 100),
                                  )
        autoe.compile(optimizer='adam',
                      loss='mse',
                      metrics = ('MSE',
                                 'MAE',
                                 KerasRegressor.r_squared,
                                 ),
                      )
        history = autoe.fit(autoe.train_data, autoe.train_data,
                            epochs=500,
                            batch_size=10,
                            verbose=0,
                            validation_data=(autoe.val_data, autoe.val_data),
                            callbacks=[
                                tfa.callbacks.TQDMProgressBar(
                                    show_epoch_progress=False),
                                tf.keras.callbacks.TensorBoard(
                                    log_dir=run_name)
                            ],
                            )

        # calc metrics
        print(f'\nValidation Scores:')
        autoe_val_pred, autoe_val_scores = pred_and_score(autoe,
                                                          autoe.val_data)

        # plot hist of loss values from training
        p = autoe.predict(autoe.train_data, verbose=0)
        train_mse = mean_squared_error(autoe.train_data, p,
                                        multioutput='raw_values')

        train_mae =mean_absolute_error(autoe.train_data, p,
                                       multioutput='raw_values')

        train_r2 =r2_score(autoe.train_data, p,
                           multioutput='raw_values')

        # fig, ax = plt.subplots()
        # ax.hist(train_mse, bins=50)
        # ax.set_xlabel('Error')
        # ax.set_ylabel('No of Occurences')
        # ax.set_title(f'{test} Histogram of Training Dataset Prediction Error')

        # calc thresholds from each metric based on mean and std
        thr_mse = np.mean(train_mse) + np.std(train_mse)
        thr_mae = np.mean(train_mae) + np.std(train_mae)
        thr_r2 = np.mean(train_r2) - np.std(train_r2)

        print(f'\nThreshold:')
        print(f'\tMSE = {thr_mse}')
        print(f'\tMAE = {thr_mae}')
        print(f'\tR2 = {thr_r2}')

        # plot loss
        # fig, ax = plt.subplots()
        # ax.plot(history.history['loss'], label='train')
        # ax.plot(history.history['val_loss'], label='test')
        # ax.set_title(f'{test} - Model Loss')
        # ax.legend()

        # prediction plot of cut 8
        # fig, ax = pred_plot(autoe, autoe.val_data, autoe_val_pred, 8)
        # ax.set_title(f'{test} Validation Predictions - {ax.get_title()}')

        # Prediction on whole dataset
        print(f'\nDataset Predictions:')
        # first scale whole dataset
        d = autoe.scaler.transform(autoe.data.values.T)
        autoe_pred, autoe_scores = pred_and_score(autoe,
                                                  d)

        # plot scatter scores with the threshold from training scores
        # TODO need to check anomalies within RMS data especially test 5 and 9
        fig, ax = scatter_scores(scores=autoe_scores)
        fig.suptitle(f'{test} - {fig._suptitle.get_text()}')
        ax[0].axhline(thr_mae, color='k', ls='--')
        ax[1].axhline(thr_mse, color='k', ls='--')
        ax[2].axhline(thr_r2, color='k', ls='--')


        def plot_to_tf_im(figure):
            """Converts the matplotlib plot specified by 'figure' to a PNG
            image and
            returns it. The supplied figure is closed and inaccessible after
            this call."""
            # Save the plot to a PNG in memory.
            buf = io.BytesIO()
            figure.savefig(buf, format='png')
            buf.seek(0)
            # Convert PNG buffer to TF image
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            # Add the batch dimension
            image = tf.expand_dims(image, 0)
            return image

        tb_writer = tf.summary.create_file_writer(run_name)
        with tb_writer.as_default():
            tf.summary.image("Scatter Predicitons", plot_to_tf_im(fig), step=0)

        # prediction plot of cut 110
        # fig, ax = pred_plot(autoe, d, autoe_pred, 110)
        # ax.set_title(f'{test} UNSEEN DATA - {ax.get_title()}')

    plt.show(block=False)
