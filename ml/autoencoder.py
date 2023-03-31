"""
@File    :   autoencoder.py
@Author  :   Tom Jessel
@Contact :   jesselt@cardiff.ac.uk

@Creation Time    @Author    @Version    @Description
------------      -------    --------    -----------
27/02/2023 11:23   tomhj      1.0         N/A
"""
# %%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from textwrap import dedent
from typing import Any, Union
import tensorflow as tf
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import transforms
import mplcursors
import pandas as pd
from collections import defaultdict
import numpy as np
from tqdm.auto import tqdm
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input, Dense, BatchNormalization, Lambda
from keras.models import Model
# from keras.regularizers import l1, l2
import tensorboard.plugins.hparams.api as hp
import tensorflow_addons as tfa
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import time
from scikeras.wrappers import KerasRegressor, BaseWrapper
from keras import backend as K

import resources

DATA_DIR = Path.home().joinpath(r'Testing/RMS')
TB_DIR = Path.home().joinpath(r'ml/Tensorboard')


def _mp_rms_process(fno: int):
    """
    Multiprocessing function to compute RMS of AE data.

    Calc averaged rms of AE data for each cut and return as array.
    Average size is 100000. RMS calcualted using rolling window of 500000.

    Args:
        fno (int): File number of AE data to calc RMS for.
    
    Returns:
        np.array: Array of RMS values for each cut.
    """
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


def mp_get_rms(fnos: list[int]):
    """
    Master multiprocessing function to compute RMS of AE data.

    Args:
        fnos (list[int]): List of file numbers to calc RMS for.
    
    Returns:
        list: RMS values for each cut.
    """
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


class RMS:
    def __init__(
            self,
            exp_name,
    ):
        """
        RMS Data Object for AutoEncoder.

        Args:
            exp_name (str): Name of experiment to load data from.
        """
        self.exp_name = exp_name.upper().replace(" ", "_")

        print(f'\nLoaded {exp_name} RMS Data')

        # Read in data from file or compute
        self._get_data()

    def _process_exp(self, save_path: Path = None):
        """
        Process AE data and save to .csv file.

        Args:
            save_path (Path, optional): Path to save .csv file to.

        Returns:
            pd.DataFrame: Dataframe of RMS data.
        """

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
        """
        Find and read in data from .csv file or process data and save to .csv.
        """
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
        """
        AuotoEncoder class.

        Takes the rms AE data and pre-processes it for training. Then\
              initialises the model based on it.

        Args:
            rms_obj (RMS): RMS object containing the AE data to use.
            tb (bool, optional): Whether to use tensorboard. Defaults to True.
            tb_logdir (str, optional): Name of tensorboard log directory.
            params (dict, optional): Dictionary of parameters to pass to\
                initialise_model.
            train_slice (tuple, optional): Tuple of start and end index\
                for training data. Defaults to (0, 100).
            random_state (int, optional): Random state for reproducibility.
        """
        self.RMS = rms_obj
        self.data = rms_obj.data
        self._train_slice = np.s_[train_slice[0]:train_slice[1]]
        self.random_state = random_state
        self._tb = tb
        self._tb_logdir = TB_DIR.joinpath('AUTOE', tb_logdir)
        self._thres = None

        # attributes to be set later for predictionsi and scores on whole data
        self.pred = None
        self.scores = None

        if params is None:
            params = {}
        self.params = params

        # pre-process data and get the train and val indices
        self.pre_process()

        self.model = self.initialise_model(**self.params)
        print(f'\n{self.run_name}')
        # self.model.initialize(X=self.train_data)
        # self.model.model_.summary()
        print()

    def pre_process(
        self,
        val_frac: float = 0.1,
    ):
        """
        Pre-process the data for training and fit scaler.

        First splits the data based on _train_slice and then splits the data \
        into training and validation sets based on val_frac and random_state.\
        Then fits the scaler to the training data and transforms both the \
        training and validation data.

        Args:
            val_frac (float, optional): Fraction of data to use for the\
                validation set. Defaults to 0.1.
        
        """
        print('Pre-Processing Data:')

        print(f'\tTraining Data: {self._train_slice}')
        data = self.data.values.T

        # get indices of training data
        ind_tr = np.arange(len(data))[self._train_slice]

        self.scaler = MinMaxScaler(feature_range=(0, 1))

        if self.random_state is None:
            ind_tr, ind_val = train_test_split(
                ind_tr,
                test_size=val_frac,
            )
        else:
            ind_tr, ind_val = train_test_split(
                ind_tr,
                test_size=val_frac,
                random_state=self.random_state,
            )

        print(f'\tInput train shape: {data[ind_tr].shape}')
        print(f'\tInput val shape: {data[ind_val].shape}')

        # fit scaler to training data and transform all data
        self.scaler.fit(data[ind_tr])
        self.data = self.scaler.transform(data)

        self._n_inputs = data[ind_tr].shape[1]
        self._ind_tr = ind_tr
        self._ind_val = ind_val

    @property
    def train_data(self):
        """
        Return the training data.
        """
        try:
            return self.data[self._ind_tr]
        except AttributeError:
            print('Data not pre-processed yet!')
            return None
    
    @property
    def val_data(self):
        """
        Return the validation data.
        """
        try:
            return self.data[self._ind_val]
        except AttributeError:
            print('Data not pre-processed yet!')
            return None

    @property
    def thres(self):
        """
        Return the threshold values for the scores. Calculated if not already.
        """
        if self._thres is None:
            self._thres = self._get_cutoffs()
        return self._thres

    def _get_cutoffs(self):
        """
        Method to get the cutoff values from the training slice scores

        Returns:
            cutoffs (dict): Dictionary of cutoff values for each score.
        """
        try:
            sc = {k: v[self._train_slice] for k, v in self.scores.items()}
        except AttributeError:
            print('Scores not calculated yet! Score then re-run.')
            return None

        cutoffs = {}
        print('\nCutoffs:')
        for key, score in sc.items():
            # check if the scores should be trying to inc or dec
            if key == 'r2':
                cutoffs[key] = np.mean(score) - np.std(score)
            else:
                cutoffs[key] = np.mean(score) + np.std(score)
            print(f'\t{key.upper()} cutoff: {cutoffs[key]:.5f}')
        return cutoffs

    @staticmethod
    def _get_autoencoder(
        n_inputs: int,
        n_bottleneck: int,
        n_size: list[int],
        activation: str,
        activity_regularizer,
    ) -> Model:
        """
        Create a Keras autoencoder model with the given parameters.

        Args:
            n_inputs (int): Number of inputs to the model.
            n_bottleneck (int): Number of nodes in the bottleneck layer.
            n_size (list[int]): List of integers for the number of nodes in \
                the encoder (and decoder but reversed)
            activation (str): Activation function to use.
            activity_regularizer: Activity regulariser to use in encoder.

        Returns:
            Model: Keras model of the autoencoder.
        """
        def get_encoder(n_inputs, n_bottleneck, n_size, activation):
            encoder_in = Input(shape=(n_inputs, ))
            e = encoder_in

            for dim in n_size:
                e = Dense(dim, activation=activation)(e)
                e = BatchNormalization()(e)

            encoder_out = Dense(n_bottleneck,
                                activation='relu',
                                activity_regularizer=activity_regularizer)(e)
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
        batch_size: int = 10,
        loss: str = 'mse',
        metrics: list[str] = ['MSE',
                              'MAE',
                              KerasRegressor.r_squared
                              ],
        optimizer='adam',
        activity_regularizer=None,
        verbose: int = 1,
        callbacks: list[Any] = None,
    ):
        """
        Initialise the model with the given parameters and callbacks.

        Creates an AutoEncoder model within a sickeras basewrapper, based on
        the inputted parameters. Also creates a unique run name for logging to
        tensorboard if chosen.

        Args:
            n_bottleneck (int, optional): Number of nodes in the bottleneck\
                  layer.
            n_size (list, optional): List of nodes in the encoder\
                  (decoder reversed).
            activation (str, optional): Activation function to use.
            epochs (int, optional): Number of epochs to train for.
            batch_size (int, optional): Batch size to use.
            loss (str, optional): Loss function to use for each node.
            metrics (list[str], optional): List of metrics to calc for.
            optimizer (str, optional): Optimizer to use.
            verbose (int, optional): Verbosity of the model.
            callbacks (list[Any], optional): List of callbacks to use.
        """

        self._tb_logdir = TB_DIR.joinpath('AUTOE', self._tb_logdir)
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
                
                t_allow = (int, float, str, bool)
                types = {k: isinstance(val, t_allow)
                         for k, val in hp_params.items()}

                for k in types.keys():
                    if types[k] is False:
                        old = hp_params.pop(k)
                        if k == 'n_size':
                            hp_params[k] = str(layers)
                        elif k == 'activity_regularizer':
                            if old is not None:
                                [[key, value]] = old.get_config().items()
                                hp_params[k] = f'{key}: {value:.3g}'
                            else:
                                hp_params[k] = str(old)
                        else:
                            hp_params[k] = str(old)
                
                hp.hparams(
                    hp_params,
                    trial_id=f'{self._tb_logdir.joinpath(self.run_name)}'
                )

        model = BaseWrapper(
            model=self._get_autoencoder,
            model__n_inputs=self._n_inputs,
            model__n_bottleneck=n_bottleneck,
            model__n_size=n_size,
            model__activation=activation,
            model__activity_regularizer=activity_regularizer,
            epochs=epochs,
            batch_size=batch_size,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            verbose=verbose,
            callbacks=callbacks,
        )

        return model

    def fit(self, x, val_data: np.ndarray = None, **kwargs):
        """
        Fit the model to the inputted data.

        Passthrough func to fit the model to the inputted data. Will also track
        use validation data if provided.

        Args:
            x (np.ndarray): Input data to fit the model to.
            val_data (np.ndarray, optional): Validation data to use.\
                Defaults to None.
            **kwargs: Additional arguments to pass to the model.fit method.
        """
        if val_data is not None:
            self.model.fit(
                X=x,
                y=x,
                validation_data=(val_data, val_data),
                **kwargs
            )
        else:
            self.model.fit(
                X=x,
                y=x,
                **kwargs
            )

    def score(
            self,
            label: str = None,
            x: np.ndarray = None,
            tb: bool = True,
            print_score: bool = True,
    ) -> tuple[tuple[np.ndarray, np.ndarray], dict]:
        """
        Score the model on the inputted data.

        Predicts the model on the inputted data and calculates the scores,
        as well as doing it for all the initiliased dataset.

        Args:
            label (str, optional): Label of the data to score the model on \
                (train, val, dataset).
            x (np.ndarray): Input data to score the model on.
            tb (bool, optional): Log to tensorboard. Defaults to True.
            print_score (bool, optional): Print the scores. Defaults to True.
        
        Returns:
            tuple[tuple[np.ndarray, np.ndarray], dict]: A tuple (input,
              prediction) and a dictionary of scores.

        """

        label_hash = {
            'train': self._ind_tr,
            'val': self._ind_val,
            'dataset': np.arange(self.data.shape[0]),
        }

        if x is None and label is None:
            raise ValueError('Must provide either x or label for scoring.')
        elif x is not None and label is not None:
            raise ValueError('Cannot provide both x and label for scoring.')

        if self.pred is None:
            pred = self.model.predict(self.data, verbose=0)
            self.pred = pred

        if self.scores is None:
            mae = mean_absolute_error(self.data.T,
                                      self.pred.T,
                                      multioutput='raw_values',
                                      )
            mse = mean_squared_error(self.data.T,
                                     self.pred.T,
                                     multioutput='raw_values',
                                     )
            r2 = r2_score(self.data.T,
                          self.pred.T,
                          multioutput='raw_values',
                          )
            self.scores = {'mae': mae, 'mse': mse, 'r2': r2}

        if x is not None:
            pred = self.model.predict(x, verbose=0)
            mae = mean_absolute_error(x.T, pred.T, multioutput='raw_values')
            mse = mean_squared_error(x.T, pred.T, multioutput='raw_values')
            r2 = r2_score(x.T, pred.T, multioutput='raw_values')
            scores = {'mae': mae, 'mse': mse, 'r2': r2}

        if label is not None and label in label_hash.keys():
            x = self.data[label_hash[label]]
            pred = self.pred[label_hash[label]]
            scores = {k: v[label_hash[label]] for k, v in self.scores.items()}
        elif label is not None and label not in label_hash.keys():
            raise KeyError(f'Label {label} not found in label_hash.')

        if self._tb and tb:
            tb_writer = tf.summary.create_file_writer(
                f'{self._tb_logdir.joinpath(self.run_name)}')

            md_scores = dedent(f'''
                    ### Scores - Validation Data

                     | MAE | MSE |  R2  |
                     | ---- | ---- | ---- |
                     | {np.mean(scores['mae']):.5f} |\
                         {np.mean(scores['mse']):.5f} |\
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

        if print_score:
            if label is not None:
                print(f'\n{label.capitalize()} Scores:')
            else:
                print('\nScores:')
            print(f'\tMAE: {np.mean(scores["mae"]):.5f}')
            print(f'\tMSE: {np.mean(scores["mse"]):.5f}')
            print(f'\tR2: {np.mean(scores["r2"]):.5f}')
        return (x, pred), scores

    def pred_plot(self, input: tuple, no: int):
        """
        Plot prediction vs real data for a given cut.

        Args:
            input (tuple): Input data array for plotting (real, pred).
            no (int): Cut number to plot.
        
        Returns:
            fig, ax: Matplotlib figure and axis.
        """
        pred_input = input[0][no, :].reshape(-1, self._n_inputs)
        x_pred = input[1][no, :].reshape(-1, self._n_inputs)

        pred_input = self.scaler.inverse_transform(pred_input)
        x_pred = self.scaler.inverse_transform(x_pred)

        mse = mean_squared_error(pred_input, x_pred)
        mae = mean_absolute_error(pred_input, x_pred)

        fig, ax = plt.subplots()
        ax.plot(pred_input.T, label='Real')
        ax.plot(x_pred.T, label='Predicition')
        ax.legend()
        ax.set_title(f'MAE: {mae:.4f} MSE: {mse:.4f}')
        return fig, ax

    def loss_plot(self):
        """
        Plot the loss and validation loss over epochs.

        Returns:
            fig, ax: Matplotlib figure and axis.
        """
        if hasattr(self.model, 'history_') is False:
            raise ValueError('Model has not been fit yet.')

        fig, ax = plt.subplots()
        ax.plot(self.model.history_['loss'], label='loss')
        ax.plot(self.model.history_['val_loss'], label='val_loss')
        ax.legend()
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        return fig, ax

    def hist_scores(self, metrics: list = None):
        """
        Plot histograms of the scores from the training and validation data.

        Args:
            metrics (list): List of metrics to plot. Default is all.
        
        Returns:
            fig, ax: Matplotlib figure and axis.
        """
        if self.scores is None:
            self.score('dataset', print_score=False)

        if metrics is None:
            metrics = ['mae', 'mse', 'r2']

        fig, ax = plt.subplots(len(metrics), 1, squeeze=False)
        for i, metric in enumerate(metrics):
            ax[i, 0].hist(self.scores[metric][self._train_slice],
                          bins=40, label=metric)
            ax[i, 0].legend()
            ax[i, 0].set_xlabel(f'{metric.upper()} score')
            ax[i, 0].set_ylabel('Frequency')
        return fig, ax


class _VariationalAutoEncoder(Model):
    def __init__(self, input_dim, latent_dim, n_size):
        super().__init__()
        # TODO change to work with n_size to allow for dynamic layers
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_size = n_size
        self.encoder = self.get_encoder(input_dim, latent_dim, n_size)
        self.decoder = self.get_decoder(input_dim, latent_dim, n_size)

    def get_encoder(self, input_dim, latent_dim, n_size):
        inputs = Input(shape=(input_dim,), name='encoder_input')
        e = inputs

        for dim in n_size:
            e = Dense(dim, activation='relu')(e)
            e = BatchNormalization()(e)

        z_mean = Dense(latent_dim, name='z_mean')(e)
        z_log_sigma = Dense(latent_dim, name='z_log_sigma')(e)

        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                      mean=0., stddev=0.1)
            return z_mean + K.exp(z_log_sigma) * epsilon
        
        z = Lambda(sampling)([z_mean, z_log_sigma])

        # encoder mapping inputs to rthe latent space
        encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
        return encoder

    def get_decoder(self, input_dim, latent_dim, n_size):
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        d = latent_inputs

        for dim in n_size[::-1]:
            d = Dense(dim, activation='relu')(d)
            d = BatchNormalization()(d)

        outputs = Dense(input_dim, activation='sigmoid')(d)

        decoder = Model(latent_inputs, outputs, name='decoder')
        return decoder

    def call(self, inputs):
        out_encoder = self.encoder(inputs)
        z_mean, z_log_sigma, z = out_encoder
        out_decoder = self.decoder(z)

        reconstruction_loss = tf.keras.metrics.mean_squared_error(
            inputs,
            out_decoder,
        )
        reconstruction_loss *= self.input_dim
        kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.add_loss(vae_loss)
        return out_decoder


class VariationalAutoEncoder(AutoEncoder):
    def __init__(
        self,
        rms_obj: RMS,
        tb: bool = True,
        tb_logdir: str = '',
        params: dict = None,
        train_slice=(0, 100),
        random_state=None,
    ):
        super().__init__(rms_obj,
                         tb,
                         tb_logdir,
                         params,
                         train_slice,
                         random_state,
                         )

    def initialise_model(
            self,
            latent_dim: int = 2,
            n_size: list = [64, 64],
            optimizer: str = 'adam',
            epochs: int = 100,
            batch_size: int = 10,
            metrics: list = ['MSE', 'MAE', KerasRegressor.r_squared],
            verbose: int = 1,
            callbacks: list = None,
    ):
        
        self._tb_logdir = TB_DIR.joinpath('VAE', self._tb_logdir)

        layers = n_size + [latent_dim] + n_size[::-1]
        t = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        self.run_name = f'VAE-{self.RMS.exp_name}-E-{epochs}-L-{layers}-T-{t}'

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

                t_allow = (int, float, str, bool)
                types = {k: isinstance(val, t_allow)
                         for k, val in hp_params.items()}

                for k in types.keys():
                    if types[k] is False:
                        old = hp_params.pop(k)
                        if k == 'n_size':
                            hp_params[k] = str(layers)
                        elif k == 'activity_regularizer':
                            if old is not None:
                                [[key, value]] = old.get_config().items()
                                hp_params[k] = f'{key}: {value:.3g}'
                            else:
                                hp_params[k] = str(old)
                        else:
                            hp_params[k] = str(old)
                
                hp.hparams(
                    hp_params,
                    trial_id=f'{self._tb_logdir.joinpath(self.run_name)}'
                )

        model = BaseWrapper(
            _VariationalAutoEncoder(
                input_dim=self._n_inputs,
                latent_dim=latent_dim,
                n_size=n_size,
            ),
            optimizer=optimizer,
            epochs=epochs,
            batch_size=batch_size,
            metrics=metrics,
            verbose=verbose,
            callbacks=callbacks,
        )
        
        return model

    def generate(
            self,
            z: Union[list, np.ndarray] = None,
            plot_fig: bool = True,
    ):
        """
        Generate a new input from the latent space.

        Create a new signal from a point within the latent space [x, y]. Also
        will plot the generated signal if plot_fig is True.

        Args:
            z (Union[list, np.ndarray], optional): Point in latent space to
                generate signal from. Defaults to None.
            plot_fig (bool, optional): Plot the generated signal. Defaults to
                True.
        
        Returns:
            gen (np.ndarray): Generated signal.
            fig (plt.figure): Figure object. If plot_fig is True.
            ax (plt.axes): Axes object. If plot_fig is True.
        """
        if z is None:
            raise ValueError('Z input must be provided for decoder. [x, y]')
        gen = self.model.model_.decoder.predict([z], verbose=0)
        gen = self.scaler.inverse_transform(gen)

        if plot_fig:
            fig, ax = plt.subplots()
            ax.plot(gen.T)
            ax.set_title(f'Generated input from latent space - Z({z})')
            return gen, fig, ax
        return gen
    
    def plot_latent_space(self):
        """
        Plot the latent space on all the data.

        Plots the z_mean of the encoded data on a scatter plot. The colour
        represents the cut number of the point.

        Returns:
            fig (plt.figure): Figure object.
            ax (plt.axes): Axes object.
        """
        encoded = self.model.model_.encoder.predict(self.data, verbose=0)
        z_mean, z_log_sigma, _ = encoded
        
        fig, ax = plt.subplots()
        s = ax.scatter(z_mean[:, 0], z_mean[:, 1],
                       c=range(len(z_mean[:, 0])),
                       cmap=plt.get_cmap('jet')
                       )
        ax.set_title('Latent space')
        cbar = plt.colorbar(s)
        cbar.set_label('Cut No.')
        return fig, ax


if __name__ == '__main__':

    # exps = ['Test 5', 'Test 7', 'Test 8', 'Test 9']
    exps = ['Test 8']

    rms = {}
    for test in exps:
        rms[test] = RMS(test)
    try:
        rms['Test 5'].data.drop(['23', '24'], axis=1, inplace=True)
    except KeyError:
        pass

    print()

    for test in exps:

        # autoe = AutoEncoder(rms[test],
        #                     random_state=1,
        #                     train_slice=(0, 100),
        #                     tb=False,
        #                     tb_logdir=rms[test].exp_name + '/neurons',
        #                     params={'n_bottleneck': 10,
        #                             'n_size': [64, 64],
        #                             'epochs': 500,
        #                             'loss': 'mse',
        #                             'batch_size': 10,
        #                             # 'activity_regularizer': None,
        #                             }
        #                     )

        # autoe.fit(
        #     x=autoe.train_data,
        #     val_data=autoe.val_data,
        #     verbose=0,
        # )

        vae = VariationalAutoEncoder(rms[test],
                                     tb=False,
                                     tb_logdir=rms[test].exp_name,
                                     train_slice=(0, 100),
                                     random_state=1,
                                     params={'latent_dim': 2,
                                             'n_size': [64, 64],
                                             'epochs': 500,
                                             'batch_size': 10,
                                             }
                                     )

        vae.fit(x=vae.train_data,
                val_data=vae.val_data,
                verbose=0,
                use_multiprocessing=True,
                )

        # %% PLOT LOSS
        # ---------------------------------------------------------------------
        vae.loss_plot()

        # %% MODEL SUMMARY
        # ---------------------------------------------------------------------
        vae.model.model_.summary()
        vae.model.model_.encoder.summary()
        vae.model.model_.decoder.summary()

        # %% SCORE THE MODEL ON TRAINING, VALIDATION AND ALL DATA
        # ---------------------------------------------------------------------
        pred_tr, scores_tr = vae.score('train')
        pred_val, scores_val = vae.score('val')
        pred_data, scores_data = vae.score('dataset')

        # %% HISTOGRAM OF SCORES
        # ---------------------------------------------------------------------
        fig, ax = vae.hist_scores(['mse'])

        # %% PLOT PREDICTIONS
        # ---------------------------------------------------------------------
        fig, ax = vae.pred_plot(pred_tr, 0)
        ax.set_title(f'{vae.RMS.exp_name} Training Data - {ax.get_title()}')
        fig, ax = vae.pred_plot(pred_val, 0)
        ax.set_title(f'{vae.RMS.exp_name} Val Data - {ax.get_title()}')

        # %% CALC CUTOFFS
        # ---------------------------------------------------------------------
        def calc_cutoff(scores):

            sc = defaultdict(list)

            for score in scores:
                for key, score in score.items():
                    sc[key].extend(score)

            cutoffs = {}
            for key, score in sc.items():
                # check if the scores should be trying to inc or dec
                if key == 'r2':
                    cutoffs[key] = np.mean(score) - np.std(score)
                else:
                    cutoffs[key] = np.mean(score) + np.std(score)
                print(f'\t{key.upper()} cutoff: {cutoffs[key]:.5f}')
            return cutoffs
        
        print('\nCutoffs:')
        thresholds = calc_cutoff([scores_tr, scores_val])

        # %% GENERATE NEW DATA
        # ---------------------------------------------------------------------
        _, fig, ax = vae.generate(z=[-2, 2])

        # %% PLOT LATENT SPACE
        # ---------------------------------------------------------------------
        fig, ax = vae.plot_latent_space()
