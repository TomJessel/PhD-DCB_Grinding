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
import matplotlib.pyplot as plt
import mplcursors
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input, Dense, BatchNormalization, Lambda, Dropout
from keras.layers import LSTM, RepeatVector, TimeDistributed
from keras.models import Model
import tensorboard.plugins.hparams.api as hp
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import PurePosixPath as Path
import time
from scikeras.wrappers import KerasRegressor, BaseWrapper
from keras import backend as K
import dill as pickle

import resources

platform = os.name
if platform == 'nt':
    onedrive = Path(r'C:\Users\tomje\OneDrive - Cardiff University')
    onedrive = onedrive.joinpath('Documents', 'PHD', 'AE')
    DATA_DIR = onedrive.joinpath('Testing', 'RMS')
    TB_DIR = onedrive.joinpath('Tensorboard')
elif platform == 'posix':
    onedrive = Path(r'/mnt/c/Users/tomje/OneDrive - Cardiff University')
    onedrive = onedrive.joinpath('Documents', 'PHD', 'AE')
    DATA_DIR = onedrive.joinpath('Testing', 'RMS')
    TB_DIR = onedrive.joinpath('Tensorboard')


def load_model(filepath):

    file_loc = Path(filepath)

    # check path exists
    if os.path.exists(file_loc) is False:
        try:
            file_loc = TB_DIR.joinpath('AUTOE', file_loc)
            if os.path.exists(file_loc) is False:
                raise FileNotFoundError()
        except FileNotFoundError:
            raise FileNotFoundError(f'{file_loc} does not exist.')

    # check if path is file or directory
    if os.path.isdir(file_loc):
        pkl_files = []
        for f in os.listdir(file_loc):
            if f.endswith('.pickle'):
                pkl_files.append(f)
        if len(pkl_files) == 1:
            file_loc = file_loc.joinpath(pkl_files[0])
        elif len(pkl_files) > 1:
            raise ValueError(f'Multiple pickle files in {file_loc}.')
        else:
            raise FileNotFoundError(f'{file_loc} does not exist.')
    elif os.path.isfile(file_loc):
        file_loc = file_loc
    else:
        raise FileNotFoundError(f'{file_loc} does not exist.')

    with open(file_loc, 'rb') as f:
        model = pickle.load(f)
        print('Model loaded:')
        print(f'\tLoad Loc: {file_loc}')
    return model


class Custom_TB_Callback(tf.keras.callbacks.Callback):
    # custom tb callback to allow for pickle saving
    def __init__(self, logdir, run_name):
        super().__init__()
        self.logdir = logdir
        self.run_name = run_name

    def on_epoch_end(self, epoch, logs=None):
        train_writer = tf.summary.create_file_writer(
            f'{self.logdir.joinpath(self.run_name)}/train'
        )
        val_writer = tf.summary.create_file_writer(
            f'{self.logdir.joinpath(self.run_name)}/validation'
        )

        for key, value in logs.items():
            if 'val_' in key:
                key = key.replace('val_', '')
                with train_writer.as_default():
                    tf.summary.scalar(
                        f'epoch_{key}',
                        value,
                        step=epoch,
                    )
            elif 'val_' not in key:
                with val_writer.as_default():
                    tf.summary.scalar(
                        f'epoch_{key}',
                        value,
                        step=epoch,
                    )
    

class AutoEncoder():
    def __init__(
        self,
        rms_obj: Any,
        data: pd.DataFrame,
        tb: bool = True,
        tb_logdir: str = '',
        params: dict = None,
        train_slice=(0, 100),
        random_state=None,
        val_frac: float = 0.1,
    ):
        """
        AuotoEncoder class.

        Takes the rms AE data and pre-processes it for training. Then\
              initialises the model based on it.

        Args:
            rms_obj (Any): RMS object with holds test information. Must have\
                exp_name attribute.
            data (pd.DataFrame): Dataframe of rms data. Columns are the \
                denote seperate cuts.
            tb (bool, optional): Whether to use tensorboard. Defaults to True.
            tb_logdir (str, optional): Name of tensorboard log directory.
            params (dict, optional): Dictionary of parameters to pass to\
                initialise_model.
            train_slice (tuple, optional): Tuple of start and end index\
                for training data. Defaults to (0, 100).
            random_state (int, optional): Random state for reproducibility.
        """
        self.RMS = rms_obj
        self._data = data
        self._train_slice = np.s_[train_slice[0]:train_slice[1]]
        self.random_state = random_state
        self._tb = tb
        self._tb_logdir = TB_DIR.joinpath('AUTOE', tb_logdir)
        self._thres = None

        # attributes to be set later for predictions and scores on whole data
        self.pred = None
        self.scores = None

        if params is None:
            params = {}
        self.params = params

        # pre-process data and get the train and val indices
        self.pre_process(val_frac=val_frac)

        self.model = self.initialise_model(**self.params)
        print(f'\n{self.run_name}')
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
        self.scaler.fit(data[ind_tr].reshape(-1, 1))
        self._data = self.scaler.transform(data.reshape(-1, 1)).reshape(
            data.shape[0], data.shape[1]
        )

        self._n_inputs = data[ind_tr].shape[1]
        self._ind_tr = ind_tr
        self._ind_val = ind_val

    @property
    def data(self):
        """
        Return the data.
        """
        try:
            return self._data
        except AttributeError:
            print('Data not pre-processed yet!')
            return None

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

    @property
    def run_name(self):
        """
        Return the run name for the model. Set if not already.
        """
        if hasattr(self, '_run_name'):
            return self._run_name
        else:
            self._run_name = self.set_run_name()

    def set_run_name(self, append: str = None):
        """
        Initialise the run name for the model.

        Args:
            append (str, optional): String to append to the run name.\
                Defaults to None.
        
        Returns:
            str: Run name.
        """
        t = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        n_size = self.params['n_size']
        n_bottle = self.params['n_bottleneck']
        layers = n_size + [n_bottle] + n_size[::-1]
        rn = f'AUTOE-{self.RMS.exp_name.replace(" ", "_")}-' \
             f'E-{self.params["epochs"]}-L-{layers}-{t}'
        if append is not None:
            rn = f'{rn}-{append}'
        self._run_name = rn
        return self._run_name

    def _get_cutoffs(self):
        """
        Method to get the cutoff values from the training slice scores

        Returns:
            cutoffs (dict): Dictionary of cutoff values for each score.
        """

        def mad_based_outlier(points, thresh=1.8):
            if len(points.shape) == 1:
                points = points[:, None]
            median = np.median(points, axis=0)
            diff = np.sum((points - median)**2, axis=-1)
            diff = np.sqrt(diff)
            med_abs_deviation = np.median(diff)

            modified_z_score = 0.6745 * diff / med_abs_deviation

            return modified_z_score > thresh
 
        try:
            sc = {k: v[self._train_slice] for k, v in self.scores.items()}
            # sc = {k: v for k, v in self.scores.items()}
        except AttributeError:
            print('Scores not calculated yet! Score then re-run.')
            return None

        cutoffs = {}
        print('\nCutoffs:')
        for key, score in sc.items():
            # check if the scores should be trying to inc or dec
            out = score[mad_based_outlier(score)]
            if key == 'r2':
                try:
                    cutoffs[key] = np.max(out)
                except ValueError:
                    print('std cutoff')
                    cutoffs[key] = np.median(score) - np.std(score)
            else:
                try:
                    cutoffs[key] = np.min(out)
                except ValueError:
                    print('std cutoff')
                    cutoffs[key] = np.median(score) + np.std(score)
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
                e = Dropout(0.1)(e)

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
                d = Dropout(0.1)(d)

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

        if callbacks is None:
            callbacks = []

        if self._tb:
            tb_callback = Custom_TB_Callback(
                logdir=self._tb_logdir,
                run_name=self.run_name,
            )
            callbacks.append(tb_callback)

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

    def fit(self, x, val_data: np.ndarray = None, verbose=1, **kwargs):
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
        print(f'Training model: {self.run_name}:')
        if val_data is not None:
            self.model.fit(
                X=x,
                y=x,
                validation_data=(val_data, val_data),
                verbose=verbose,
                **kwargs
            )
        else:
            self.model.fit(
                X=x,
                y=x,
                verbose=verbose,
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

        print('\nPredicting data:')
        if self.pred is None:
            pred = self.model.predict(self.data, verbose=1)
            self.pred = pred

        if self.scores is None:
            mae = mean_absolute_error(self.data,
                                      self.pred,
                                      multioutput='raw_values',
                                      )
            mse = mean_squared_error(self.data,
                                     self.pred,
                                     multioutput='raw_values',
                                     )
            r2 = r2_score(self.data,
                          self.pred,
                          multioutput='raw_values',
                          )
            self.scores = {'mae': mae, 'mse': mse, 'r2': r2}

        if x is not None:
            pred = self.model.predict(x, verbose=1)
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

    def pred_plot(self, no: int, input: tuple = None, plt_ax=None):
        """
        Plot prediction vs real data for a given cut.

        Args:
            input (tuple): Input data array for plotting (real, pred).
            no (int): Cut number to plot.
        
        Returns:
            fig, ax: Matplotlib figure and axis.
        """
        if input is not None:
            pred_input = input[0][no, :].reshape(-1, 1)
            x_pred = input[1][no, :].reshape(-1, 1)
        else:
            pred_input = self.data[no, :].reshape(-1, 1)
            x_pred = self.pred[no, :].reshape(-1, 1)

        pred_input = self.scaler.inverse_transform(pred_input)
        x_pred = self.scaler.inverse_transform(x_pred)

        mse = mean_squared_error(pred_input, x_pred)
        mae = mean_absolute_error(pred_input, x_pred)

        if plt_ax is None:
            fig, ax = plt.subplots()
        else:
            ax = plt_ax
        ax.plot(pred_input, label='Real')
        ax.plot(x_pred, label='Predicition')
        ax.legend()
        ax.set_title(f'MAE: {mae:.4f} MSE: {mse:.4f}')
        if plt_ax is None:
            return fig, ax
        else:
            return ax

    def loss_plot(self, plt_ax=None):
        """
        Plot the loss and validation loss over epochs.

        Returns:
            fig, ax: Matplotlib figure and axis.
        """
        if hasattr(self.model, 'history_') is False:
            raise ValueError('Model has not been fit yet.')

        if plt_ax is None:
            fig, ax = plt.subplots()
        else:
            ax = plt_ax

        ax.plot(self.model.history_['loss'], label='loss')
        ax.plot(self.model.history_['val_loss'], label='val_loss')
        ax.legend()
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        if plt_ax is None:
            return fig, ax
        else:
            return ax

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
            ax[i, 0].hist(self.scores[metric][self._ind_tr],
                          bins=50, label=metric)
            ax[i, 0].legend()
            ax[i, 0].set_xlabel(f'{metric.upper()} score')
            ax[i, 0].set_ylabel('Frequency')
        return fig, ax

    def scatter_scores(self, metrics: list = None, plt_ax=None):
        """
        Plot scatter plots of the scores from the training and validation data.

        Args:
            metrics (list): List of metrics to plot. Default is all.
        
        Returns:
            fig, ax: Matplotlib figure and axis.
        """

        def onclick(event):
            if event.dblclick:
                x = round(event.xdata)
                fig, ax = self.pred_plot(x)
                ax.set_title(f'Cut {x} {ax.get_title()}')
                plt.show()

        if self.scores is None:
            self.score('dataset', print_score=False)

        if metrics is None:
            metrics = self.scores.keys()

        if hasattr(self, "thres") is False:
            self.thres

        if plt_ax is None:
            fig, ax = plt.subplots(len(metrics), 1,
                                   figsize=(7.5, 5),
                                   sharex=True
                                   )
            try:
                ax = ax.ravel()
            except AttributeError:
                ax = [ax]
        else:
            ax = [plt_ax]

        for i, metric in enumerate(metrics):
            ax[i].axhline(self.thres[metric], color='r', linestyle='--')
            if metric == 'r2':
                cmap = ['b' if y > self.thres[metric] else 'r'
                        for y in self.scores[metric]]
            else:
                cmap = ['r' if y > self.thres[metric] else 'b'
                        for y in self.scores[metric]]
            # cmap = ['C0' for y in self.scores[metric]]

            ax[i].scatter(x=range(len(self.scores[metric])),
                          y=self.scores[metric],
                          s=2,
                          label=metric,
                          c=cmap
                          )
            # trans = transforms.blended_transform_factory(
            #     ax[i].get_yticklabels()[0].get_transform(), ax[i].transData)
            # ax[i].text(0, self.thres[metric], f'{self.thres[metric]:.3f}',
            #            color="red",
            #            transform=trans,
            #            ha="right",
            #            va="center"
            #            )
            ax[i].set_ylabel(f'{metric.upper()} score')
        ax[-1].set_xlabel('Cut number')
        
        try:
            fig.canvas.mpl_connect('button_press_event', onclick)
        except UnboundLocalError:
            return ax

        return fig, ax

    def save_model(self, folder_path=None) -> Union[Path, str]:
        if folder_path is None:
            folder_path = self._tb_logdir.joinpath(self.run_name)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        assert os.path.exists(folder_path), f'{folder_path} does not exist.'
            
        file_loc = folder_path.joinpath(f'{self.run_name}.pickle')

        with open(file_loc, 'wb') as f:
            pickle.dump(self, f)
            print('Model saved')
            print(f'\tSave Loc: {file_loc}')
        return file_loc

            
class _VariationalAutoEncoder(Model):
    def __init__(self, input_dim, latent_dim, n_size):
        super().__init__()
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
            e = Dropout(0.01)(e)

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
            d = Dropout(0.01)(d)

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
        rms_obj: Any,
        data: pd.DataFrame,
        tb: bool = True,
        tb_logdir: str = '',
        params: dict = None,
        train_slice=(0, 100),
        random_state=None,
        **kwargs,
    ):
        super().__init__(rms_obj,
                         data,
                         tb,
                         tb_logdir,
                         params,
                         train_slice,
                         random_state,
                         **kwargs,
                         )

    def set_run_name(self, append: str = None):
        """
        Initialise the run name for the model.

        Args:
            append (str, optional): String to append to the run name.\
                Defaults to None.
        
        Returns:
            str: Run name.
        """
        t = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        n_size = self.params['n_size']
        n_bottle = self.params['n_bottleneck']
        layers = n_size + [n_bottle] + n_size[::-1]
        rn = f'VAE-{self.RMS.exp_name.replace(" ", "_")}-' \
             f'E-{self.params["epochs"]}-L-{layers}-{t}'
        if append is not None:
            rn = f'{rn}-{append}'
        self._run_name = rn
        return self._run_name

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
        layers = n_size + ['Z'] + n_size[::-1]

        if callbacks is None:
            callbacks = []
        
        if self._tb:
            tb_callback = Custom_TB_Callback(
                logdir=self._tb_logdir,
                run_name=self.run_name,
            )
            callbacks.append(tb_callback)

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
            ax.set_title(f'Generated input - Z({z[0]:.3f}, {z[1]:.3f})')
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
        def onclick(event):
            if event.dblclick:
                if event.button == 1:
                    self.generate(z=[event.xdata, event.ydata])
                    plt.show()

        encoded = self.model.model_.encoder.predict(self.data, verbose=0)
        z_mean, z_log_sigma, _ = encoded
        
        labels = [f'Cut {i}' for i in range(len(self.data))]

        fig, ax = plt.subplots()
        s = ax.scatter(z_mean[:, 0], z_mean[:, 1],
                       c=range(len(z_mean[:, 0])),
                       cmap=plt.get_cmap('jet')
                       )
        ax.set_title('Latent space')
        cbar = plt.colorbar(s)
        cbar.set_label('Cut No.')
        mplcursors.cursor(ax, highlight=True, hover=2).connect(
            "add", lambda sel: sel.annotation.set_text(
                f'{labels[sel.index]}' +
                f' MSE: {self.scores["mse"][sel.index]:.5f}'
            )
        )
        fig.canvas.mpl_connect('button_press_event', onclick)
        return fig, ax


class LSTMAutoEncoder(AutoEncoder):
    def __init__(
        self,
        rms_obj: Any,
        data: pd.DataFrame,
        tb: bool = True,
        tb_logdir: str = '',
        params: dict = None,
        train_slice: tuple = (0, 100),
        random_state: int = None,
        **kwargs,
    ):
        self.seq_len = params.pop('seq_len', 50)
        super().__init__(rms_obj,
                         data,
                         tb,
                         tb_logdir,
                         params,
                         train_slice,
                         random_state,
                         **kwargs,
                         )

    def set_run_name(self, append: str = None):
        """
        Initialise the run name for the model.

        Args:
            append (str, optional): String to append to the run name.\
                Defaults to None.
        
        Returns:
            str: Run name.
        """
        t = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        n_size = self.params['n_size']
        n_bottle = self.params['n_bottleneck']
        layers = n_size + [n_bottle] + n_size[::-1]
        win_len = self.seq_len
        rn = f'LSTMAE-{self.RMS.exp_name.replace(" ", "_")}-' \
             f'WIN-{win_len}-E-{self.params["epochs"]}-L-{layers}-{t}'
        if append is not None:
            rn = f'{rn}-{append}'
        self._run_name = rn
        return self._run_name

    def pre_process(self, val_frac: float = 0.2):
        print('Pre-processing Data:')
        
        print('\tCombining RMS data...')
        org_sig_len = np.shape(self.data.values)[0]
        joined_rms = []
        for i in range(np.shape(self.data)[1]):
            joined_rms.extend(self.data.iloc[:, i].values.T)
        joined_rms = np.array(joined_rms).reshape(-1, 1)
        print(f'\tNumber of RMS samples: {np.shape(joined_rms)}')

        assert ~np.isnan(joined_rms).any(), 'NaN values in RMS data'

        print(f'\n\tTraining Data: {self._train_slice}')
        ind_tr = np.arange(len(joined_rms))
        ind_tr = ind_tr[(self._train_slice.start * org_sig_len):
                        (self._train_slice.stop * org_sig_len)]
        
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

        print(f'\tInput train shape: {joined_rms[ind_tr].shape}')
        print(f'\tInput val shape: {joined_rms[ind_val].shape}')

        # fit the data to all the training and val data
        # contanination cant be avoided as its time series data
        self.scaler.fit(joined_rms[np.concatenate([ind_tr, ind_val])])
        self.joined_data = self.scaler.transform(joined_rms)
        self._data = self.sequence_inputs(self.joined_data, self.seq_len)

        self._n_inputs = org_sig_len
        self._ind_tr = ind_tr
        self._ind_val = ind_val
        
    @staticmethod
    def sequence_inputs(data, seq_len):
        d = []
        for index in range(len(data) - seq_len + 1):
            d.append(data[index: (index + seq_len)])
        return np.stack(d)
    
    @property
    def seq_data(self):
        """
        Return the sequenced data.
        """
        try:
            return self._seq_data
        except AttributeError:
            print('Data not pre-processed yet!')
            return None
    
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

    def _get_cutoffs(self):
        """
        Method to get the cutoff values from the training slice scores

        Returns:
            cutoffs (dict): Dictionary of cutoff values for each score.
        """

        try:
            sc = {k: v[self._train_slice] for k, v in self.scores.items()}
            # sc = {k: v for k, v in self.scores.items()}
        except AttributeError:
            print('Scores not calculated yet! Score then re-run.')
            return None

        cutoffs = {}
        print('\nCutoffs:')
        for key, score in sc.items():
            # check if the scores should be trying to inc or dec
            if key == 'r2':
                cutoffs[key] = np.median(score) - np.std(score)
            else:
                cutoffs[key] = np.median(score) + np.std(score)
                # cutoffs[key] = np.percentile(score, 97)
            print(f'\t{key.upper()} cutoff: {cutoffs[key]:.5f}')
        return cutoffs

    @staticmethod
    def _get_autoencoder(
        seq_len: int,
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
        def get_encoder(seq_len, n_inputs, n_bottleneck, n_size, activation):
            encoder_in = Input(shape=(seq_len, 1))
            e = encoder_in

            for dim in n_size:
                e = LSTM(dim, return_sequences=True)(e)
                e = Dropout(0.1)(e)
                e = BatchNormalization()(e)

            encoder_out = LSTM(n_bottleneck,
                               return_sequences=False)(e)
            encoder = Model(encoder_in, encoder_out, name='Encoder')
            return encoder

        def get_decoder(seq_len, n_inputs, n_bottleneck, n_size, activation):
            decoder_in = Input(shape=(n_bottleneck,))
            d = decoder_in

            d = RepeatVector(seq_len)(d)

            for dim in n_size[::-1]:
                d = LSTM(dim, return_sequences=True)(d)
                d = Dropout(0.1)(d)
                d = BatchNormalization()(d)

            decoder_out = TimeDistributed(Dense(1))(d)
            decoder = Model(decoder_in, decoder_out, name='Decoder')
            return decoder

        encoder = get_encoder(seq_len,
                              n_inputs,
                              n_bottleneck,
                              n_size,
                              activation
                              )
        decoder = get_decoder(seq_len,
                              n_inputs,
                              n_bottleneck,
                              n_size,
                              activation
                              )

        autoencoder_in = Input(shape=(seq_len, 1), name='Input')
        encoded = encoder(autoencoder_in)
        decoded = decoder(encoded)
        autoencoder = Model(autoencoder_in, decoded, name='AutoEncoder')
        return autoencoder

    def initialise_model(
        self,
        n_bottleneck: int = 10,
        n_size: list = [64, 64],
        activation: str = None,
        epochs: int = 500,
        batch_size: int = 10,
        loss: str = 'mse',
        metrics: list[str] = ['MSE',
                              'MAE',
                              KerasRegressor.r_squared
                              ],
        optimizer=None,
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

        if callbacks is None:
            callbacks = []
            
        if self._tb:
            tb_callback = Custom_TB_Callback(
                logdir=self._tb_logdir,
                run_name=self.run_name,
            )
            callbacks.append(tb_callback)

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

        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(lr=0.001,
                                                 beta_1=0.9,
                                                 beta_2=0.999,
                                                 amsgrad=False,
                                                 clipnorm=1.,
                                                 clipvalue=0.5,
                                                 )

        model = BaseWrapper(
            model=self._get_autoencoder,
            model__seq_len=self.seq_len,
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

        print('\nPredicting data:')

        if self.pred is None:
            pred = self.model.predict(self.data, verbose=1)
            self.pred = pred

        if self.scores is None:
            mae = mean_absolute_error(self.data.squeeze().T,
                                      self.pred.squeeze().T,
                                      multioutput='raw_values',
                                      )
            mse = mean_squared_error(self.data.squeeze().T,
                                     self.pred.squeeze().T,
                                     multioutput='raw_values',
                                     )
            r2 = r2_score(self.data.squeeze().T,
                          self.pred.squeeze().T,
                          multioutput='raw_values',
                          )
            self.scores = {'mae': mae, 'mse': mse, 'r2': r2}

        if x is not None:
            pred = self.model.predict(x, verbose=1)
            mae = mean_absolute_error(x.squeeze().T,
                                      pred.squeeze().T,
                                      multioutput='raw_values'
                                      )
            mse = mean_squared_error(x.squeeze().T,
                                     pred.squeeze().T,
                                     multioutput='raw_values'
                                     )
            r2 = r2_score(x.squeeze().T,
                          pred.squeeze().T,
                          multioutput='raw_values'
                          )
            scores = {'mae': mae, 'mse': mse, 'r2': r2}

        if label is not None and label in label_hash.keys():
            x = np.squeeze(self.data[label_hash[label]])
            pred = np.squeeze(self.pred[label_hash[label]])
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

    def anom_plot(self, anomaly_metric: str = 'mse', plt_ax=None):
        if plt_ax is None:
            fig, ax = plt.subplots(figsize=(7.5, 5))
        else:
            ax = plt_ax

        if self.scores is None:
            self.scores('dataset', print_score=False)
        
        if hasattr(self, 'thres') is False:
            self.thres
        
        anomalies = self.scores[anomaly_metric] > self.thres[anomaly_metric]
        anomalous_data_indices = [False] * len(self.joined_data)
        np.array(anomalous_data_indices)
        for data_idx in range(self.seq_len - 1, len(self.joined_data)):
            if np.all(anomalies[data_idx - self.seq_len + 1:data_idx]):
                anomalous_data_indices[data_idx] = True
        # for i, anom in enumerate(anomalies):
            # if anom:
                # for j in range(i, (i + self.seq_len - 0)):
                # anomalous_data_indices[i] = True

        df_full_rms = pd.DataFrame(self.joined_data)
        df_anom = df_full_rms.copy()
        df_anom.loc[np.invert(anomalous_data_indices)] = np.nan

        df_full_rms.plot(legend=False, ax=ax)
        df_anom.plot(legend=False, ax=ax, color="r")

        ax.set_xlabel('Samples')
        ax.set_ylabel('Rolling RMS (V)')
        ax.legend(labels=['Data', 'Anomalous Data'])
        plt.autoscale(enable=True, axis='x', tight=True)
        self.anomalies = anomalies


if __name__ == '__main__':

    # exps = ['Test 5', 'Test 7', 'Test 8', 'Test 9']
    exps = ['Test 7']

    rms = {}
    for test in exps:
        rms[test] = resources.ae.RMS(test)
        rms[test].data.drop(['0', '1', '2'], axis=1, inplace=True)
    try:
        rms['Test 5'].data.drop(['23', '24'], axis=1, inplace=True)
    except KeyError:
        pass

    # remove outside triggers and DC offset
    def remove_dc(sig):
        return sig - np.nanmean(sig)

    for test in exps:
        rms[test]._data = rms[test].data.T.reset_index(drop=True).T
        rms[test]._data = rms[test].data.iloc[50:350, :].reset_index(drop=True)
        rms[test]._data = rms[test].data.apply(remove_dc, axis=0)

    print()

    for test in exps:
        '''
        autoe = AutoEncoder(rms[test],
                            rms[test].data,
                            random_state=1,
                            train_slice=(0, 100),
                            tb=False,
                            tb_logdir='pickle_test',
                            params={'n_bottleneck': 10,
                                    'n_size': [64, 64],
                                    'epochs': 50,
                                    'loss': 'mse',
                                    'batch_size': 10,
                                    # 'activity_regularizer': None,
                                    }
                            )
        '''
        '''
        autoe = VariationalAutoEncoder(rms[test],
                                       rms[test].data,
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
        '''
        # '''
        autoe = LSTMAutoEncoder(rms[test],
                                rms[test].data,
                                train_slice=(0, 60),
                                tb=False,
                                tb_logdir=rms[test].exp_name,
                                random_state=1,
                                params={'n_bottleneck': 32,
                                        'n_size': [64, 64],
                                        'epochs': 25,
                                        'loss': 'mse',
                                        'batch_size': 32,
                                        'callbacks': [
                                            tf.keras.callbacks.EarlyStopping(
                                                monitor='val_loss',
                                                patience=10,
                                                mode='min',
                                            ),
                                        ]
                                        })
        # '''
        
        # %% ADD MODEL CHECKPOITN CALLBACK
        # -------------------------------------------------------------------
        name = autoe.run_name
        model_folder = TB_DIR.joinpath(autoe._tb_logdir.joinpath(name))
        if not os.path.exists(model_folder):
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

        # %% FIT THE MODEL
        # -------------------------------------------------------------------
        autoe.fit(x=autoe.train_data,
                  val_data=autoe.val_data,
                  verbose=1,
                  use_multiprocessing=True,
                  )

        # %% PLOT LOSS
        # -------------------------------------------------------------------
        autoe.loss_plot()

        # %% MODEL SUMMARY
        # -------------------------------------------------------------------
        autoe.model.model_.summary()
        #autoe.model.model_.encoder.summary()
        #autoe.model.model_.decoder.summary()

        # %% SCORE THE MODEL ON TRAINING, VALIDATION AND ALL DATA
        # -------------------------------------------------------------------
        pred_tr, scores_tr = autoe.score('train')
        pred_val, scores_val = autoe.score('val')
        pred_data, scores_data = autoe.score('dataset')

        # %% HISTOGRAM OF SCORES
        # -------------------------------------------------------------------
        fig, ax = autoe.hist_scores(['mse'])

        # %% PLOT PREDICTIONS
        # -------------------------------------------------------------------
        fig, ax = autoe.pred_plot(autoe._ind_tr[0])
        ax.set_title(f'{autoe.RMS.exp_name} Training Data - {ax.get_title()}')
        fig, ax = autoe.pred_plot(autoe._ind_val[0])
        ax.set_title(f'{autoe.RMS.exp_name} Val Data - {ax.get_title()}')

        # %% CALC CUTOFFS
        # -------------------------------------------------------------------
        autoe.thres

        # %% PLOT SCORES ON SCATTER
        # -------------------------------------------------------------------
        fig, ax = autoe.scatter_scores()

        # %% GENERATE NEW DATA
        # -------------------------------------------------------------------
        try:
            _, fig, ax = autoe.generate(z=[-2, 2])
        except AttributeError:
            pass

        # %% PLOT LATENT SPACE
        # -------------------------------------------------------------------
        try:
            fig, ax = autoe.plot_latent_space()
        except AttributeError:
            pass
        plt.show(block=True)
        
        # %% SAVE MODEL
        # -------------------------------------------------------------------
        mod_path = autoe.save_model()

        # %% LOAD MODEL
        # -------------------------------------------------------------------
        autoe_2 = load_model(mod_path)
