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
from collections import defaultdict
from matplotlib import transforms
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


def pred_plot(mod, input: tuple, no: int):
    """
    Plot prediction vs real data for a given cut.

    Args:
        mod (AutoEncoder): Model to make prediction with.
        input (tuple): Input data array for plotting (real, pred).
        no (int): Cut number to plot.
    
    Returns:
        fig, ax: Matplotlib figure and axis.
    """
    pred_input = input[0][no, :].reshape(-1, mod._n_inputs)
    x_pred = input[1][no, :].reshape(-1, mod._n_inputs)

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

        # scale self.data to be used for predictions
        self.data = self.scaler.transform(data)

        self._n_inputs = x_train.shape[1]
        self.train_data = x_train
        self.val_data = x_test

    @staticmethod
    def get_autoencoder(
        n_inputs: int,
        n_bottleneck: int,
        n_size: list[int],
        activation: str,
    ) -> Model:
        """
        Create a Keras autoencoder model with the given parameters.

        Args:
            n_inputs (int): Number of inputs to the model.
            n_bottleneck (int): Number of nodes in the bottleneck layer.
            n_size (list[int]): List of integers for the number of nodes in \
                the encoder (and decoder but reversed)
            activation (str): Activation function to use.

        Returns:
            Model: Keras model of the autoencoder.
        """
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
        batch_size: int = 10,
        loss: str = 'mse',
        metrics: list[str] = ['MSE',
                              'MAE',
                              KerasRegressor.r_squared
                              ],
        optimizer='adam',
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
                    _ = hp_params.pop('n_size')
                    hp_params['n_size'] = str(layers)

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
            x: np.ndarray,
            print_score: bool = True,
    ) -> tuple[tuple[np.ndarray, np.ndarray], dict]:
        """
        Score the model on the inputted data.

        Scores the model based on predictions made from the input data, will
        also log to tensorboard if the self._tb flag is set to True, and
        print to the console if the print_score flag is set to True.

        Args:
            x (np.ndarray): Input data to score the model on.
            print_score (bool, optional): Print the scores. Defaults to True.
        
        Returns:
            tuple[tuple[np.ndarray, np.ndarray], dict]: A tuple (input,
              prediction) and a dictionary of scores.

        """
        pred = self.model.predict(x, verbose=0)

        mae = mean_absolute_error(x.T, pred.T, multioutput='raw_values')
        mse = mean_squared_error(x.T, pred.T, multioutput='raw_values')
        r2 = r2_score(x.T, pred.T, multioutput='raw_values')

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

        if print_score:
            print(f'\tMAE: {np.mean(mae):.5f}')
            print(f'\tMSE: {np.mean(mse):.5f}')
            print(f'\tR2: {np.mean(r2):.5f}')
        return (x, pred), scores


if __name__ == '__main__':

    # exps = ['Test 5', 'Test 7', 'Test 8', 'Test 9']
    exps = ['Test 7']

    rms = {}
    for test in exps:

        rms[test] = RMS(test)
    try:
        rms['Test 5'].data.drop(['23', '24'], axis=1, inplace=True)
    except KeyError:
        pass

    print()

    for test in exps:

        autoe = AutoEncoder(rms[test],
                            random_state=0,
                            train_slice=(0, 100),
                            tb=False,
                            tb_logdir=rms[test].exp_name,
                            params={'n_bottleneck': 10,
                                    'n_size': [64, 64],
                                    'epochs': 500,
                                    'loss': 'mse',
                                    'batch_size': 10,
                                    }
                            )

        autoe.fit(
            x=autoe.train_data,
            val_data=autoe.val_data,
            verbose=0,
        )

        # compare scores between training and validation data
        print('\nTraining Scores:')
        pred_tr, scores_tr = autoe.score(autoe.train_data)
        print('\nValidation Scores:')
        pred_val, scores_val = autoe.score(autoe.val_data)

        # plot a prediciton from both the training and validation data
        pred_plot(autoe, pred_tr, 0)
        pred_plot(autoe, pred_val, 0)

        # plot histogram of training scores
        def hist_scores(scores, metrics: list = None):

            sc = defaultdict(list)

            if metrics is None:
                metrics = scores[0].keys()
            for score in scores:
                for key, score in score.items():
                    if key in metrics:
                        sc[key].extend(score)

            for key, score in sc.items():
                fig, ax = plt.subplots()
                ax.hist(score, bins=50)
                ax.set_xlabel(f'{key} error')
                ax.set_ylabel('No of Occurences')
                ax.set_title(f'Histogram of training dataset prediciton {key}')

        # plot a scatter graph of the training scores
        def scatter_scores(scores, thr: dict = None, metrics: list = None):

            sc = defaultdict(list)

            if metrics is None:
                metrics = scores[0].keys()
            for score in scores:
                for key, score in score.items():
                    if key in metrics:
                        sc[key].extend(score)

            for key, score in sc.items():
                fig, ax = plt.subplots()
                ax.set_xlabel('Cut Number')
                ax.set_ylabel(f'{key.upper()}')
                ax.set_title(f'Scatter of training dataset prediciton {key}')
                if thresholds is not None and key in thr.keys():
                    ax.axhline(thr[key], color='r', linestyle='--')

                    # create cmap for plot depending on if the scores is
                    # above/below the threshold
                    if key == 'r2':
                        cmap = ['b' if y > thr[key] else 'r'
                                for y in sc[key]]
                    else:
                        cmap = ['r' if y > thr[key] else 'b'
                                for y in sc[key]]
                        
                    ax.scatter(x=range(len(sc[key])),
                               y=sc[key],
                               s=2,
                               c=cmap)
                    trans = transforms.blended_transform_factory(
                        ax.get_yticklabels()[0].get_transform(), ax.transData)
                    ax.text(0, thr[key], "{:.2f}".format(thr[key]),
                            color="red",
                            transform=trans,
                            ha="right",
                            va="center"
                            )

        hist_scores([scores_tr, scores_val], metrics=['mse'])

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
                print(f'{key.upper()} cutoff: {cutoffs[key]:.5f}')
            return cutoffs

        print('\nCutoffs:')
        thresholds = calc_cutoff([scores_tr, scores_val])

        # prediction on whole dataset
        print('\nWhole Dataset Scores:')
        pred, scores = autoe.score(autoe.data)
        scatter_scores([scores], thr=thresholds)

        fig, ax = pred_plot(autoe, pred, 150)
        ax.set_title(f'{autoe.exp_name} Unseen Data - {ax.get_title()}')

        plt.show(block=True)

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
