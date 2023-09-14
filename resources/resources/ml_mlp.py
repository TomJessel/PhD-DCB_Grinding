"""
@File    :   ml_mlp.py
@Author  :   Tom Jessel
@Contact :   jesselt@cardiff.ac.uk

@Creation Time    @Author    @Version    @Description
------------      -------    --------    -----------
26/10/2022 10:01   tomhj      1.0        ML classes for architectures
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import multiprocessing
import time
from textwrap import dedent
from typing import Any, Dict, Iterable, List, Union, Tuple
from pathlib import PurePosixPath as Path
import numpy as np
import pandas as pd
from collections import deque
from matplotlib import pyplot as plt
from scikeras.wrappers import KerasRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

import tensorflow as tf  # noqa: E402
import tensorflow_addons as tfa  # noqa: E402
import tensorboard.plugins.hparams.api as hp
from tensorboard.backend.event_processing import event_accumulator

tf.config.set_visible_devices([], 'GPU')
from absl import logging
logging.set_verbosity(logging.ERROR)

from keras.layers import Dense, Dropout, LSTM  # noqa: E402
from keras.models import Sequential  # noqa: E402
from keras.optimizers import Adam  # noqa: E402


def parse_tensorboard(path: str, scalars: list[str]):
    """
    Creates a Dict of dataframes for each requested scalar from the folder path

    Args:
       path: Path containing TB files
       scalars: List of scalars in the TB logs

    Returns:

    """
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={'tensors': 0},
    )
    ea.Reload()
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


class Base_Model:
    def __init__(
            self,
            target: Iterable = None,
            feature_df: pd.DataFrame = None,
            tb: bool = True,
            params: Dict = None
    ):
        """
        Base_Model constructor.

        Args:
            target: Label of the feature to predict from the dataframe
            feature_df: Feature dataframe of which to train and predict
            tb: Option to record model progress and results in Tensorboard
        """
        self._no_features = None
        self.seq_len = None
        self.model = None
        self.main_df = feature_df
        self.target = target
        self.train_data = pd.DataFrame
        self.val_data = pd.DataFrame
        self._tb = tb
        self._run_name = None

        # Tensorboard filename
        dirname = self.get_file_dir()
        self.tb_log_dir = os.path.join(dirname, 'Tensorboard')

        if self.target is None:
            raise AttributeError('There is no TARGET attribute set.')
        if self.main_df is None:
            raise AttributeError('There is no MAIN_DF attribute.')

        if params is None:
            params = {}
        self.params = params

    @staticmethod
    def get_file_dir():
        """
        Get path to the AE file in OneDrive folder

        Returns:
            file path to AE folder

        """
        platform = os.name
        if platform == 'nt':
            onedrive = Path(r'C:\Users\tomje\OneDrive - Cardiff University')
            filename = onedrive.joinpath('Documents/PHD/AE')
        elif platform == 'posix':
            onedrive = Path(
                r'/mnt/c/Users/tomje/OneDrive - Cardiff University/Documents'
            )
            filename = onedrive.joinpath('PHD/AE')
        return filename

    def pre_process(self):
        raise AttributeError('No assigned function to pre-process \
                            data for Base_Model')

    def fit(
            self,
            X: np.ndarray = None,
            y: np.ndarray = None,
            **kwargs,
    ):
        """
        Fit the model with the keras fit method.

        Args:
            X: Input training data for model to learn from.
            y: Corresponding output for model to train with.
            **kwargs: Additional inputs for keras fit method.
        """
        if X is None:
            try:
                X = self.train_data[0].values
            except AttributeError:
                X = self.train_data[0]
        if y is None:
            try:
                y = self.train_data[1].values
            except AttributeError:
                y = self.train_data[1]

        print('-' * 65)
        print(f'{self._run_name.split(self.tb_log_dir)[1][1:]}')

        self.model.fit(X=X, y=y, **kwargs)

        if self._tb:
            tb_writer = tf.summary.create_file_writer(self._run_name)
            self.tb_model_desc(tb_wr=tb_writer)

    def tb_model_desc(self, tb_wr):
        # Model.summary()
        lines = []
        self.model.model_.summary(print_fn=lines.append)

        dropout = self.model.model_.layers[1].get_config()['rate']
        layers = self.model.model_.get_config()['layers']
        nodes = [layer['config']['units'] for layer in layers if
                 layer['class_name'] in ('Dense', 'LSTM')]
        no_layers = len(nodes) - 1
        activation = layers[1]['config']['activation']
        opt = self.model.model_.optimizer.get_config()
        optimiser = opt['name']
        learning_rate = opt['learning_rate']

        hp = dedent(f"""
            ### Parameters:
            ___

            | Epochs | Batch Size | No Layers | No Neurons | Init Mode |\
             Activation | Dropout | Loss | Optimiser | Learning rate |
            |--------|------------|-----------|------------|-----------|\
            ------------|---------|------|-----------|---------------|
            |{self.model.epochs}|{self.model.batch_size}|{no_layers}|{nodes[:-1]}|{self.model.model__init_mode}|\
            {activation}|{dropout:.3f}|{self.model.loss}|{optimiser}|{learning_rate:.3E}|

            """)

        with tb_wr.as_default():
            # Code to ouput Tensorboard model.summary
            # lines = '    ' + '\n    '.join(lines)
            # tf.summary.text('Model Info', lines, step=0)
            # Code to output Tensorboard hyperparams
            tf.summary.text('Model Info', hp, step=1)

    def score(
            self,
            model: KerasRegressor = None,
            X: np.ndarray = None,
            y: np.ndarray = None,
            plot_fig: bool = False,
            print_score: bool = True
    ) -> Dict:
        """
        Score the mlp regression model on unseen data.

        Use trained model to predict results on unseen validation data,
        and then calc metrics for scoring.
        Args:
            model: ML model to score
            X: Inputs for predictions from unseen validation data set
            y: Corresponding outputs from validation data set
            plot_fig: Choice to plot the predictions plot
            print_score: Choice to print scores to terminal

        Returns:
            Dict containing the calculated scores
        """
        if model is None:
            model = self.model
        if X is None:
            try:
                X = self.val_data[0].values
            except AttributeError:
                X = self.val_data[0]
        if y is None:
            try:
                y = self.val_data[1].values
            except AttributeError:
                y = self.val_data[1]

        # noinspection PyTypeChecker
        y_pred = model.predict(X, verbose=0)
        _test_score = {
            'MAE': mean_absolute_error(y, y_pred),
            'MSE': mean_squared_error(y, y_pred),
            'r2': r2_score(y, y_pred),
        }
        if print_score:
            print('-' * 65)
            # print(f'{self._run_name.split(self.tb_log_dir)[1][1:]}')
            print('Validation Scores:')
            print('-' * 65)
            print(f'MAE = {np.abs(_test_score["MAE"]) * 1000:.3f} um')
            print(f'MSE = {np.abs(_test_score["MSE"]) * 1_000_000:.3f} um^2')
            print(f'R^2 = {np.mean(_test_score["r2"]):.3f}')
            # print('-' * 65)

        if plot_fig:
            fig, ax = plt.subplots()
            ax.plot(y, color='red', label='Real data')
            ax.plot(y_pred, color='blue', ls='--', label='Predicted data')
            ax.set_title('Model Predictions - Test Set')
            ax.set_ylabel('Mean Radius (mm)')
            ax.set_xlabel('Data Points')
            ax.legend()
            plt.show()

        if self._tb:
            tb_writer = tf.summary.create_file_writer(self._run_name)
            md_scores = dedent(f'''
                    ### Scores - Validation Data

                     | MAE | MSE |  R2  |
                     | ---- | ---- | ---- |
                     | {_test_score['MAE'] * 1e3:.3f} |\
                         {_test_score['MSE'] * 1e6:.3f} |\
                             {_test_score['r2']:.3f} |

                    ''')
            with tb_writer.as_default():
                tf.summary.text('Model Info', md_scores, step=2)
                tf.summary.scalar('Val MSE (\u00B5m\u00B2)',
                                  (np.abs(_test_score["MSE"]) * 1_000_000),
                                  step=1)
                tf.summary.scalar('Val MAE (\u00B5m)',
                                  (np.abs(_test_score["MAE"]) * 1000), step=1)
                tf.summary.scalar('Val R\u00B2',
                                  (np.mean(_test_score['r2'])), step=1)

        return _test_score

    def initialise_model(self, verbose=1, **params) -> Any:
        self._run_name = f'{self.tb_log_dir}\\Base-\
        {time.strftime("%Y%m%d-%H%M%S",time.localtime())}'
        pass

    def _cv_model(
            self,
            run_no,
            tr_index,
            te_index,
    ) -> Tuple[Dict, Any]:
        """
        Multiprocessing worker function to cross validate \
         one instance of the model.

        Args:
            run_no: Iteration of the cv model
            tr_index: Index locations for the training data in the split
            te_index: Index locations for the testing data in the split

        Returns:
            Dict containing models scores
            Model that was scored on
        """
        self._tb = False
        model = self.initialise_model(verbose=0, **self.params)
        model.callbacks = None

        try:
            X = self.train_data[0].values
            y = self.train_data[1].values
        except AttributeError:
            X = self.train_data[0]
            y = self.train_data[1]

        model.fit(
            X=X[tr_index],
            y=y[tr_index],
            # validation_data=(self.train_data[0].values[tr_index],\
            #  self.train_data[1].values[tr_index])
        )
        score = self.score(
            model=model,
            X=X[te_index],
            y=y[te_index],
            print_score=False,
        )
        self._tb = True
        score['run_no'] = run_no
        return score, model

    def _cv_model_star(self, args):
        """
        Function to allow imap to use multiple inputs
        """
        return self._cv_model(*args)

    def cv(self,
           n_splits: int = 5,
           n_repeats: int = None,
           random_state: Any = None,
           ) -> Dict:
        """
        Cross validate a ML model with either Kfold or Repeated Kfold

        If no input given to n_repeats, Kfold is used, if n_repeats is an int \
            Repeated Kfold is used instead

        Args:
            n_splits: No splits in the CV
            n_repeats: No repeats to do with Repeated Kfold CV
            random_state: Random state of the CV, if no input random

        Returns:
            Dict of cv scores including mean values and stds

        """

        from sklearn.model_selection import KFold, RepeatedKFold

        print('-' * 65)
        print(f'{self._run_name.split(self.tb_log_dir)[1][1:]}')

        # Use Kfold or Repeated Kfold
        if n_repeats is None:
            cv = KFold(n_splits=n_splits,
                       shuffle=True,
                       random_state=random_state,
                       )
        else:
            cv = RepeatedKFold(n_splits=n_splits,
                               n_repeats=n_repeats,
                               random_state=random_state,
                               )

        try:
            X = self.train_data[0].values
        except AttributeError:
            X = self.train_data[0]

        cv_items = [(i, train, test) for i, (train, test) in
                    enumerate(cv.split(X))]

        with multiprocessing.Pool(processes=20, maxtasksperchild=1) as pool:
            outputs = list(tqdm(pool.imap(self._cv_model_star,
                                          cv_items,
                                          chunksize=1,
                                          ),
                                total=len(cv_items),
                                desc='CV Model'
                                ))
            pool.close()
            pool.join()

        # outputs = []
        # for cv_item in tqdm(cv_items):
        #     outputs.append(self._cv_model_star(cv_item))

        scores = [output[0] for output in outputs]
        # models = [output[1] for output in outputs]

        mean_MAE = np.mean([score['MAE'] for score in scores])
        mean_MSE = np.mean([score['MSE'] for score in scores])
        mean_r2 = np.mean([score['r2'] for score in scores])

        std_MAE = np.std([score['MAE'] for score in scores])
        std_MSE = np.std([score['MSE'] for score in scores])
        std_r2 = np.std([score['r2'] for score in scores])

        print('-' * 65)
        # print(f'{self._run_name.split(self.tb_log_dir)[1][1:]}')
        print('CV Scores:')
        print('-' * 65)
        print(dedent(f'MAE: {mean_MAE * 1_000:.3f} (\u00B1'
                     f'{std_MAE * 1_000:.3f})\u00B5m'))
        print(dedent(f'MSE: {mean_MSE * 1_000_000:.3f} (\u00B1'
                     f'{std_MSE * 1_000_000:.3f}) \u00B5m\u00B2'))
        print(f'R^2: {mean_r2:.3f} (\u00B1{std_r2: .3f})')
        # print('-' * 65)

        _cv_score = {
            'MAE': mean_MAE,
            'MSE': mean_MSE,
            'r2': mean_r2,
            'std_MAE': std_MAE,
            'std_MSE': std_MSE,
            'std_r2': std_r2,
        }

        if self._tb:
            tb_writer = tf.summary.create_file_writer(self._run_name)
            md_scores = dedent(f'''
                    ### Scores - Cross-validation
                    No splits = {n_splits}  No repeats = {n_repeats}

                     | MAE | MSE |  R2  |
                     | ---- | ---- | ---- |
                     | {mean_MAE * 1e3:.3f} | {mean_MSE * 1e6:.3f} |\
                         {mean_r2:.3f}  |
                     | (\u00B1{std_MAE * 1_000: .3f}) |\
                         (\u00B1{std_MSE * 1_000_000: .3f}) |\
                             (\u00B1{std_r2: .3f})

                    ''')
            with tb_writer.as_default():
                tf.summary.text('Model Info', md_scores, step=3)
                tf.summary.scalar('CV MSE (\u00B5m\u00B2)',
                                  (mean_MSE * 1e6), step=1)
                tf.summary.scalar('CV MAE (\u00B5m)', (mean_MAE * 1e3), step=1)
                tf.summary.scalar('CV R\u00B2', mean_r2, step=1)
                tf.summary.scalar('CV Std MSE (\u00B1 \u00B5m\u00B2)',
                                  (std_MSE * 1e6), step=1)
                tf.summary.scalar('CV Std MAE (\u00B1 \u00B5m)',
                                  (std_MAE * 1e3), step=1)
                tf.summary.scalar('CV Std R\u00B2 (\u00B1)', std_r2, step=1)
                step = 0
                step = tf.convert_to_tensor(step, dtype=tf.int64)
                for score in scores:
                    tf.summary.scalar('cv_iter/mae', score['MAE'], step=step)
                    tf.summary.scalar('cv_iter/mse', score['MSE'], step=step)
                    tf.summary.scalar('cv_iter/r2', score['r2'], step=step)
                    step += 1
        return _cv_score

    def loss_plot(self,
                  ax: plt.Axes = None,
                  ) -> Any:
         
        if hasattr(self.model, 'history_') is False:
            raise ValueError('Model has not been fit yet.')
        
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        
        ax.plot(self.model.history_['loss'], label='Loss')
        ax.plot(self.model.history_['val_loss'], label='Val Loss')
        ax.legend()
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        return fig, ax


class MLP_Model(Base_Model):
    def __init__(
            self,
            # params: Dict = None,
            tb_logdir: str = '',
            random_state: int = None,
            shuffle: bool = True,
            **kwargs,
    ):
        """
        MLP_Model constructor.

        Passes the params dict to the creation of the mlp model. And passes\
             **kwargs to class creation

        Args:
            params: Dict of mlp model parameters passed to \
                self.initialise_model
            tb_logdir: Folder path for tb logging, i.e. "CV" -> \
                "Tensorboard\\MLP\\CV"
            **kwargs: Inputs for Base_Model init
        """

        super().__init__(**kwargs)
        # if params is None:
        #     params = {}
        # self.params = params

        self.tb_log_dir = os.path.join(self.tb_log_dir, 'MLP', tb_logdir)

        if self.main_df is not None:
            self.pre_process(random_state=random_state,
                             shuffle=shuffle,
                             )
            self.model = self.initialise_model(**self.params)
        else:
            print('Choose data file to import as main_df')

    def pre_process(
            self: Base_Model,
            val_frac: float = 0.2,
            random_state: int = None,
            shuffle: bool = True,
    ) -> Tuple[Tuple[Any, Any], Tuple[Any, Any]]:
        """
        Pre-process the data for training an MLP model

        Split the data for the test into a training and validation set
        Then scale the data using a MinMax scaler, based off the
        training data. Then save the splits into a tuple which
        contains the features and results separately.

        Args:
            val_frac: Fraction of data points used within the validation set

        Returns:
            Two tuples which contain dataframes of the features and results
                for the training and validation sets.

        """
        scaler = MinMaxScaler()
        
        train, test = train_test_split(self.main_df,
                                       test_size=val_frac,
                                       random_state=random_state,
                                       shuffle=shuffle,
                                       )
        
        train_y = train[[self.target]].to_numpy()
        train_x = train.drop(columns=self.target).to_numpy()
        test_y = test[[self.target]].to_numpy()
        test_x = test.drop(columns=self.target).to_numpy()

        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)

        self.train_data = (train_x, train_y)
        self.val_data = (test_x, test_y)
        self.scaler = scaler
        return self.train_data, self.val_data

    @staticmethod
    def build_mod(
            no_features=7,
            dropout=0.2,
            no_layers=2,
            no_nodes=32,
            activation='relu',
            init_mode='glorot_normal',
    ) -> Sequential:
        """
        Creates a vanilla MLP Keras Sequential Model.

        Returns a simple MLP model consisting of Dense and Dropout layers,\
             the number of which and sizes can be determined via function\
             inputs.

        Args:
            no_features: No of input features calc from train data
            dropout: Dropout rate for every dropout layer
            no_layers: No of Dense layers to include in the model
            no_nodes: No of neurons to include in each dense layer
            activation: The activation function for the neurons'
            init_mode: Initialisation method for the neuron weights

        Returns:
            model: Compiled Keras MLP model

        """
        model = Sequential(name='MLP_reg')
        model.add(Dense(
            units=no_nodes,
            activation=activation,
            input_shape=(no_features,),
            kernel_initializer=init_mode,
            name='dense1',
            use_bias=True,
        ))
        model.add(Dropout(rate=dropout, name='dropout1'))
        for i, _ in enumerate(range(no_layers - 1)):
            model.add(Dense(
                units=no_nodes,
                activation=activation,
                kernel_initializer=init_mode,
                name=f'dense{i + 2}',
                use_bias=True,
            ))
            model.add(Dropout(rate=dropout, name=f'dropout{i + 2}'))
        model.add(Dense(1, name='output'))
        return model

    def initialise_model(
            self,
            dropout: float = 0.1,
            activation: str = 'relu',
            no_nodes: int = 32,
            no_layers: int = 2,
            init_mode: str = 'glorot_normal',
            epochs: int = 500,
            batch_size: int = 10,
            loss: str = 'mae',
            metrics: Union[list, tuple] = ('MSE',
                                           'MAE',
                                           KerasRegressor.r_squared
                                           ),
            optimizer: str = Adam,
            learning_rate: float = 0.001,
            verbose: int = 1,
            callbacks: List[Any] = None,
            **params,
    ) -> KerasRegressor:
        """
        Initialise a SciKeras Regression model with the given hyperparameters

        Uses the specified parameters to create a Scikeras KerasRegressor.\
        Allowing a Keras model to be produced easily with all the inputted\
        parameters.

        Args:
            dropout: Dropout rate of each dropout layer
            activation: Activation function for the neurons'
            no_nodes: No neurons in each Dense layer
            no_layers: No of Dense and Dropout layers to create the model with
            init_mode: Initialisation method for the weights of each neuron
            epochs: No of epochs to train the model with
            batch_size: No of samples for the model to train with before\
                 updating weights
            loss: Metric to optimise the model for.
            metrics: Metrics to track during training of the model
            optimizer: Optimizer function used to allow model to learn
            learning_rate: Learning rate of the optimizer
            verbose: Output type for the console
            callbacks: Callbacks to add into the Keras model.

        Returns: A KerasRegressor into the model attribute

        """
        try:
            no_features = pd.DataFrame.to_numpy(self.train_data[0]).shape[1]
        except AttributeError:
            no_features = self.train_data[0].shape[1]

        # tensorboard set-up
        logdir = self.tb_log_dir
        t = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        self._run_name = os.path.join(logdir,
                                      f'MLP-E-{epochs}-B-{batch_size}-'
                                      f'L{np.full(no_layers, no_nodes)}-'
                                      f'D-{dropout}'
                                      f'-{t}')

        if callbacks is None:
            callbacks = []
        # Add in TQDM progress bar for fitting
        callbacks.append(tfa.callbacks.TQDMProgressBar(
            show_epoch_progress=False))

        if self._tb:
            # Add in tensorboard logging
            callbacks.append(tf.keras.callbacks.TensorBoard(
                log_dir=self._run_name, histogram_freq=1))
            tb_writer = tf.summary.create_file_writer(self._run_name)
            with tb_writer.as_default():
                hp_params = self.params
                hp_params.pop('callbacks', None)
                hp.hparams(
                    hp_params,
                    trial_id=self._run_name.split(self.tb_log_dir)[1][1:]
                )

        model = KerasRegressor(
            model=self.build_mod,
            model__no_features=no_features,
            model__dropout=dropout,
            model__activation=activation,
            model__no_nodes=no_nodes,
            model__no_layers=no_layers,
            model__init_mode=init_mode,
            epochs=epochs,
            batch_size=batch_size,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            optimizer__learning_rate=learning_rate,
            verbose=verbose,
            callbacks=callbacks,
        )
        return model


class Linear_Model(Base_Model):
    def __init__(
            self,
            # params: Dict = None,
            **kwargs,
    ):
        """
        Linear_Model constructor.

        Passes the params dict to the creation of the linear model. And passes\
         **kwargs to class creation

        Args:
            params: Dict of linear model parameters passed to \
            self.initialise_model
            **kwargs: Inputs for Base_Model init
        """
        super().__init__(**kwargs)
        # if params is None:
        #     params = {}
        # self.params = params

        if self.main_df is not None:
            self.pre_process()
            self.model = self.initialise_model(**self.params)
        else:
            print('Choose data file to import as main_df')
        self._tb = False

    def pre_process(
            self,
            val_frac: float = 0.2,
    ) -> tuple[Any, Any]:
        """
        Function to pre-process the data for a linear model

            Uses MLP_Model pre_process function.

        Args:
            val_frac: Fraction of data points used within the validation set

        Returns:
            Two tuples which contain dataframes of the features and results \
            for the training and validation sets.

        """
        func = MLP_Model.pre_process
        train_data, val_data = func(self, val_frac=val_frac)
        return train_data, val_data

    def initialise_model(
            self,
            **params,
    ) -> Any:
        """
        Initialise a Linear regression model with optional parameters

        Args:
            **params: Optional params passed to sklearn.LinearRegression func

        Returns:
            Sklearn LinearRegression model
        """
        from sklearn.linear_model import LinearRegression
        model = LinearRegression(**params)
        t = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        self._run_name = os.path.join(self.tb_log_dir, f'Lin_Reg-{t}')
        return model

    def score(
            self,
            model: KerasRegressor = None,
            X: np.ndarray = None,
            y: np.ndarray = None,
            plot_fig: bool = False,
            print_score: bool = True
    ) -> Dict:
        """
        Score the linear regression model on unseen data.

        Use trained model to predict results on unseen validation data, \
        and then calc metrics for scoring. And plot visualisation of the \
        predictions and feature impacts.

        Args:
            model: ML model to score
            X: Inputs for predictions from unseen validation data set
            y: Corresponding outputs from validation data set
            plot_fig: Choice to plot the predictions plot
            print_score: Choice to print scores to terminal

        Returns:
            Dict containing the calculated scores
        """
        from sklearn.model_selection import RepeatedKFold, cross_validate

        if model is None:
            model = self.model
        if X is None:
            X = self.val_data[0].values
        if y is None:
            y = self.val_data[1].values

        scoring = {'MAE': 'neg_mean_absolute_error',
                   'MSE': 'neg_mean_squared_error',
                   'r2': 'r2'}

        cv = RepeatedKFold(n_splits=10, n_repeats=10)

        _test_score = cross_validate(estimator=model,
                                     X=X,
                                     y=y,
                                     scoring=scoring,
                                     cv=cv,
                                     return_train_score=False,
                                     n_jobs=-1,
                                     )

        if print_score:
            print('-' * 65)
            print(f'{self._run_name.split(self.tb_log_dir)[1][1:]}')
            print('Validation Scores:')
            print('-' * 65)
            print(
                f'MAE = {np.abs(_test_score["test_MAE"].mean()) * 1e3:.3f}um'
            )
            print(
                f'MSE = {np.abs(_test_score["test_MSE"].mean()) * 1e6:.3f}um^2'
            )
            print(f'R^2 = {np.mean(_test_score["test_r2"].mean()):.3f}')
            # print('-' * 65)

        if plot_fig:
            model.fit(X, y)

            no_features = np.shape(X)[1]
            fig, ax = plt.subplots(1, no_features)
            # loop to isolate results/predictions based on one feature
            for i in range(no_features):
                xaxis = np.arange(X[:, i].min(), X[:, i].max(), 0.01)
                x_0 = np.zeros((len(xaxis), no_features))
                x_0[:, i] = xaxis
                yaxis = model.predict(x_0)

                ax[i].scatter(X[:, i], y)
                ax[i].plot(xaxis, yaxis, color='r')
            plt.show()

            y_pred = model.predict(X)
            fig, ax = plt.subplots()
            ax.plot(y, color='red', label='Real data')
            ax.plot(y_pred, color='blue', ls='--', label='Predicted data')
            ax.set_title('Model Predictions - Test Set')
            ax.set_ylabel('Mean Radius (mm)')
            ax.set_xlabel('Data Points')
            ax.legend()
            plt.show()
        return _test_score

    def fit(
            self,
            X: np.ndarray = None,
            y: np.ndarray = None,
            **kwargs,
    ):
        """
        Fit the model with the keras fit method.

        Args:
            X: Input training data for model to learn from.
            y: Corresponding output for model to train with.
            **kwargs: Additional inputs for keras fit method.
        """
        if X is None:
            try:
                X = self.train_data[0].values
            except AttributeError:
                X = self.train_data[0]
        if y is None:
            try:
                y = self.train_data[1].values
            except AttributeError:
                y = self.train_data[1]

        self.model.fit(X=X, y=y, **kwargs)


class MLP_Win_Model(Base_Model):
    def __init__(
            self,
            # params: Dict = None,
            tb_logdir: str = '',
            random_state: int = None,
            shuffle: bool = True,
            **kwargs,
    ):
        """
        MLP_Win_Model constructor.

        Passes the params dict to the creation of the mlp win model. \
        And passes **kwargs to class creation

        Args:
            params: Dict of mlp model parameters passed to \
                self.initialise_model
            tb_logdir: Folder path for tb logging, i.e. "CV" -> \
                "Tensorboard\\MLP\\CV"
            **kwargs: Inputs for Base_Model init
        """

        super().__init__(**kwargs)

        # if params is None:
        #     params = {}
        # self.params = params
        # get sequnce len for window from param dict
        self.seq_len = self.params.pop('seq_len', 10)

        self.tb_log_dir = os.path.join(self.tb_log_dir, 'MLP_WIN', tb_logdir)

        if self.main_df is not None:
            self.pre_process(random_state=random_state,
                             shuffle=shuffle,
                             )
            self.model = self.initialise_model(**self.params)
        else:
            print('Choose data file to import as main_df')

    def sequence_data(self, d: np.ndarray):
        """
        Applies window effect to change feature to include previous \
        self.seq_len features

        Args:
            d: array of features and targets

        Returns:
            tuple containing the windowed data and answer
        """
        seq_data = []
        seq_len = self.seq_len
        prev_points = deque(maxlen=seq_len)

        for i in d:
            # todo change this to adapt if multiple targets
            prev_points.append([n for n in i[:-1]])
            if len(prev_points) == seq_len:
                seq_data.append([np.array(prev_points), i[-1]])
        return seq_data

    def pre_process(
            self: Base_Model,
            val_frac: float = 0.2,
            random_state: int = None,
            shuffle: bool = True,
    ) -> Tuple[List[Any], List[Any]]:
        """
        Pre-process the data for training an MLP Win model

            Split the data for the test into a training and validation set. \
            Then scale the data using a MinMax scaler, based off the training \
            data index. Then window the data, to preserve time. Then split the\
             data based on the saved indicies.

        Args:
            val_frac: Fraction of data points used within the validation set

        Returns:
            Two tuples which contain dataframes of the features and results \
            for the training and validation sets.

        """

        # 4* seqlen for removing overlap
        indx = np.arange(len(self.main_df))
        # save index and pos of the train test split
        train_i, test_i = train_test_split(indx,
                                           test_size=val_frac,
                                           shuffle=shuffle,
                                           random_state=random_state,
                                           )

        scaler = MinMaxScaler()
        m_df = self.main_df

        y = m_df[self.target].to_numpy()
        X = m_df.drop(columns=self.target).to_numpy()

        scaler.fit(X[train_i])
        X = scaler.transform(X)
        self.scaler = scaler

        Xy = np.column_stack((X, y))
        m_df = pd.DataFrame(Xy, columns=m_df.columns)

        # window the dataset
        m_df = self.sequence_data(m_df.values)
        m_df = pd.DataFrame(m_df, columns=['features', 'target'])
        m_df.index += self.seq_len - 1

        # index position of the end of each dataframe
        # todo need to change to get automatically
        df_ends = np.cumsum([207, 159, 172, 154])

        # try to remove overlapping
        # indicies of data to remove from model
        del_indx = list(range(0, (self.seq_len - 1)))
        indx = np.delete(indx, del_indx)

        for end in df_ends[:-1]:
            del_indx_overlap = list(range(end, (end + (self.seq_len - 1))))
            try:
                indx = np.delete(indx, del_indx_overlap)
            except IndexError:
                print('Overlapping sections between exps not removed!')

        # split data set indicies into train and test
        temp_train_i = [element for element in train_i
                        if element not in del_indx]
        temp_test_i = [element for element in test_i
                       if element not in del_indx]

        # separate the train and test datasets
        train = m_df.loc[temp_train_i, :]
        test = m_df.loc[temp_test_i, :]

        train_X = []
        train_y = []

        for X, y in train.values:
            train_X.append(X)
            train_y.append(y)

        test_X = []
        test_y = []

        for X, y in test.values:
            test_X.append(X)
            test_y.append(y)

        train_X = np.asarray(train_X)
        train_y = np.asarray(train_y)
        test_X = np.asarray(test_X)
        test_y = np.asarray(test_y)

        self.train_data = [train_X, train_y]
        self.val_data = [test_X, test_y]
        # reshape feature data for MLP win model input
        self._no_features = (self.train_data[0].shape[1] * self.train_data[0]
                             .shape[2])
        self.train_data[0] = self.train_data[0].reshape(
            (self.train_data[0].shape[0], self._no_features))
        self.val_data[0] = self.val_data[0].reshape(
            (self.val_data[0].shape[0], self._no_features))
        return self.train_data, self.val_data

    def initialise_model(
            self,
            dropout: float = 0.1,
            activation: str = 'relu',
            no_nodes: int = 32,
            no_layers: int = 2,
            init_mode: str = 'glorot_normal',
            epochs: int = 500,
            batch_size: int = 10,
            loss: str = 'mae',
            metrics: Union[list, tuple] = ('MSE',
                                           'MAE',
                                           KerasRegressor.r_squared
                                           ),
            optimizer: str = Adam,
            learning_rate: float = 0.001,
            verbose: int = 1,
            callbacks: List[Any] = None,
            **params,
    ) -> KerasRegressor:
        """
        Initialise a SciKeras Regression model with the given hyperparameters

        Uses the specified parameters to create a Scikeras KerasRegressor. \
        Allowing a Keras model to be produced easily with all the \
        inputted parameters.

        Args:
            dropout: Dropout rate of each dropout layer
            activation: Activation function for the neurons'
            no_nodes: No neurons in each Dense layer
            no_layers: No of Dense and Dropout layers to create the model with
            init_mode: Initialisation method for the weights of each neuron
            epochs: No of epochs to train the model with
            batch_size: No of samples for the model to train with before \
                updating weights
            loss: Metric to optimise the model for.
            metrics: Metrics to track during training of the model
            optimizer: Optimizer function used to allow model to learn
            learning_rate: Learning rate of the optimizer
            verbose: Output type for the console
            callbacks: Callbacks to add into the Keras model.

        Returns: A KerasRegressor into the model attribute

        """

        # tensorboard set-up
        logdir = self.tb_log_dir
        t = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        self._run_name = os.path.join(logdir,
                                      f'MLP_Win-WLEN-{self.seq_len}-'
                                      f'E-{epochs}-'
                                      f'B-{batch_size}-'
                                      f'L-{np.full(no_layers, no_nodes)}-'
                                      f'D-{dropout}-'
                                      f'{t}')

        if callbacks is None:
            callbacks = []
        # Add in TQDM progress bar for fitting
        callbacks.append(tfa.callbacks.TQDMProgressBar(
            show_epoch_progress=False))

        if self._tb:
            # Add in tensorboard logging
            callbacks.append(tf.keras.callbacks.TensorBoard(
                log_dir=self._run_name, histogram_freq=1))
            tb_writer = tf.summary.create_file_writer(self._run_name)
            with tb_writer.as_default():
                hp_params = self.params
                hp_params['seq_len'] = self.seq_len
                hp_params.pop('callbacks', None)
                hp.hparams(
                    hp_params,
                    trial_id=self._run_name.split(self.tb_log_dir)[1][1:]
                )

        model = KerasRegressor(
            model=MLP_Model.build_mod,
            model__no_features=self._no_features,
            model__dropout=dropout,
            model__activation=activation,
            model__no_nodes=no_nodes,
            model__no_layers=no_layers,
            model__init_mode=init_mode,
            epochs=epochs,
            batch_size=batch_size,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            optimizer__learning_rate=learning_rate,
            verbose=verbose,
            callbacks=callbacks,
        )
        return model

    def tb_model_desc(self, tb_wr):
        # Model.summary()
        lines = []
        self.model.model_.summary(print_fn=lines.append)

        dropout = self.model.model_.layers[1].get_config()['rate']
        layers = self.model.model_.get_config()['layers']
        nodes = [layer['config']['units'] for layer in layers
                 if layer['class_name'] in ('Dense', 'LSTM')]
        no_layers = len(nodes) - 1
        activation = layers[1]['config']['activation']
        opt = self.model.model_.optimizer.get_config()
        optimiser = opt['name']
        learning_rate = opt['learning_rate']

        hp = dedent(f"""
            ### Parameters:
            ___

            |Seq Len| Epochs | Batch Size | No Layers | No Neurons | \
            Init Mode | Activation | Dropout | Loss | \
            Optimiser | Learning rate |
            |--------|--------|------------|-----------|------------|\
            -----------|------------|---------|------|-----------|\
            ---------------|
            |{self.seq_len}|{self.model.epochs}|{self.model.batch_size}|{no_layers}|{nodes[:-1]}|\
            {self.model.model__init_mode}|{activation}|{dropout:.3f}|{self.model.loss}|{optimiser}|\
            {learning_rate:.3E}|

            """)

        with tb_wr.as_default():
            # Code to ouput Tensorboard model.summary
            # lines = '    ' + '\n    '.join(lines)
            # tf.summary.text('Model Info', lines, step=0)
            # Code to output Tensorboard hyperparams
            tf.summary.text('Model Info', hp, step=1)


class LSTM_Model(Base_Model):
    def __init__(
            self,
            # params: Dict = None,
            tb_logdir: str = '',
            random_state: int = None,
            shuffle: bool = True,
            val_frac: float = 0.2,
            **kwargs,
    ):
        """
        LSTM_Model constructor.

        Passes the params dict to the creation of the lstm model. And passes \
        **kwargs to class creation

        Args:
            params: Dict of mlp model parameters passed to \
                self.initialise_model
            tb_logdir: Folder path for tb logging, i.e. "CV" -> \
                "Tensorboard\\LSTM\\CV"
            **kwargs: Inputs for Base_Model init
        """
        super().__init__(**kwargs)
        # if params is None:
        #     params = {}
        # self.params = params
        self.seq_len = self.params.pop('seq_len', 10)

        self.tb_log_dir = os.path.join(self.tb_log_dir, 'LSTM', tb_logdir)

        if self.main_df is not None:
            self.pre_process(val_frac=val_frac,
                             random_state=random_state,
                             shuffle=shuffle,
                             )
            self.model = self.initialise_model(**self.params)
        else:
            print('Choose data file to import as main_df')

    def sequence_data(self, d: np.ndarray):
        """
        Applies window effect to change feature to include previous \
        self.seq_len features

        Args:
            d: array of features and targets

        Returns:
            tuple containing the windowed data and answer
        """
        seq_data = []
        seq_len = self.seq_len
        prev_points = deque(maxlen=seq_len)

        for i in d:
            # todo change this to adapt if multiple targets
            prev_points.append([n for n in i[:-1]])
            if len(prev_points) == seq_len:
                seq_data.append([np.array(prev_points), i[-1]])
        return seq_data

    def pre_process(
            self: Base_Model,
            val_frac: float = 0.2,
            random_state: int = None,
            shuffle: bool = True,
    ) -> Tuple[List[Any], List[Any]]:
        """
        Pre-process the data for training an LSTM model

            Split the data for the test into a training and validation set. \
            Then scale the data using a MinMax scaler, based off the training \
            data index. Then window the data, to preserve time. Then split \
            the data based on the saved indicies.

        Args:
            val_frac: Fraction of data points used within the validation set

        Returns:
            Two tuples which contain dataframes of the features and results \
            for the training and validation sets.

        """

        # 4* seqlen for removing overlap
        indx = np.arange(len(self.main_df))
        # save index and pos of the train test split
        train_i, test_i = train_test_split(indx,
                                           test_size=val_frac,
                                           shuffle=shuffle,
                                           random_state=random_state,
                                           )

        scaler = MinMaxScaler()
        m_df = self.main_df

        y = m_df[self.target].to_numpy()
        X = m_df.drop(columns=self.target).to_numpy()

        scaler.fit(X[train_i])
        X = scaler.transform(X)
        self.scaler = scaler

        Xy = np.column_stack((X, y))
        m_df = pd.DataFrame(Xy, columns=m_df.columns)

        # window the dataset
        m_df = self.sequence_data(m_df.values)
        m_df = pd.DataFrame(m_df, columns=['features', 'target'])
        m_df.index += self.seq_len - 1

        # index position of the end of each dataframe
        # todo need to change to get automatically
        # df_ends = [211, 374, 550, 708]
        df_ends = np.cumsum([207, 159, 172, 154])

        # try to remove overlapping
        # indicies of data to remove from model
        del_indx = list(range(0, (self.seq_len - 1)))
        indx = np.delete(indx, del_indx)

        for end in df_ends[:-1]:
            del_indx_overlap = list(range(end, (end + (self.seq_len - 1))))
            try:
                indx = np.delete(indx, del_indx_overlap)
            except IndexError:
                print('Overlapping sections between exps not removed!')
        
        # split data set indicies into train and test
        temp_train_i = [element for element in train_i
                        if element not in del_indx]
        temp_test_i = [element for element in test_i
                       if element not in del_indx]

        # separate the train and test datasets
        train = m_df.loc[temp_train_i, :]
        test = m_df.loc[temp_test_i, :]

        train_X = []
        train_y = []

        for X, y in train.values:
            train_X.append(X)
            train_y.append(y)

        test_X = []
        test_y = []

        for X, y in test.values:
            test_X.append(X)
            test_y.append(y)

        train_X = np.asarray(train_X)
        train_y = np.asarray(train_y)
        test_X = np.asarray(test_X)
        test_y = np.asarray(test_y)

        self.train_data = [train_X, train_y]
        self.val_data = [test_X, test_y]
        # print(f'Train data shape:\t{train_X.shape}')
        # print(f'Test data shape:\t{test_X.shape}')
        self._no_features = (self.train_data[0].shape[1:])
        return self.train_data, self.val_data

    @staticmethod
    def build_mod(
            no_features,
            dropout,
            no_layers,
            no_dense,
            no_nodes,
            activation,
            init_mode,
    ) -> Sequential:
        """
        Creates a vanilla MLP Keras Sequential Model.

        Returns a simple MLP model consisting of Dense and Dropout layers, \
        the number of which and sizes can be determined via function inputs.

        Args:
            no_features: No of input features calc from train data
            dropout: Dropout rate for every dropout layer
            no_layers: No of LSTM layers to include in the model
            no_dense: No of Dense layers to include after the LSTM layers
            no_nodes: No of neurons to include in each dense layer
            activation: The activation function for the neurons'
            init_mode: Initialisation method for the neuron weights

        Returns:
            model: Compiled Keras LSTM model

        """
        model = Sequential(name='LSTM_reg')
        model.add(LSTM(
            units=no_nodes,
            # activation=activation,
            input_shape=no_features,
            kernel_initializer=init_mode,
            return_sequences=True if no_layers > 1 else False,
            name='lstm1',
        ))
        model.add(Dropout(rate=dropout, name='dropout1'))

        i = 0
        for i in list(range(no_layers - 1)):
            model.add(LSTM(
                units=no_nodes,
                # activation=activation,
                kernel_initializer=init_mode,
                return_sequences=True if (i + 2) < no_layers else False,
                name=f'lstm{i + 2}',
            ))
            model.add(Dropout(rate=dropout, name=f'dropout{i + 2}'))

        for j in list(range(no_dense)):
            model.add(Dense(
                units=no_nodes,
                activation=activation,
                kernel_initializer=init_mode,
                name=f'dense{j + 1}',
                use_bias=True,
            ))
            model.add(Dropout(
                rate=dropout, name=f'dropout{no_layers + j + 1}'))
        model.add(Dense(1, name='output', activation='linear'))
        return model

    def initialise_model(
            self,
            dropout: float = 0.1,
            activation: str = 'relu',
            no_nodes: int = 32,
            no_layers: int = 2,
            no_dense: int = 1,
            init_mode: str = 'glorot_normal',
            epochs: int = 500,
            batch_size: int = 10,
            loss: str = 'mae',
            metrics: Union[list, tuple] = ('MSE',
                                           'MAE',
                                           KerasRegressor.r_squared
                                           ),
            optimizer: str = Adam,
            learning_rate: float = 0.001,
            verbose: int = 1,
            callbacks: List[Any] = None,
            **params,
    ) -> KerasRegressor:
        """
        Initialise a SciKeras Regression model with the given hyperparameters

        Uses the specified parameters to create a Scikeras KerasRegressor.\
        Allowing a Keras model to be produced easily with all the inputted\
        parameters.

        Args:
            dropout: Dropout rate of each dropout layer
            activation: Activation function for the neurons'
            no_nodes: No neurons in each Dense layer
            no_layers: No of LSTM and Dropout layers to create the model with
            no_dense: No of Dense layers after LSTM layers
            init_mode: Initialisation method for the weights of each neuron
            epochs: No of epochs to train the model with
            batch_size: No of samples for the model to train with before\
                 updating weights
            loss: Metric to optimise the model for.
            metrics: Metrics to track during training of the model
            optimizer: Optimizer function used to allow model to learn
            learning_rate: Learning rate of the optimizer
            verbose: Output type for the console
            callbacks: Callbacks to add into the Keras model.

        Returns: A KerasRegressor into the model attribute

        """

        # tensorboard set-up
        logdir = self.tb_log_dir
        t = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        layers = (no_layers + no_dense)
        self._run_name = os.path.join(logdir,
                                      f'LSTM-WLEN-{self.seq_len}-'
                                      f'E-{epochs}-'
                                      f'B-{batch_size}-'
                                      f'L-{np.full(layers, no_nodes)}-'
                                      f'D-{dropout}-'
                                      f'{t}')

        if callbacks is None:
            callbacks = []
        # Add in TQDM progress bar for fitting
        callbacks.append(tfa.callbacks.TQDMProgressBar(
            show_epoch_progress=False))

        if self._tb:
            # Add in tensorboard logging
            callbacks.append(tf.keras.callbacks.TensorBoard(
                log_dir=self._run_name, histogram_freq=1))
            tb_writer = tf.summary.create_file_writer(self._run_name)
            with tb_writer.as_default():
                hp_params = self.params
                hp_params['seq_len'] = self.seq_len
                hp_params.pop('callbacks', None)
                hp.hparams(
                    hp_params,
                    trial_id=self._run_name.split(self.tb_log_dir)[1][1:]
                )

        model = KerasRegressor(
            model=self.build_mod,
            model__no_features=self._no_features,
            model__dropout=dropout,
            model__activation=activation,
            model__no_nodes=no_nodes,
            model__no_layers=no_layers,
            model__no_dense=no_dense,
            model__init_mode=init_mode,
            epochs=epochs,
            batch_size=batch_size,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            optimizer__learning_rate=learning_rate,
            verbose=verbose,
            callbacks=callbacks,
        )
        return model

    def tb_model_desc(self, tb_wr):
        # Model.summary()
        lines = []
        self.model.model_.summary(print_fn=lines.append)

        dropout = self.model.model_.layers[1].get_config()['rate']
        try:
            no_layers = self.params['no_layers']
        except KeyError:
            no_layers = 2
        try:
            no_dense = self.params['no_dense']
        except KeyError:
            no_dense = 1
        layers = self.model.model_.get_config()['layers']
        nodes = [layer['config']['units'] for layer in layers
                 if layer['class_name'] in ('Dense', 'LSTM')]
        activation = layers[1]['config']['activation']
        opt = self.model.model_.optimizer.get_config()
        optimiser = opt['name']
        learning_rate = opt['learning_rate']

        hp = dedent(f"""
            ### Parameters:
            ___

            |Seq Len| Epochs | Batch Size | No Layers | No Dense | \
            No Neurons | Init Mode | Activation | Dropout | Loss | \
            Optimiser | Learning rate |
            |--------|--------|------------|-----------|------------|\
            -----------|------------|---------|------|-----------|\
            ---------------|-------|
            |{self.seq_len}|{self.model.epochs}|{self.model.batch_size}|\
            {no_layers}|{no_dense}|{nodes[:-1]}|\
            {self.model.model__init_mode}|{activation}|{dropout:.3f}|\
            {self.model.loss}|{optimiser}|{learning_rate:.3E}|

            """)

        with tb_wr.as_default():
            # Code to ouput Tensorboard model.summary
            # lines = '    ' + '\n    '.join(lines)
            # tf.summary.text('Model Info', lines, step=0)
            # Code to output Tensorboard hyperparams
            tf.summary.text('Model Info', hp, step=1)
