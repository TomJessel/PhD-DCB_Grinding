"""
@File    :   ml_mlp.py
@Author  :   Tom Jessel
@Contact :   jesselt@cardiff.ac.uk

@Creation Time    @Author    @Version    @Description
------------      -------    --------    -----------
26/10/2022 10:01   tomhj      1.0         Script to contain all code relating to MLP models
"""
import multiprocessing
import random
import time
import warnings
from textwrap import dedent
from typing import List, Union, Iterable, Dict, Any

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm.auto import tqdm

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
import tensorflow_addons as tfa

tf.config.set_visible_devices([], 'GPU')
tf.get_logger().setLevel('ERROR')
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import resources
import platform

PLATFORM = platform.system()


class Base_Model:
    def __init__(
            self,
            target: Iterable = None,
            feature_df: pd.DataFrame = None,
            tb: bool = True,
    ):
        """
        Base_Model constructor.

        Args:
            target: Label of the feature to predict from the dataframe
            feature_df: Feature dataframe of which to train and predict
            tb: Option to record model progress and results in Tensorboard
        """
        self.model = None
        self.main_df = feature_df
        self.target = target
        self.train_data = pd.DataFrame
        self.val_data = pd.DataFrame
        self._tb = tb
        self._run_name = None
        self.params = {}

        # Tensorboard filename
        dirname = self.get_file_dir()
        self._tb_log_dir = os.path.join(dirname, 'Tensorboard')

        if self.target is None:
            raise AttributeError('There is no TARGET attribute set.')
        if self.main_df is None:
            raise AttributeError('There is no MAIN_DF attribute.')

    @staticmethod
    def get_file_dir():
        """
        Get path to the AE file in OneDrive folder

        Returns:
            file path to AE folder

        """
        import os
        import re

        dirname = os.path.dirname(__file__)

        regex_folder = re.compile("(tomje)")
        result = regex_folder.search(dirname)
        if PLATFORM == 'Windows':
            filename = dirname[:result.end()] + r"\Documents\PhD\AE"
        elif PLATFORM == 'Linux':
            filename = dirname[:result.end()] + r"/ml"
        filename = os.path.abspath(filename)
        return filename

    def pre_process(self):
        raise AttributeError('No assigned function to pre-process data for Base_Model')

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

        if self._tb:
            tb_writer = tf.summary.create_file_writer(self._run_name)
            self.tb_model_desc(tb_wr=tb_writer)

    def tb_model_desc(self, tb_wr):
        # Model.summary()
        lines = []
        self.model.model_.summary(print_fn=lines.append)

        dropout = self.model.model_.layers[1].get_config()['rate']
        layers = self.model.model_.get_config()['layers']
        nodes = [layer['config']['units'] for layer in layers if layer['class_name'] in ('Dense', 'LSTM')]
        no_layers = len(nodes) - 1
        activation = layers[1]['config']['activation']
        opt = self.model.model_.optimizer.get_config()
        optimiser = opt['name']
        learning_rate = opt['learning_rate']
        decay = opt['decay']

        hp = dedent(f"""       
            ### Parameters:   
            ___

            | Epochs | Batch Size | No Layers | No Neurons | Init Mode | Activation | Dropout | Loss | Optimiser |\
             Learning rate | Decay |
            |--------|------------|-----------|------------|-----------|------------|---------|------|-----------|\
            ---------------|-------|
            |{self.model.epochs}|{self.model.batch_size}|{no_layers}|{nodes[:-1]}|{self.model.model__init_mode}|\
            {activation}|{dropout:.3f}|{self.model.loss}|{optimiser}|{learning_rate:.3E}|{decay:.3E}|

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

        Use trained model to predict results on unseen validation data, and then calc metrics for scoring.
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
            print(f'Model Validation Scores:')
            print('-' * 65)
            print(f'MAE = {np.abs(_test_score["MAE"]) * 1000:.3f} um')
            print(f'MSE = {np.abs(_test_score["MSE"]) * 1_000_000:.3f} um^2')
            print(f'R^2 = {np.mean(_test_score["r2"]):.3f}')
            print('-' * 65)

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
                     | {_test_score['MAE'] * 1e3:.3f} | {_test_score['MSE'] * 1e6:.3f} | {_test_score['r2']:.3f} |  

                    ''')
            with tb_writer.as_default():
                tf.summary.text('Model Info', md_scores, step=2)

        return _test_score

    def initialise_model(self, verbose=1, **params) -> Any:
        self._run_name = f'{self._tb_log_dir}\\Base-{time.strftime("%Y%m%d-%H%M%S", time.localtime())}'
        pass

    def _cv_model(
            self,
            run_no,
            tr_index,
            te_index,
    ) -> List[Dict, KerasRegressor]:
        """
        multiprocessing worker function to cross validate one instance of the model

        Args:
            run_no: Iteration of the cv model
            tr_index: Index locations for the training data in the split
            te_index: Index locations for the testing data in the split

        Returns:
            Dict containing models scores
            Model that was scored on
        """
        model = self.initialise_model(verbose=0, **self.params)
        model.callbacks = None
        self._tb = False

        try:
            X = self.train_data[0].values
            y = self.train_data[1].values
        except AttributeError:
            X = self.train_data[0]
            y = self.train_data[1]

        model.fit(
            X=X[tr_index],
            y=y[tr_index],
            # validation_data=(self.train_data[0].values[tr_index], self.train_data[1].values[tr_index])
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

        If no input given to n_repeats, Kfold is used, if n_repeats is an int Repeated Kfold is used instead
        Args:
            n_splits: No splits in the CV
            n_repeats: No repeats to do with Repeated Kfold CV
            random_state: Random state of the CV, if no input random

        Returns:
            Dict of cv scores including mean values and stds

        """

        from sklearn.model_selection import KFold, RepeatedKFold
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

        cv_items = [(i, train, test) for i, (train, test) in enumerate(cv.split(X))]

        with multiprocessing.Pool(processes=20) as pool:
            outputs = list(tqdm(pool.imap(self._cv_model_star, cv_items),
                                total=len(cv_items),
                                desc='CV Model'
                                ))
            pool.close()
            pool.join()

        # with multiprocessing.Pool(processes=20) as pool:
        #     outputs = pool.starmap(self._cv_model, cv_items)

        # outputs = []
        # for cv_item in tqdm(cv_items):
        #     outputs.append(self._cv_model_star(cv_item))

        scores = [output[0] for output in outputs]
        # models = [output[1] for output in outputs]

        # bmod_r2 = max(scores, key=lambda x: x['r2'])['run_no']
        # bmod_MAE = min(scores, key=lambda x: x['MAE'])['run_no']
        # bmod_MSE = min(scores, key=lambda x: x['MSE'])['run_no']
        # # Whichever score is first is list will be default if no best over more than one score
        # bmod = [bmod_MAE, bmod_MSE, bmod_r2]
        #
        # model = models[max(bmod, key=bmod.count, default=None)]
        # self.model = model

        mean_MAE = np.mean([score['MAE'] for score in scores])
        mean_MSE = np.mean([score['MSE'] for score in scores])
        mean_r2 = np.mean([score['r2'] for score in scores])

        std_MAE = np.std([score['MAE'] for score in scores])
        std_MSE = np.std([score['MSE'] for score in scores])
        std_r2 = np.std([score['r2'] for score in scores])

        print('-' * 65)
        print(f'CV Training Scores:')
        print('-' * 65)
        print(f'MAE: {mean_MAE * 1_000:.3f} (\u00B1{std_MAE * 1_000: .3f}) \u00B5m')
        print(f'MSE: {mean_MSE * 1_000_000:.3f} (\u00B1{std_MSE * 1_000_000: .3f}) \u00B5m\u00B2')
        print(f'R^2: {mean_r2:.3f} (\u00B1{std_r2: .3f})')
        print('-' * 65)

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
                     | {mean_MAE * 1e3:.3f} | {mean_MSE * 1e6:.3f} | {mean_r2:.3f}  | 
                     | (\u00B1{std_MAE * 1_000: .3f}) | (\u00B1{std_MSE * 1_000_000: .3f}) | (\u00B1{std_r2: .3f}) |
                     
                    ''')
            with tb_writer.as_default():
                tf.summary.text('Model Info', md_scores, step=3)
        return _cv_score


class MLP_Model(Base_Model):
    def __init__(
            self,
            params: Dict = None,
            tb_logdir: str = '',
            **kwargs,
    ):
        """
        MLP_Model constructor.

        Passes the params dict to the creation of the mlp model. And passes **kwargs to class creation

        Args:
            params: Dict of mlp model parameters passed to self.initialise_model
            tb_logdir: Folder path for tb logging, i.e. "CV" -> "Tensorboard\\MLP\\CV"
            **kwargs: Inputs for Base_Model init
        """

        super().__init__(**kwargs)
        if params is None:
            params = {}
        self.params = params

        self._tb_log_dir = os.path.join(self._tb_log_dir, 'MLP', tb_logdir)

        if self.main_df is not None:
            self.pre_process()
            self.model = self.initialise_model(**params)
        else:
            print('Choose data file to import as main_df with ".load_testdata()')

    def pre_process(
            self: Base_Model,
            val_frac: float = 0.2,
    ) -> List[pd.DataFrame, pd.DataFrame]:
        """
        Pre-process the data for training an MLP model

            Split the data for the test into a training and validation set. Then scale the data using a MinMax scaler,
            based off the training data. Then save the splits into a tuple which contains the features and results
            separately.

        Args:
            val_frac: Fraction of data points used within the validation set

        Returns:
            Two tuples which contain dataframes of the features and results for the training and validation sets.

        """
        scaler = MinMaxScaler()

        train, test = train_test_split(self.main_df, test_size=val_frac)

        for col in self.main_df.columns:
            if col not in self.target:
                train[col] = scaler.fit_transform(train[[col]])
                test[col] = scaler.transform(test[[col]])
        train.dropna(inplace=True)
        test.dropna(inplace=True)

        self.train_data = (train.drop(columns=self.target), train[[self.target]])
        self.val_data = (test.drop(columns=self.target), test[[self.target]])
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

        Returns a simple MLP model consisting of Dense and Dropout layers, the number of which and sizes can be
        determined via function inputs.

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
            metrics: Union[list, tuple] = ('MSE', 'MAE', KerasRegressor.r_squared),
            optimizer: str = Adam,
            learning_rate: float = 0.001,
            decay: float = 1e-6,
            verbose: int = 1,
            callbacks: List[Any] = None,
            **params,
    ) -> KerasRegressor:
        """
        Initialise a SciKeras Regression model with the given hyperparameters

        Uses the specified parameters to create a Scikeras KerasRegressor. Allowing a Keras model to be produced easily
        with all the inputted parameters.

        Args:
            dropout: Dropout rate of each dropout layer
            activation: Activation function for the neurons'
            no_nodes: No neurons in each Dense layer
            no_layers: No of Dense and Dropout layers to create the model with
            init_mode: Initialisation method for the weights of each neuron
            epochs: No of epochs to train the model with
            batch_size: No of samples for the model to train with before updating weights
            loss: Metric to optimise the model for.
            metrics: Metrics to track during training of the model
            optimizer: Optimizer function used to allow model to learn
            learning_rate: Learning rate of the optimizer
            decay: Decay rate of the optimizer
            verbose: Output type for the console
            callbacks: Callbacks to add into the Keras model.

        Returns: A KerasRegressor into the model attribute

        """
        no_features = pd.DataFrame.to_numpy(self.train_data[0]).shape[1]

        # tensorboard set-up
        logdir = self._tb_log_dir
        self._run_name = os.path.join(logdir,
                                      f'MLP-E-{epochs}-B-{batch_size}-L{np.full(no_layers, no_nodes)}-D-{dropout}'
                                      f'-{time.strftime("%Y%m%d-%H%M%S", time.localtime())}')

        if callbacks is None:
            callbacks = []
        # Add in TQDM progress bar for fitting
        callbacks.append(tfa.callbacks.TQDMProgressBar(show_epoch_progress=False))

        if self._tb:
            # Add in tensorboard logging
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=self._run_name, histogram_freq=1))

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
            optimizer__decay=decay,
            verbose=verbose,
            callbacks=callbacks,
        )
        return model


class Linear_Model(Base_Model):
    def __init__(
            self,
            params: Dict = None,
            **kwargs,
    ):
        """
        Linear_Model constructor.

        Passes the params dict to the creation of the linear model. And passes **kwargs to class creation

        Args:
            params: Dict of linear model parameters passed to self.initialise_model
            **kwargs: Inputs for Base_Model init
        """
        super().__init__(**kwargs)
        if params is None:
            params = {}
        self.params = params

        if self.main_df is not None:
            self.pre_process()
            self.model = self.initialise_model(**params)
        else:
            print('Choose data file to import as main_df with ".load_testdata()')
        self._tb = False

    def pre_process(
            self,
            val_frac: float = 0.2,
    ) -> List[pd.DataFrame, pd.DataFrame]:
        """
        Function to pre-process the data for a linear model

            Uses MLP_Model pre_process function.

        Args:
            val_frac: Fraction of data points used within the validation set

        Returns:
            Two tuples which contain dataframes of the features and results for the training and validation sets.

        """
        func = MLP_Model.pre_process
        func(self)

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

        Use trained model to predict results on unseen validation data, and then calc metrics for scoring. And plot
        visualisation of the predictions and feature impacts.
        Args:
            model: ML model to score
            X: Inputs for predictions from unseen validation data set
            y: Corresponding outputs from validation data set
            plot_fig: Choice to plot the predictions plot
            print_score: Choice to print scores to terminal

        Returns:
            Dict containing the calculated scores
        """
        from sklearn.model_selection import cross_validate, RepeatedKFold

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
            print(f'Model Validation Scores:')
            print('-' * 65)
            print(f'MAE = {np.abs(_test_score["test_MAE"].mean()) * 1000:.3f} um')
            print(f'MSE = {np.abs(_test_score["test_MSE"].mean()) * 1_000_000:.3f} um^2')
            print(f'R^2 = {np.mean(_test_score["test_r2"].mean()):.3f}')
            print('-' * 65)

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


class MLP_Win_Model(Base_Model):
    def __init__(
            self,
            params: Dict = None,
            tb_logdir: str = '',
            **kwargs,
    ):
        """
        MLP_Win_Model constructor.

        Passes the params dict to the creation of the mlp win model. And passes **kwargs to class creation

        Args:
            params: Dict of mlp model parameters passed to self.initialise_model
            tb_logdir: Folder path for tb logging, i.e. "CV" -> "Tensorboard\\MLP\\CV"
            **kwargs: Inputs for Base_Model init
        """

        super().__init__(**kwargs)

        if params is None:
            params = {}
        # get sequnce len for window from param dict
        self.seq_len = params.pop('seq_len', 10)
        self.params = params

        self._tb_log_dir = os.path.join(self._tb_log_dir, 'MLP_WIN', tb_logdir)

        if self.main_df is not None:
            self.pre_process()
            self.model = self.initialise_model(**params)
        else:
            print('Choose data file to import as main_df with ".load_testdata()')

    def pre_process(
            self: Base_Model,
            val_frac: float = 0.2,
    ) -> List[np.ndarray, np.ndarray]:
        """
        Pre-process the data for training an MLP Win model

            Split the data for the test into a training and validation set. Then scale the data using a MinMax scaler,
            based off the training data index. Then window the data, to preserve time. Then split the data based on the
            saved indicies.

        Args:
            val_frac: Fraction of data points used within the validation set

        Returns:
            Two tuples which contain dataframes of the features and results for the training and validation sets.

        """

        from collections import deque
        def sequence_data(d: np.ndarray):
            """
            Applies window effect to change feature to include previous self.seq_len features

            Args:
                d: array of features and targets

            Returns:
                tuple containing the windowed data and answer
            """
            seq_data = []
            seq_len = self.seq_len
            prev_points = deque(maxlen=seq_len)

            for i in d:
                prev_points.append([n for n in i[:-1]])  # todo change this to adapt if multiple targets
                if len(prev_points) == seq_len:
                    seq_data.append([np.array(prev_points), i[-1]])
            return seq_data

        # 4* seqlen for removing overlap
        indx = np.arange(len(self.main_df))
        # save index and pos of the train test split
        train_i, test_i = train_test_split(indx, test_size=val_frac, shuffle=True)

        scaler = MinMaxScaler()
        main_df = self.main_df

        # scale the data transforming only on the training data and fitting on test data
        for col in self.main_df.columns:
            if col not in self.target:
                main_df[col][train_i] = scaler.fit_transform(main_df[col][train_i].values.reshape(-1, 1)).squeeze()
                main_df[col][test_i] = scaler.transform(main_df[col][test_i].values.reshape(-1, 1)).squeeze()

        # window the dataset
        main_df = sequence_data(main_df.values)

        # index position of the end of each dataframe need to change to get automatically
        df_ends = [211, 374, 550, 708]

        # try to remove overlapping
        # indicies of data to remove from model
        del_indx = list(range(0, self.seq_len)) + list(range(df_ends[0], (df_ends[0] + self.seq_len))) + list(
            range(df_ends[1], (df_ends[1] + self.seq_len))) + list(range(df_ends[2], (df_ends[2] + self.seq_len)))
        indx = np.delete(indx, del_indx)

        # split data set indicies into train and test
        temp_train_i = [element for element in train_i if element not in del_indx]
        temp_test_i = [element for element in test_i if element not in del_indx]

        # separate the train and test datasets
        train = [main_df[np.where(indx == j)[0][0]] for j in temp_train_i]
        test = [main_df[np.where(indx == j)[0][0]] for j in temp_test_i]

        train_X = []
        train_y = []

        for X, y in train:
            train_X.append(X)
            train_y.append(y)

        test_X = []
        test_y = []

        for X, y in test:
            test_X.append(X)
            test_y.append(y)

        train_X = np.asarray(train_X)
        train_y = np.asarray(train_y)
        test_X = np.asarray(test_X)
        test_y = np.asarray(test_y)

        self.train_data = [train_X, train_y]
        self.val_data = [test_X, test_y]
        # reshape feature data for MLP win model input
        self._no_features = self.train_data[0].shape[1] * self.train_data[0].shape[2]
        self.train_data[0] = self.train_data[0].reshape((self.train_data[0].shape[0], self._no_features))
        self.val_data[0] = self.val_data[0].reshape((self.val_data[0].shape[0], self._no_features))
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
            metrics: Union[list, tuple] = ('MSE', 'MAE', KerasRegressor.r_squared),
            optimizer: str = Adam,
            learning_rate: float = 0.001,
            decay: float = 1e-6,
            verbose: int = 1,
            callbacks: List[Any] = None,
            **params,
    ) -> KerasRegressor:
        """
        Initialise a SciKeras Regression model with the given hyperparameters

        Uses the specified parameters to create a Scikeras KerasRegressor. Allowing a Keras model to be produced easily
        with all the inputted parameters.

        Args:
            dropout: Dropout rate of each dropout layer
            activation: Activation function for the neurons'
            no_nodes: No neurons in each Dense layer
            no_layers: No of Dense and Dropout layers to create the model with
            init_mode: Initialisation method for the weights of each neuron
            epochs: No of epochs to train the model with
            batch_size: No of samples for the model to train with before updating weights
            loss: Metric to optimise the model for.
            metrics: Metrics to track during training of the model
            optimizer: Optimizer function used to allow model to learn
            learning_rate: Learning rate of the optimizer
            decay: Decay rate of the optimizer
            verbose: Output type for the console
            callbacks: Callbacks to add into the Keras model.

        Returns: A KerasRegressor into the model attribute

        """

        # tensorboard set-up
        logdir = self._tb_log_dir
        self._run_name = os.path.join(logdir,
                                      f'MLP_Win-WLEN-{self.seq_len}-E-{epochs}-B-{batch_size}-L-'
                                      f'{np.full(no_layers, no_nodes)}-D-{dropout}-'
                                      f'{time.strftime("%Y%m%d-%H%M%S", time.localtime())}')

        if callbacks is None:
            callbacks = []
        # Add in TQDM progress bar for fitting
        callbacks.append(tfa.callbacks.TQDMProgressBar(show_epoch_progress=False))

        if self._tb:
            # Add in tensorboard logging
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=self._run_name, histogram_freq=1))

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
            optimizer__decay=decay,
            verbose=verbose,
            callbacks=callbacks,
        )
        return model


if __name__ == "__main__":
    __spec__ = None
    multiprocessing.set_start_method("spawn")

    print('START')
    exp5 = resources.load('Test 5')
    exp7 = resources.load('Test 7')
    exp8 = resources.load('Test 8')
    exp9 = resources.load('Test 9')

    dfs = [exp5.features.drop([23, 24]), exp7.features, exp8.features, exp9.features]
    main_df = pd.concat(dfs)
    main_df = main_df.drop(columns=['Runout', 'Form error', 'Peak radius', 'Radius diff']).drop([0, 1, 2, 3])
    main_df.reset_index(drop=True, inplace=True)

    # MLP MODEL
    mlp_reg = MLP_Model(feature_df=main_df,
                        target='Mean radius',
                        tb=False,
                        tb_logdir='log test',
                        params={'loss': 'mse',
                                'epochs': 100,
                                'no_layers': 2,
                                },
                        )

    mlp_reg.cv(n_splits=10)
    mlp_reg.fit(validation_split=0.2, verbose=0)
    mlp_reg.score(plot_fig=False)

    # MLP WINDOW MODEL
    mlp_win_reg = MLP_Win_Model(feature_df=main_df,
                                target='Mean radius',
                                tb=False,
                                tb_logdir='',
                                params={'seq_len': 10,
                                        'loss': 'mae',
                                        'epochs': 100,
                                        'no_layers': 3,
                                        'no_nodes': 128,
                                        },
                                )
    mlp_win_reg.cv(n_splits=10)
    mlp_win_reg.fit(validation_split=0.2, verbose=0)
    mlp_win_reg.score(plot_fig=False)

    # MULTIPLE LINEAR MODEL
    lin_reg = Linear_Model(feature_df=main_df, target='Mean radius')
    lin_reg.fit()
    lin_reg.score()

    print('END')

# todo add LSTM classes
# todo try loss of r2 instead of MAE or MSE
# todo add logger compatibility to log progress and scores incase of TensorBoard failure
# todo change tb model desc for mlp-win model to include seqlen
# todo add model identifier when printing scores
# todo mlp_window for removing overlap needs to get positions of overlaps to work from end index of dfs
# todo add random state for pre-process and cv
# https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/
