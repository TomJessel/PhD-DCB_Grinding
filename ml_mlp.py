"""
@File    :   ml_mlp.py
@Author  :   Tom Jessel
@Contact :   jesselt@cardiff.ac.uk

@Creation Time    @Author    @Version    @Description
------------      -------    --------    -----------
26/10/2022 10:01   tomhj      1.0         Script to contain all code relating to MLP models
"""
from typing import Union, Any, Iterable

import warnings

import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scikeras.wrappers import KerasRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import resources


class Base_Model:
    def __init__(
            self,
            model=None,
            main_df: pd.DataFrame = None,
            target: Iterable = None,
    ):
        self.model = model
        self.main_df = main_df
        self.target = target
        self.train_data = pd.DataFrame,
        self.val_data = pd.DataFrame,

    @staticmethod
    def build_mod(
            # self,
            no_features=7,
            dropout=0.2,
            no_layers=2,
            no_nodes=32,
            activation='relu',
            init_mode='glorot_normal',
    ):
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
            optimizer: str = 'adam',
            learning_rate: float = 0.001,
            decay: float = 1e-6,
            verbose: int = 1,
            callbacks: Any = None,
    ):
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
        self.model = KerasRegressor(
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


class MLP_Model(Base_Model):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.pre_process()
        self.initialise_model()

    def pre_process(
            self,
            val_frac: float = 0.2,
    ):
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


if __name__ == "__main__":
    print('START')
    exp = resources.load('Test5')
    main_df = exp.features.drop(columns=['Runout', 'Form error']).drop([0, 1, 23, 24])

    mlp_reg = MLP_Model(main_df=main_df, target='Mean radius')
    mlp_reg.model.fit(mlp_reg.train_data[0].values, mlp_reg.train_data[1].values)

# todo add model scoring
# todo add tensorboard interface
# todo add cross-validation
# todo add MLP-window and LSTM classes
