import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import time
import tqdm
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from src import config

HOME_DIR, BASE_DIR, CODE_DIR, TB_DIR, RMS_DATA_DIR = config.config_paths()

# Regression Models


class Base_Model(Model):
    def __init__(self,
                 inputData: np.ndarray | None,
                 targetData: np.ndarray | None,
                 modelParams: dict | None,
                 compileParams: dict | None = None,
                 tb: bool = True,
                 tbLogDir: str | None = None,
                 randomState: int | None = None,
                 shuffle: bool = True,
                 **kwargs,
                 ):
        super().__init__(**kwargs)

        self.runName: str | None = None
        self._tb: bool = tb
        self._randomState: int | None = randomState
        self._shuffle: bool = shuffle
        self._trainIdx: list[int] | None = None
        self._testIdx: list[int] | None = None

        # Initialse tensorboard directory
        self._tbLogDir: Path = Path(TB_DIR)
        if tbLogDir:
            self._tbLogDir = self._tbLogDir.joinpath(tbLogDir)

        # Initialse model parameter dict
        if modelParams:
            self.modelParams = modelParams
        else:
            self.modelParams = {}

        if compileParams:
            self.compileParams = compileParams
        else:
            self.compileParams = None

        # Initialise input and target data
        self._inputData: np.ndarray | None = inputData
        self._targetData: np.ndarray | None = targetData

        self._nFeatures = self._inputData.shape[1]
        self._nOutputs = self._targetData.shape[1]

        if isinstance(self._inputData, np.ndarray):
            if len(self._inputData.shape) == 1:
                self._inputData = self._inputData.reshape(-1, 1)

        if isinstance(self._targetData, np.ndarray):
            if len(self._targetData.shape) == 1:
                self._targetData = self._targetData.reshape(-1, 1)

    def summary(self, print_fn=None):
        x = Input(shape=(self.trainData[0].shape[1],))
        return Model(inputs=x, outputs=self.call(x)).summary(print_fn=print_fn)

    def score(self,
              x: np.ndarray | None = None,
              y: np.ndarray | None = None,
              printout: bool = True,
              ) -> dict:
        
        if x is None and y is None:
            x, y = self.testData
        elif x is None or y is None:
            raise ValueError(
                "Either both x and y must be provided or neither."
            )

        def _score(self, y_true, y_pred):
            metrics = []
            metrics.append(self.compileParams['loss'])
            if 'metrics' in self.compileParams:
                metrics.extend(self.compileParams['metrics'])

            metrics = [tf.keras.metrics.get(metric) for metric in metrics]
            sc = {metric.name: metric(y_true, y_pred).numpy()
                  for metric in metrics}
            return sc

        pred = self.predict(x)

        score_all = _score(self, y, pred)
        score_outputs = []
        if self._nOutputs == 1:
            score_outputs.append(
                _score(self, y, pred)
            )
        else:
            for i in range(self._nOutputs):
                score_outputs.append(
                    _score(self,
                           y[:, i].reshape(-1, 1),
                           pred[:, i].reshape(-1, 1),
                           )
                )

        if printout:
            if self._nOutputs == 1:
                print("Test Data:")
                for key, val in score_all.items():
                    print(f"\t{key}: {val:.4f}")

            else:
                print("Test Data - Combined Outputs:")
                for key, val in score_all.items():
                    print(f"\t{key}: {val:.4f}")

                print("\nTest Data - Individual Outputs:")
                for i, score in enumerate(score_outputs):
                    print(f"Output {i+1}:")
                    for key, val in score.items():
                        print(f"\t{key}: {val:.4f}")
            
        if self._tb:
            # todo add seperate scores for each output to tb
            tb_writer = tf.summary.create_file_writer(str(self._tbLogDir))
            md_scores = ('### Scores - Test Data\n'
                         '#### Combined Outputs\n'
                         '| Metric | Value |\n'
                         '|--------|-------|\n'
                         )
            for key, val in score_all.items():
                md_scores += f"| {key} | {val:.4f} |\n"
            
            if self._nOutputs > 1:
                md_scores += ('\n#### Individual Outputs\n'
                              '| Output | Metric | Value |\n'
                              '|--------|--------|-------|\n'
                              )
                for i, score in enumerate(score_outputs):
                    for key, val in score.items():
                        md_scores += f"| {i+1} | {key} | {val:.4f} |\n"

            with tb_writer.as_default():
                tf.summary.text('Scores', md_scores, step=1)

        return score_all, score_outputs

    def plot_loss(self):
        if self.history is not None:
            fig, ax = plt.subplots()
            ax.plot(self.history.history['loss'], label='train')
            if 'val_loss' in self.history.history.keys():
                ax.plot(self.history.history['val_loss'], label='validation')
            ax.legend()
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            return fig, ax

    def compile(self,
                optimiser: str = 'adam',
                loss: str = 'mse',
                metrics: list | None = None,
                ):
        # Check compile parameters and set defaults
        if 'optimiser' not in self.compileParams:
            self.compileParams['optimiser'] = optimiser
        if 'loss' not in self.compileParams:
            self.compileParams['loss'] = loss
        if 'metrics' not in self.compileParams:
            self.compileParams['metrics'] = ['root_mean_squared_error',
                                             'mean_absolute_error',
                                             'r2_score',
                                             ]

        super().compile(optimizer=self.compileParams['optimizer'],
                        loss=self.compileParams['loss'],
                        metrics=self.compileParams['metrics'],
                        )

    def fit(self,
            x: np.ndarray | None = None,
            y: np.ndarray | None = None,
            epochs: int = 100,
            batch_size: int = 32,
            validation_split: float = 0.1,
            verbose: int = 0,
            callbacks: list | None = None,
            **kwargs,
            ):
        if x is None and y is None:
            x, y = self.trainData

        if callbacks is None:
            callbacks = [tqdm.keras.TqdmCallback(verbose=verbose,
                                                 tqdm_class=tqdm.tqdm,
                                                 )]

        self.fitParams = {'epochs': epochs,
                          'batch_size': batch_size,
                          'val_split': validation_split,
                          }

        if self._tb:
            callbacks.append(
                tf.keras.callbacks.TensorBoard(log_dir=str(self._tbLogDir),
                                               histogram_freq=1,
                                               )
            )

        super().fit(x, y,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split,
                    verbose=verbose,
                    callbacks=callbacks,
                    )
        if self._tb:
            tb_writer = tf.summary.create_file_writer(str(self._tbLogDir))
            self._tb_model_desc(tb_writer)

    def _tb_model_desc(self, tb_writer):
        if isinstance(self.modelParams['nUnits'], int):
            units = np.fill(self.modelParams['nLayers'],
                            self.modelParams['nUnits'],
                            )
        else:
            units = self.modelParams['nUnits']

        params = {'epochs': self.fitParams['epochs'],
                  'batch_size': self.fitParams['batch_size'],
                  'units': units,
                  'init_mode': self.modelParams['initMode'],
                  'activation': self.modelParams['activation'],
                  'dropout': self.modelParams['dropout'],
                  'loss': self.compileParams['loss'],
                  'optimiser': self.compileParams['optimiser'],

                  }

        hp = ('### Model parameters:\n'
              '___\n'
              '| Parameter | Value |\n'
              '|-----------|-------|\n'
              )

        for key, val in params.items():
            hp += f"| {key} | {val} |\n"

        with tb_writer.as_default():
            tf.summary.text('Model Parameters:', hp, step=0)


class MLP_Model(Base_Model):
    #todo add CV to model
    #todo add docstrings to functions

    def __init__(self,
                 testFrac: float = 0.33,
                 **kwargs,
                 ):
        super().__init__(**kwargs)

        if self._inputData is None:
            raise ValueError("Data is required for pre-processing.")
        self.pre_process_data(testFrac=testFrac)

        self.get_model()

        if self.compileParams:
            self.compile()

        if self.runName is None:
            t = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            if isinstance(self.modelParams['nUnits'], int):
                units = np.fill(self.modelParams['nLayers'],
                                self.modelParams['nUnits'],
                                )
            else:
                units = self.modelParams['nUnits']

            if 'dropout' in self.modelParams:
                d = self.modelParams['dropout']
            else:
                d = 0.01

            self.runName = (f'MLP-{units}-D{d}-{t}')

        self._tbLogDir = self._tbLogDir.joinpath('MLP',
                                                 self.runName,
                                                 )

    @property
    def trainData(self):
        try:
            return (self._inputData[self._trainIdx],
                    self._targetData[self._trainIdx]
                    )
        except AttributeError:
            raise AttributeError(
                "Train data not found. Please run pre_process_data() method."
            )

    @property
    def testData(self):
        try:
            return (self._inputData[self._testIdx],
                    self._targetData[self._testIdx]
                    )
        except AttributeError:
            raise AttributeError(
                "Test data not found. Please run pre_process_data() method."
            )

    def pre_process_data(self,
                         testFrac: float = 0.33,
                         scaler: callable = MinMaxScaler,
                         ):
        idx = np.arange(self._inputData.shape[0])
        train_idx, test_idx = train_test_split(idx,
                                               test_size=testFrac,
                                               random_state=self._randomState,
                                               shuffle=self._shuffle,
                                               )
        self._trainIdx = train_idx
        self._testIdx = test_idx
        
        # Scale X input data only based on training data split
        self._scaler = scaler()
        self._scaler.fit(self._inputData[train_idx, :])

        self._inputData = self._scaler.transform(self._inputData)
        return self._trainIdx, self._testIdx
    
    def get_model(self):
        # Check model parameters and set defaults
        if 'nLayers' not in self.modelParams:
            self.modelParams['nLayers'] = 3
        if 'nUnits' not in self.modelParams:
            self.modelParams['nUnits'] = 32
        if 'activation' not in self.modelParams:
            self.modelParams['activation'] = 'relu'
        if 'initMode' not in self.modelParams:
            self.modelParams['initMode'] = 'glorot_uniform'
        if 'dropout' not in self.modelParams:
            self.modelParams['dropout'] = 0.01

        # Check no layers and units match
        if isinstance(self.modelParams['nUnits'], list):
            if len(self.modelParams['nUnits']) != self.modelParams['nLayers']:
                raise ValueError(
                    "Number of layers and units do not match."
                )
            nUnits = self.modelParams['nUnits']
        elif isinstance(self.modelParams['nUnits'], int):
            nUnits = [self.modelParams['nUnits']] * self.modelParams['nLayers']

        self.mlpLayers = []

        for nUnit in nUnits:
            self.mlpLayers.append(
                Dense(nUnit,
                      activation=self.modelParams['activation'],
                      kernel_initializer=self.modelParams['initMode'],
                      use_bias=True,
                      )
            )
            self.mlpLayers.append(Dropout(rate=self.modelParams['dropout']))

        self.mlpOutputs = []
        for _ in range(self._nOutputs):
            self.mlpOutputs.append(Dense(1, activation='linear'))

        # Linear output layer for regression
        self.mlpLayers.append(Dense(self._nOutputs,
                                    activation='linear',
                                    ))

    def call(self, inputs):
        # Forward pass
        for layer in self.mlpLayers:
            inputs = layer(inputs)
        return inputs


class LSTM_Model(Base_Model):
    #todo add LSTM model
    pass


if __name__ == "__main__":

    exampleData = np.random.rand(100, 10)

    mlp = MLP_Model(modelParams={'nLayers': 3,
                                 'nUnits': [64, 32, 32],
                                 'activation': 'relu',
                                 },
                    compileParams={'optimizer': 'adam',
                                   'loss': 'mse',
                                   'metrics': ['root_mean_squared_error',
                                               'mean_absolute_error',
                                               'r2_score',
                                               ],
                                   },
                    inputData=exampleData,
                    targetData=exampleData[:, :2],
                    tb=False,
                    )
    mlp.summary()
    mlp.compile(optimizer='adam',
                loss='mse',
                metrics=['mae'],
                )
    history = mlp.fit(mlp.trainData[0],
                      mlp.trainData[1],
                      epochs=2,
                      batch_size=32,
                      verbose=2,
                      )