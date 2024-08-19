import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
from pathlib import Path
import time
from collections import deque
import tqdm
from tqdm.keras import TqdmCallback
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold

from src import config

HOME_DIR, BASE_DIR, CODE_DIR, TB_DIR, RMS_DATA_DIR = config.config_paths()


# Regression Models


class Base_Model(Model):
    def __init__(self,
                 inputData: np.ndarray | None,
                 targetData: np.ndarray | None,
                 modelParams: dict | None,
                 compileParams: dict | None = None,
                 fitParams: dict | None = None,
                 tb: bool = True,
                 tbLogDir: str | None = None,
                 randomState: int | None = None,
                 shuffle: bool = True,
                 scaler: callable = None,
                 **kwargs,
                 ):
        super().__init__(**kwargs)

        self.runName: str | None = None
        self._tb: bool = tb
        self._randomState: int | None = randomState
        self._shuffle: bool = shuffle
        self._trainIdx: list[int] | None = None
        self._testIdx: list[int] | None = None

        if scaler is None:
            self._scaler = MinMaxScaler
        else:
            self._scaler = scaler

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

        if fitParams:
            self.fitParams = fitParams
        else:
            self.fitParams = {}
        
        if 'epochs' not in self.fitParams:
            self.fitParams['epochs'] = 100
        if 'batch_size' not in self.fitParams:
            self.fitParams['batch_size'] = 32
        if 'validation_split' not in self.fitParams:
            self.fitParams['validation_split'] = 0.1
        if 'verbose' not in self.fitParams:
            self.fitParams['verbose'] = 0
        
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
        x = Input(shape=(self.trainData[0].shape[1:]))
        return Model(inputs=x, outputs=self.call(x)).summary(print_fn=print_fn)

    def score(self,
              x: np.ndarray | None = None,
              y: np.ndarray | None = None,
              model: Model | None = None,
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

            metrics = [keras.metrics.get(metric) for metric in metrics]
            sc = {metric.name: metric(y_true, y_pred).numpy()
                  for metric in metrics}
            return sc

        pred = self.predict(x, verbose=0)

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
        if self._history is not None:
            fig, ax = plt.subplots()
            ax.plot(self._history.history['loss'], label='train')
            if 'val_loss' in self._history.history.keys():
                ax.plot(self._history.history['val_loss'], label='validation')
            ax.legend()
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            return fig, ax
        return None

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
            if metrics is not None:
                self.compileParams['metrics'] = metrics
            else:
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
            epochs: int | None = None,
            batch_size: int | None = None,
            validation_split: float | None = None,
            verbose: int = 0,
            callbacks: list | None = None,
            **kwargs,
            ):
        if x is None and y is None:
            x, y = self.trainData

        if callbacks is None:
            help
            callbacks = [TqdmCallback(verbose=verbose,
                                      tqdm_class=tqdm.tqdm,
                                      )]

        if epochs is not None:
            self.fitParams['epochs'] = epochs
        if batch_size is not None:
            self.fitParams['batch_size'] = batch_size
        if validation_split is not None:
            self.fitParams['validation_split'] = validation_split
        if verbose != 0:
            self.fitParams['verbose'] = verbose

        if self._tb:
            callbacks.append(
                keras.callbacks.TensorBoard(log_dir=str(self._tbLogDir),
                                            histogram_freq=1,
                                            )
            )

        self._history = super().fit(x, y,
                                    callbacks=callbacks,
                                    **self.fitParams,
                                    )

        if self._tb:
            tb_writer = tf.summary.create_file_writer(str(self._tbLogDir))
            self._tb_model_desc(tb_writer)

    def _tb_model_desc(self, tb_writer):
        params = self._tb_desc_dict()

        hp = ('### Model parameters:\n'
              '___\n'
              '| Parameter | Value |\n'
              '|-----------|-------|\n'
              )

        for key, val in params.items():
            hp += f"| {key} | {val} |\n"

        with tb_writer.as_default():
            tf.summary.text('Model Parameters:', hp, step=0)

    def cv(self,
           nSplits: int = 5,
           nRepeats: int = 1,
           randomState: int | None = None,
           printOut: bool = True,
           ):
        # Use KFold or RepeatedKFold
        if nRepeats == 1:
            cv = KFold(n_splits=nSplits,
                       shuffle=True,
                       random_state=randomState,
                       )
        else:
            cv = RepeatedKFold(n_splits=nSplits,
                               n_repeats=nRepeats,
                               random_state=randomState,
                               )

        # split training data into cv sets
        cv_items = [(i, train, test) for i, (train, test)
                    in enumerate(cv.split(self.trainData[0]))]

        hold_tb = self._tb
        hold_valSplit = self.fitParams['validation_split']
        self._tb = False

        outputs = []
        for i, train_idx, test_idx in tqdm.tqdm(cv_items,
                                                desc='CV',
                                                total=len(cv_items),
                                                ):
            self.fit(x=self.trainData[0][train_idx],
                     y=self.trainData[1][train_idx],
                     validation_split=0,
                     epochs=self.fitParams['epochs'],
                     batch_size=self.fitParams['batch_size'],
                     callbacks=[]
                     )
            outputs.append(self.score(x=self.trainData[0][test_idx],
                                      y=self.trainData[1][test_idx],
                                      printout=False,
                                      )
                           )
            self._reinitialise()
        self._tb = hold_tb
        self.fitParams['validation_split'] = hold_valSplit

        scores = [output[0] for output in outputs]
        mean_scores = {key: np.mean([score[key] for score in scores])
                       for key in scores[0].keys()}
        std_scores = {key: np.std([score[key] for score in scores])
                      for key in scores[0].keys()}
        
        _cv_scores = {'mean': mean_scores,
                      'std': std_scores,
                      'all': scores,
                      }
        
        individualOutputScores = [output[1] for output in outputs]
        for i in range(len(individualOutputScores[0])):
            _cv_scores[f'output_{i+1}'] = {
                'mean': {key: np.mean([score[i][key]
                                       for score in individualOutputScores])
                         for key in individualOutputScores[0][i].keys()},
                'std': {key: np.std([score[i][key]
                                    for score in individualOutputScores])
                        for key in individualOutputScores[0][i].keys()},
                'all': [score[i] for score in individualOutputScores],
            }

        if printOut:
            print("\nCV Scores:")
            print("Overall:")
            for key, val in mean_scores.items():
                print(f"\t{key}: {val:.4f} (\u00B1 {std_scores[key]:.4f})")
            if len(individualOutputScores[0]) > 1:
                print("\nIndividual Outputs:")
                for i in range(len(individualOutputScores[0])):
                    print(f"Output {i+1}:")
                    for key, val in (_cv_scores[f'output_{i+1}']
                                     ['mean'].items()):
                        std = _cv_scores[f'output_{i+1}']['std'][key]
                        print(f"\t{key}: {val:.4f} (\u00B1 {std:.4f})")
        
        if self._tb:
            tb_writer = tf.summary.create_file_writer(str(self._tbLogDir))
            md_scores = ('### Scores - CV\n'
                         '#### Combined Outputs\n'
                         '| Metric | Mean | Std |\n'
                         '|--------|------|-----|\n'
                         )
            for key, val in mean_scores.items():
                md_scores += f"| {key} | {val:.4f} | {std_scores[key]:.4f} |\n"
            
            if len(individualOutputScores[0]) > 1:
                md_scores += ('\n#### Individual Outputs\n'
                              '| Output | Metric | Mean | Std |\n'
                              '|--------|--------|------|-----|\n'
                              )
                for i in range(len(individualOutputScores[0])):
                    for key, val in (_cv_scores[f'output_{i+1}']
                                     ['mean'].items()):
                        std = _cv_scores[f'output_{i+1}']['std'][key]
                        md_scores += (
                            f"| {i+1} | {key} | {val:.4f} | {std:.4f} |\n"
                        )

            with tb_writer.as_default():
                tf.summary.text('CV Scores', md_scores, step=1)

                for key, val in mean_scores.items():
                    tf.summary.scalar(f'CV Overall/CV {key}', val, step=1)
                step = tf.convert_to_tensor(0, dtype=tf.int64)
                for iter in scores:
                    for key, val in iter.items():
                        tf.summary.scalar(f'cv_iter/CV {key}', val, step=step)
                    step += 1

        return _cv_scores

    def _reinitialise(self):
        for lay in self.layers:
            if isinstance(lay, tf.keras.Model):
                self._reinitialise(lay)
                continue
            if hasattr(lay, "kernel_initializer"):
                lay.kernel.assign(lay.kernel_initializer(tf.shape(lay.kernel)))
            if hasattr(lay, "bias_initializer"):
                lay.bias.assign(lay.bias_initializer(tf.shape(lay.bias)))
            if hasattr(lay, "recurrent_initializer"):
                lay.recurrent_kernel.assign(
                    lay.recurrent_initializer(tf.shape(lay.recurrent_kernel))
                )


class MLP_Model(Base_Model):
    #todo add docstrings to functions

    def __init__(self,
                 testFrac: float = 0.2,
                 **kwargs,
                 ):
        super().__init__(**kwargs)

        if self._inputData is None:
            raise ValueError("Data is required for pre-processing.")
        self.pre_process_data(testFrac=testFrac,
                              scaler=self._scaler,
                              )

        self.get_model()

        if self.compileParams:
            self.compile()

        if self.runName is None:
            t = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            if isinstance(self.modelParams['nUnits'], int):
                units = (self.modelParams['nLayers'] *
                         [self.modelParams['nUnits']]
                         )
            else:
                units = self.modelParams['nUnits']

            if 'dropout' in self.modelParams:
                d = self.modelParams['dropout']
            else:
                d = 0.01

            self.runName = f'MLP-{units}-D{d}-{t}'

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

        # self.mlpOutputs = []
        # for _ in range(self._nOutputs):
        #     self.mlpOutputs.append(Dense(1, activation='linear'))

        # Linear output layer for regression
        self.mlpLayers.append(Dense(self._nOutputs,
                                    activation='linear',
                                    ))
        
    def call(self, inputs):
        # Forward pass
        for layer in self.mlpLayers:
            inputs = layer(inputs)
        return inputs

    def _tb_desc_dict(self):
        if isinstance(self.modelParams['nUnits'], int):
            units = self.modelParams['nLayers'] * [self.modelParams['nUnits']]
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
        return params


class LSTM_Model(Base_Model):
    def __init__(self,
                 testFrac: float = 0.33,
                 endPointsData: list[int] | None = None,
                 **kwargs,
                 ):
        super().__init__(**kwargs)
        
        if self._inputData is None:
            raise ValueError("Data is required for pre-processing.")

        if 'seqLen' not in self.modelParams:
            raise ValueError(
                "Sequence length must be provided in model params."
            )

        if endPointsData is None:
            raise ValueError(
                "End points of each dataset must be provided "
                "for proper sequencing without overlap."
            )
        assert np.cumsum(endPointsData)[-1] == self._inputData.shape[0], (
            "End points of each dataset must sum to "
            "the total number of data points."
        )

        self.pre_process_data(seqLen=self.modelParams['seqLen'],
                              testFrac=testFrac,
                              endPointsData=endPointsData,
                              scaler=self._scaler,
                              )

        self.get_model()

        if self.compileParams:
            self.compile()

        if self.runName is None:
            t = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            if isinstance(self.modelParams['nLSTMUnits'], int):
                lstmUnits = (self.modelParams['nLSTMLayers'] *
                             [self.modelParams['nLSTMUnits']]
                             )
            else:
                lstmUnits = self.modelParams['nLSTMUnits']
            
            if isinstance(self.modelParams['nDenseUnits'], int):
                denseUnits = (self.modelParams['nDenseLayers'] *
                              [self.modelParams['nDenseUnits']]
                              )
            else:
                denseUnits = self.modelParams['nDenseUnits']

            units = np.concatenate((lstmUnits, denseUnits))

            if 'dropout' in self.modelParams:
                d = self.modelParams['dropout']
            else:
                d = 0.01

            self.runName = f'LSTM-{units}-D{d}-{t}'

        self._tbLogDir = self._tbLogDir.joinpath('LSTM',
                                                 self.runName,
                                                 )

    @property
    def seqData(self):
        try:
            return (self._seqInputData[self._seqIdx, :, :],
                    self._seqTargetData[self._seqIdx, :],
                    )
        except AttributeError:
            raise AttributeError(
                "Sequence data not found. "
                "Please run pre_process_data() method."
            )

    @property
    def trainData(self):
        try:
            return (self._seqInputData[self._trainIdx, :, :],
                    self._seqTargetData[self._trainIdx, :]
                    )
        except AttributeError:
            raise AttributeError(
                "Train data not found. Please run pre_process_data() method."
            )

    @property
    def testData(self):
        try:
            return (self._seqInputData[self._testIdx, :, :],
                    self._seqTargetData[self._testIdx, :]
                    )
        except AttributeError:
            raise AttributeError(
                "Test data not found. Please run pre_process_data() method."
            )

    @staticmethod
    def sequence_data(inputData: np.ndarray,
                      targetData: np.ndarray,
                      seqLen: int,
                      ):
        seqDataInput = []
        seqDataOutput = []
        prev_points = deque(maxlen=seqLen)

        for inp, out in zip(inputData, targetData):
            prev_points.append(inp)
            if len(prev_points) == seqLen:
                seqDataInput.append(np.array(prev_points))
                seqDataOutput.append(out)
        return np.array(seqDataInput), np.array(seqDataOutput)

    def pre_process_data(self,
                         seqLen: int,
                         endPointsData: list[int],
                         testFrac: float = 0.33,
                         scaler: callable = MinMaxScaler,
                         ):
        # index starts from (seqLen -1) to allow for deque to fill
        idx = np.arange(self._inputData.shape[0] - (seqLen - 1))

        # Remove overlapping of exps data to prevent weird learning
        # If not removed it will include a sequence that increases in radius
        # end idx of each test dataset to remove overlap
        endPointsData = np.cumsum(endPointsData)
        delIdx = []
        for end in endPointsData[:-1]:
            end = end - seqLen + 1
            delIdx.extend(list(range(end, (end + (seqLen - 1)))))

        try:
            idx = np.delete(idx, delIdx)
        except IndexError:
            print('Overlapping sections between exps not removed!')

        train_idx, test_idx = train_test_split(idx,
                                               test_size=testFrac,
                                               random_state=self._randomState,
                                               shuffle=self._shuffle,
                                               )

        # Scale X input data only based on training data split
        self._scaler = scaler()
        self._scaler.fit(self._inputData[train_idx, :])
        self._inputData = self._scaler.transform(self._inputData)

        seqData = self.sequence_data(self._inputData,
                                     self._targetData,
                                     seqLen,
                                     )

        self._seqIdx = idx
        self._trainIdx = train_idx
        self._testIdx = test_idx

        self._seqInputData = seqData[0]
        self._seqTargetData = seqData[1]
        return seqData

    def get_model(self, ):
        if 'nLSTMLayers' not in self.modelParams:
            self.modelParams['nLSTMLayers'] = 3
        if 'nLSTMUnits' not in self.modelParams:
            self.modelParams['nLSTMUnits'] = 32
        if 'nDenseLayers' not in self.modelParams:
            self.modelParams['nDenseLayers'] = 1
        if 'nDenseUnits' not in self.modelParams:
            self.modelParams['nDenseUnits'] = 32
        if 'activation' not in self.modelParams:
            self.modelParams['activation'] = 'relu'
        if 'initMode' not in self.modelParams:
            self.modelParams['initMode'] = 'glorot_uniform'
        if 'dropout' not in self.modelParams:
            self.modelParams['dropout'] = 0.01
        if 'seqLen' not in self.modelParams:
            self.modelParams['seqLen'] = 15

        # Check no layers and units match
        if isinstance(self.modelParams['nLSTMUnits'], list):
            if (len(self.modelParams['nLSTMUnits']) !=
                    self.modelParams['nLSTMLayers']):
                raise ValueError(
                    "Number of layers and units do not match."
                )
            nLSTMUnits = self.modelParams['nLSTMUnits']
        elif isinstance(self.modelParams['nLSTMUnits'], int):
            nLSTMUnits = ([self.modelParams['nLSTMUnits']] *
                          self.modelParams['nLSTMLayers'])

        if isinstance(self.modelParams['nDenseUnits'], list):
            if (len(self.modelParams['nDenseUnits']) !=
                    self.modelParams['nDenseLayers']):
                raise ValueError(
                    "Number of layers and units do not match."
                )
            nDenseUnits = self.modelParams['nDenseUnits']
        elif isinstance(self.modelParams['nDenseUnits'], int):
            nDenseUnits = ([self.modelParams['nDenseUnits']] *
                           self.modelParams['nDenseLayers'])

        self.lstmLayers = []

        for i, nUnit in enumerate(nLSTMUnits):
            self.lstmLayers.append(
                LSTM(nUnit,
                     kernel_initializer=self.modelParams['initMode'],
                     return_sequences=(True if i < len(nLSTMUnits) - 1
                                       else False),
                     ),
            )
            self.lstmLayers.append(Dropout(rate=self.modelParams['dropout']))

        for nUnit in nDenseUnits:
            self.lstmLayers.append(
                Dense(nUnit,
                      activation=self.modelParams['activation'],
                      kernel_initializer=self.modelParams['initMode'],
                      use_bias=True,
                      )
            )
            self.lstmLayers.append(Dropout(rate=self.modelParams['dropout']))
        
        self.lstmLayers.append(Dense(self._nOutputs,
                                     activation='linear',
                                     ))
    
    def call(self, inputs):
        # Forward pass
        for layer in self.lstmLayers:
            inputs = layer(inputs)
        return inputs

    def _tb_desc_dict(self):
        if isinstance(self.modelParams['nLSTMUnits'], int):
            lstmUnits = (self.modelParams['nLSTMLayers'] *
                         [self.modelParams['nLSTMUnits']]
                         )
        else:
            lstmUnits = self.modelParams['nLSTMUnits']

        if isinstance(self.modelParams['nDenseUnits'], int):
            denseUnits = (self.modelParams['nDenseLayers'] *
                          [self.modelParams['nDenseUnits']]
                          )
        else:
            denseUnits = self.modelParams['nDenseUnits']

        params = {'epochs': self.fitParams['epochs'],
                  'batch_size': self.fitParams['batch_size'],
                  'lstm_units': lstmUnits,
                  'dense_units': denseUnits,
                  'init_mode': self.modelParams['initMode'],
                  'activation': self.modelParams['activation'],
                  'dropout': self.modelParams['dropout'],
                  'loss': self.compileParams['loss'],
                  'optimiser': self.compileParams['optimiser'],
                  }
        return params


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
                    inputData=exampleData[:, 2:],
                    targetData=exampleData[:, :2],
                    tb=False,
                    )
    mlp.summary()
    history = mlp.fit(mlp.trainData[0],
                      mlp.trainData[1],
                      epochs=2,
                      batch_size=32,
                      verbose=2,
                      )
