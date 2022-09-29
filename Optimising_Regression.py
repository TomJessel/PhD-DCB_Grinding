#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@File    :   Optimising_Regression.py
@Author  :   Tom Jessel
@Contact :   jesselt@cardiff.ac.uk

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
21/09/2022 09:23   tomhj      1.0         None
"""
# %%
import os
from typing import Dict, Any
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import KFold, train_test_split, RepeatedKFold
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from tensorflow import keras
from tqdm import tqdm

from Experiment import load

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s: %(name)s: %(levelname)s: %(message)s')
console_formatter = logging.Formatter('%(name)s: %(levelname)s: %(message)s')

file_handler = logging.FileHandler('ML.log')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# %% Load and pre-process dataset
logger.info('=' * 65)
exp = load(file='Test 5')
logger.info('Loaded Dateset')
dataframe = exp.features.drop(columns=['Runout', 'Form error', 'Freq 10 kHz', 'Freq 134 kHz'])
dataset = dataframe.values
X = dataset[:, :-1]
y = dataset[:, -1]

#%% Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
logger.info('Split Dataset into Train and Test')

#%% Scale data
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
# logger.info('Scaling Dataset')


#%% Setup regression model


def get_regression(meta, hidden_layer_sizes, dropout, init_mode='glorot_uniform'):
    n_features_in_ = meta['n_features_in_']
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(n_features_in_,)))
    for hidden_layer_size in hidden_layer_sizes:
        model.add(keras.layers.Dense(hidden_layer_size, activation='relu', kernel_initializer=init_mode))
        model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(1))
    return model


reg = KerasRegressor(
    model=get_regression,
    model__init_mode='glorot_normal',
    optimizer='adam',
    optimizer__learning_rate=0.001,
    loss='mae',
    metrics=['MAE', 'MSE', 'mean_absolute_percentage_error'],
    hidden_layer_sizes=(30, 30),
    batch_size=10,
    dropout=0.1,
    epochs=500,
    verbose=0,
)

#%% Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', reg),
])
#%% GridsearchCV


def Model_gridsearch(
        model: KerasRegressor,
        Xdata: np.ndarray,
        ydata: np.ndarray,
        param_grid: Dict,
        cv: int = 5
) -> object:

    kfold = KFold(n_splits=cv, shuffle=True)

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        n_jobs=-1,
        cv=kfold,
        scoring='r2',
        verbose=True
    )
    logger.info(f'___GridSearchCV___')
    gd_result = grid.fit(Xdata, ydata)

    logger.info(f'Best Estimator: {gd_result.best_score_:.6f}')
    logger.info(f'Using: {gd_result.best_params_}')
    # print("Best: %f using %s" % (gd_result.best_score_, gd_result.best_params_))
    # print('-'*82)
    # means = gd_result.cv_results_['mean_test_score']
    # stds = gd_result.cv_results_['std_test_score']
    # params = gd_result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f (%f) with: %r" % (mean, stdev, param))
    return gd_result


# Batch Size
batch_size = [5, 8, 10, 15, 25, 32]
# Epochs
epochs = [450, 500, 600]
# Number of neurons
model__hidden_layer_sizes = [(80, ), (30, 25), (30, 30)]
# Dropout
model__dropout = [0, 0.1, 0.3, 0.5]
# Neuron initiation mode
init_mode = ['lecun_uniform', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
# Optimizer
optimizer = ['adam', 'SGD', 'RMSprop', 'Adagrad', 'Adamax', 'Adadelta']
# Learining rate and Momentum
learn_rate = [0.0005, 0.0075, 0.001, 0.0025, 0.005, 0.01]
# Loss function
loss = ['mse', 'mae']

param_grid = dict(
    # model__init_mode=init_mode,
    # model__hidden_layer_sizes=model__hidden_layer_sizes,
    # model__dropout=model__dropout,
    # loss=loss,
    # batch_size=batch_size,
    epochs=epochs,
    # optimizer=optimizer,
    # optimizer__learning_rate=learn_rate,
)


# grid_result = Model_gridsearch(model=reg, Xdata=X_train, ydata=y_train, param_grid=param_grid, cv=10)
# reg = grid_result.best_estimator_


# %% Train model and show validation history


def model_history(model: KerasRegressor, Xdata: np.ndarray, ydata: np.ndarray, cv: int = 5) -> KerasRegressor:
    kfold = KFold(n_splits=cv, shuffle=True)
    ind = []
    history = []
    logger.info('___Model learning/validation history___')

    for train, val in tqdm(kfold.split(X=Xdata, y=ydata), total=kfold.get_n_splits(), desc='Model History'):
        model.fit(Xdata[train], ydata[train], validation_data=(Xdata[val], ydata[val]))
        hist = model.history_
        history.append(pd.DataFrame(hist).drop(columns=['loss', 'val_loss']).rename(columns={
            'mean_absolute_error': 'MAE-train',
            'mean_squared_error': 'MSE-train',
            'mean_absolute_percentage_error': 'MAPE-train',
            'val_mean_absolute_error': 'MAE-val',
            'val_mean_squared_error': 'MSE-val',
            'val_mean_absolute_percentage_error': 'MAPE-val',
        }))
        index = {
            'train_i': train,
            'test_i': val,
        }
        ind.append(index)
    history = pd.concat(history)

    # Graph History Results
    mean_hist = history.groupby(level=0).mean()

    ax = mean_hist.plot(
        subplots=[('MAE-train', 'MAE-val'), ('MSE-train', 'MSE-val'), ('MAPE-train', 'MAPE-val')],
        xlabel='Epoch',
        xlim=(-1, len(mean_hist)),
        title='Regression Model Learning History'
    )
    ax[0].set_ylabel('Mean\nAbsolute Error')
    ax[1].set_ylabel('Mean\nSquared Error')
    ax[2].set_ylabel('Mean Absolute\nPercentage Error')
    fig_ = ax[2].get_figure()
    png_name_ = f'ML learning history'
    fig_.savefig(png_name_, dpi=300)
    logger.info(f'Figure saved - {png_name_}')
    return model


# reg = model_history(model=reg, Xdata=X_train, ydata=y_train, cv=5)

# %% Scoring with Cross Validation


def scoring_model(model: Any, Xdata: np.ndarray, ydata: np.ndarray, cv: int = 10):
    # kfold = KFold(n_splits=cv, shuffle=True, random_state=0)
    kfold = RepeatedKFold(n_splits=cv, n_repeats=5)
    scoring = {
        'MAE': 'neg_mean_absolute_error',
        'MSE': 'neg_mean_squared_error',
        'r2': 'r2',
    }
    # print('Scoring Model...\n')
    logger.info(f'___Cross Validating Model - Repeated KFold___')

    scores_ = cross_validate(
        estimator=model,
        X=Xdata,
        y=ydata,
        cv=kfold,
        scoring=scoring,
        return_train_score=True,
        verbose=0,
        n_jobs=-1,
        return_estimator=True
    )
    ind = np.argmax(scores_['test_r2'])
    best_model = scores_['estimator'][ind]

    # logger.info(best_model.model_.summary(print_fn=logger.info))
    logger.info('-' * 65)
    logger.info(f'Model Training Scores:')
    logger.info('-' * 65)
    logger.info(f'Train time = {np.mean(scores_["fit_time"]):.2f} s')
    logger.info(f'Predict time = {np.mean(scores_["score_time"]):.2f} s')
    logger.info(f'MAE = {np.abs(np.mean(scores_["test_MAE"])) * 1000:.3f} um')
    logger.info(f'MSE = {np.abs(np.mean(scores_["test_MSE"])) * 1_000_000:.3f} um^2')
    logger.info(f'R^2 = {np.mean(scores_["test_r2"]):.3f}')
    logger.info('-' * 65)
    return best_model, scores_


reg, train_scores = scoring_model(model=pipe, Xdata=X_train, ydata=y_train)

# %% Evaluate model again test set

logger.info('Evaluating model with TEST set')
# reg.fit(X_train, y_train)
# y_pred = reg.predict(X_test, verbose=0)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test, verbose=0)

test_score = {
    'MAE': mean_absolute_error(y_test, y_pred),
    'MSE': mean_squared_error(y_test, y_pred),
    'r2': r2_score(y_test, y_pred),
}
logger.info('-' * 65)
logger.info(f'Model Test Scores:')
logger.info('-' * 65)
logger.info(f'MAE = {np.abs(test_score["MAE"]) * 1000:.3f} um')
logger.info(f'MSE = {np.abs(test_score["MSE"]) * 1_000_000:.3f} um^2')
logger.info(f'R^2 = {np.mean(test_score["r2"]):.3f}')
logger.info('-' * 65)

fig, ax = plt.subplots()
ax.plot(y_test, color='red', label='Real data')
ax.plot(y_pred, color='blue', ls='--', label='Predicted data')
ax.set_title('Model Predictions - Test Set')
ax.set_ylabel('Mean Radius (mm)')
ax.set_xlabel('Data Points')
ax.legend()
png_name = f'ML Predictions Test Set'
fig.savefig(png_name, dpi=300)
logger.info(f'Figure saved - {png_name}')
fig.show()
logger.info('=' * 65)

# todo properly implement pipelines instead of scaling then model
# todo change file into package that I can import to use in future
