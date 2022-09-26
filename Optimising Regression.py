#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@File    :   Optimising Regression.py
@Author  :   Tom Jessel
@Contact :   jesselt@cardiff.ac.uk

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
21/09/2022 09:23   tomhj      1.0         None
"""
# %%
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import KFold, train_test_split, RepeatedKFold
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from tensorflow import keras
from tqdm import tqdm

from Experiment import load

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_regression(meta, hidden_layer_sizes, dropout, init_mode='glorot_uniform'):
    n_features_in_ = meta['n_features_in_']
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(n_features_in_,)))
    for hidden_layer_size in hidden_layer_sizes:
        model.add(keras.layers.Dense(hidden_layer_size, activation='relu', kernel_initializer=init_mode))
        model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(1))
    return model


# %% Load and pre-process dataset
exp = load(file='Test 5')
dataframe = exp.features.drop(columns=['Runout', 'Form error', 'Freq 10 kHz', 'Freq 134 kHz'])
dataset = dataframe.values
X = dataset[:, :-1]
y = dataset[:, -1]

sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#%% Setup regression model

reg = KerasRegressor(
    model=get_regression,
    optimizer='adam',
    loss='mae',
    metrics=['MAE', 'MSE', 'mean_absolute_percentage_error'],
    hidden_layer_sizes=(30, 30),
    batch_size=8,
    dropout=0.2,
    epochs=300,
    verbose=0,
)
#%% GridsearchCV


def Model_gridsearch(
        model: KerasRegressor,
        Xdata: np.ndarray,
        ydata: np.ndarray,
        param_grid: Dict,
        cv: int = 5
) -> object:

    kfold = KFold(n_splits=cv, shuffle=True, random_state=0)

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        n_jobs=-1,
        cv=kfold,
        scoring='r2',
        verbose=True
    )
    gd_result = grid.fit(Xdata, ydata)

    print("Best: %f using %s" % (gd_result.best_score_, gd_result.best_params_))
    print('-'*82)
    means = gd_result.cv_results_['mean_test_score']
    stds = gd_result.cv_results_['std_test_score']
    params = gd_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return gd_result


# Batch Size and Epoch
batch_size = [5, 8, 10, 12, 15, 20]
epochs = [100, 200, 300, 400, 500]

# Number of neurons
model__hidden_layer_sizes = [(10,), (20,), (30,), (10, 10), (20, 20), (30, 30)]
model__dropout = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Neuron initiation mode
init_mode = ['uniform', 'lecun_uniform', 'normal', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']

# Learining rate and Momentum
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]

param_grid = dict(
    # model__init_mode=init_mode,
    model__hidden_layer_sizes=model__hidden_layer_sizes,
    model__dropout=model__dropout,
    batch_size=batch_size,
    epochs=epochs,
    optimizer__learning_rate=learn_rate,
)

grid_result = Model_gridsearch(model=reg, Xdata=X_train, ydata=y_train, param_grid=param_grid)
reg = grid_result.best_estimator_

# %% Train model and show validation history


def model_history(model: KerasRegressor, Xdata: np.ndarray, ydata: np.ndarray, cv: int = 5) -> KerasRegressor:
    kfold = KFold(n_splits=cv, shuffle=True, random_state=0)
    ind = []
    history = []

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

    fig = mean_hist.plot(
        subplots=[('MAE-train', 'MAE-val'), ('MSE-train', 'MSE-val'), ('MAPE-train', 'MAPE-val')],
        xlabel='Epoch',
        xlim=(-1, len(mean_hist)),
        title='Regression Model Learning History'
    )
    fig[0].set_ylabel('Mean\nAbsolute Error')
    fig[1].set_ylabel('Mean\nSquared Error')
    fig[2].set_ylabel('Mean Absolute\nPercentage Error')
    return model


reg = model_history(model=reg, Xdata=X_train, ydata=y_train, cv=5)

# %% Scoring with Cross Validation


def scoring_model(model: KerasRegressor, Xdata: np.ndarray, ydata: np.ndarray, cv: int = 10):
    # kfold = KFold(n_splits=cv, shuffle=True, random_state=0)
    kfold = RepeatedKFold(n_splits=cv, n_repeats=5, random_state=0)
    scoring = {
        'MAE': 'neg_mean_absolute_error',
        'MSE': 'neg_mean_squared_error',
        'r2': 'r2',
    }
    print('Scoring Model...\n')

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

    best_model.model_.summary()
    print('-' * 65)
    print(f'Model Training Scores:')
    print('-' * 65)
    print(f'Train time = {np.mean(scores_["fit_time"]):.2f} s')
    print(f'Predict time = {np.mean(scores_["score_time"]):.2f} s')
    print(f'MAE = {np.abs(np.mean(scores_["test_MAE"])) * 1000:.3f} um')
    print(f'MSE = {np.abs(np.mean(scores_["test_MSE"])) * 1_000_000:.3f} um^2')
    print(f'R^2 = {np.mean(scores_["test_r2"]):.3f}')
    print('-' * 65)
    return best_model, scores_


reg, train_scores = scoring_model(model=reg, Xdata=X_train, ydata=y_train)

# %% Evaluate model again test set

reg.fit(X_train, y_train)
y_pred = reg.predict(X_test, verbose=0)
test_score = {
    'MAE': mean_absolute_error(y_test, y_pred),
    'MSE': mean_squared_error(y_test, y_pred),
    'r2': r2_score(y_test, y_pred),
}
print('-' * 65)
print(f'Model Test Scores:')
print('-' * 65)
print(f'MAE = {np.abs(test_score["MAE"]) * 1000:.3f} um')
print(f'MSE = {np.abs(test_score["MSE"]) * 1_000_000:.3f} um^2')
print(f'R^2 = {np.mean(test_score["r2"]):.3f}')
print('-' * 65)

fig, ax = plt.subplots()
ax.plot(y_test, color='red', label='Real data')
ax.plot(y_pred, color='blue', ls='--', label='Predicted data')
ax.set_title('Model Predictions - Test Set')
ax.set_ylabel('Mean Radius (mm)')
ax.set_xlabel('Data Points')
ax.legend()
fig.show()

# %%
# todo use gridsearch to optimise hyperparameters
# todo use adaptive learning rates
