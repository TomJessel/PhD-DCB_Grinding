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
#%%
import os

import matplotlib.pyplot as plt
import pandas as pd
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
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

#%%
exp = load(file='Test 5')
dataframe = exp.features.drop(columns=['Runout', 'Form error', 'Freq 10 kHz', 'Freq 134 kHz'])
dataset = dataframe.values
X = dataset[:, :-1]
y = dataset[:, -1]

sc = StandardScaler()
X = sc.fit_transform(X)

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
#%%
kfold = KFold(n_splits=5, shuffle=True, random_state=0)

#%%
ind = []
history = []

for train, test in tqdm(kfold.split(X=X, y=y), total=kfold.get_n_splits(), desc='K-Fold'):
    reg.fit(X[train], y[train], validation_data=(X[test], y[test]))
    hist = reg.history_
    history.append(pd.DataFrame(hist).drop(columns=['loss', 'val_loss']).rename(columns={
        'mean_absolute_error': 'MAE-train',
        'mean_squared_error': 'MSE-train',
        'mean_absolute_percentage_error': 'MAPE-train',
        'val_mean_absolute_error': 'MAE-test',
        'val_mean_squared_error': 'MSE-test',
        'val_mean_absolute_percentage_error': 'MAPE-test',
    }))
    index = {
        'train_i': train,
        'test_i': test,
    }
    ind.append(index)
history = pd.concat(history)

#%% Graph History Results
mean_hist = history.groupby(level=0).mean()

fig = mean_hist.plot(
        subplots=[('MAE-train', 'MAE-test'), ('MSE-train', 'MSE-test'), ('MAPE-train', 'MAPE-test')],
        xlabel='Epoch',
        xlim=(-1, len(mean_hist)),
        title='Regression Model Learning History'
        )
fig[0].set_ylabel('Mean\nAbsolute Error')
fig[1].set_ylabel('Mean\nSquared Error')
fig[2].set_ylabel('Mean Absolute\nPercentage Error')
# [f.autoscale(axis='y', tight=True) for f in fig]

#%%
scoring = {
    'MAE': 'neg_mean_absolute_error',
    'MSE': 'neg_mean_squared_error',
    'MAPE': 'neg_mean_absolute_percentage_error',
    'r2': 'r2',
}

scores = cross_validate(reg, X, y, cv=kfold, scoring=scoring, return_train_score=True, verbose=0, n_jobs=-1)

#%%
# Batch Size and Epoch
# batch_size = [5, 8, 10, 12, 15, 20]
# epochs = [1000, 1500, 2000]
# param_grid = dict(batch_size=batch_size, epochs=epochs)

# Number of neurons
# model__hidden_layer_sizes = [(10, 10, 10), (20, 20), (30, 30)]
# model__dropout = [0.2, 0.3, 0.4]
# param_grid = dict(model__hidden_layer_sizes=model__hidden_layer_sizes, model__dropout=model__dropout)

# Neuron initiation mode
# init_mode = ['uniform', 'lecun_uniform', 'normal', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
# param_grid = dict(model__init_mode=init_mode)

# Learining rate and Momentum
# learn_rate = [0.3, 0.4, 0.6, 0.8, 1]
# epochs = [1000, 1250, 1500, 2000]
# param_grid = dict(optimizer__learning_rate=learn_rate, epochs=epochs)

# grid = GridSearchCV(
#     estimator=reg,
#     param_grid=param_grid,
#     n_jobs=-1,
#     cv=3,
#     scoring='r2',
#     verbose=True
# )
# grid_result = grid.fit(X, y)
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# print('-'*82)
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
#
# model = grid_result.best_estimator_
# scores = cross_val_score(model, X, y, cv=10, scoring='r2')

# reg.fit(x_train, y_train)
# print(reg.model_.summary())
#
# y_pred = reg.predict(x_test)
#
# plt.plot(y_test, color='red', label='Real data')
# plt.plot(y_pred, color='blue', label='Predicted data')
# plt.title('Prediction')
# plt.legend()
# plt.show()
