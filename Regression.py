#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@File    :   Regression.py   
@Author  :   Tom Jessel
@Contact :   jesselt@cardiff.ac.uk
 
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
20/09/2022 10:56   tomhj      1.0         None
"""

import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from autokeras import StructuredDataRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from tqdm import tqdm

from Experiment import load

exp = load(file='Test 5')
dataframe = exp.features.drop(columns=['Runout', 'Form error', 'Freq 10 kHz', 'Freq 134 kHz'])
dataset = dataframe.values
x = dataset[:, :-1]
y = dataset[:, -1]

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# search = StructuredDataRegressor(max_trials=20, loss='mean_absolute_error')
# search.fit(x=x_train, y=y_train, verbose=0)
# mae, _ = search.evaluate(x_test, y_test, verbose=0)
# print(f'MAE: {mae:.3f}')
# model = search.export_model()
# model.summary()
# model.sae('model_AE.h5')

sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)
x = sc.fit_transform(x)


def basic_model():
    m = Sequential()
    m.add(Dense(units=32, activation='relu', input_dim=5))
    m.add(Dense(units=32, activation='relu'))
    m.add(Dense(units=32, activation='relu'))
    m.add(Dense(units=32, activation='relu'))
    m.add(Dense(units=1))
    m.compile(optimizer='adam', loss='mean_squared_error')
    return m


# model = basic_model()
# history = model.fit(x_train, y_train, batch_size=10, epochs=1500, verbose=2)
kfold = KFold(n_splits=5, shuffle=True, random_state=0)
mae = []

for train, test in (pbar := tqdm(kfold.split(x, y), total=kfold.get_n_splits(), desc='K-Fold')):
    model = basic_model()
    model.fit(x[train], y[train], epochs=1250, batch_size=10, verbose=0)
    y_pred = model.predict(x[test], verbose=0)
    score = mean_absolute_error(y[test], y_pred)
    mae.append(score)
    pbar.set_description(f'K-Fold: MAE= {score*1000:.3f} um')
    # print(f'MAE: {score*1000:.3f} um')

mae = [m*1000 for m in mae]
print(f'AVG MAE: {np.mean(mae):.3f} um (+/- {np.std(mae):.2f}%)')

plt.plot(y[test], color='red', label='Real data')
plt.plot(y_pred, color='blue', label='Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()


# y_pred = model.predict(x_test)
# mae = mean_absolute_error(y_test, y_pred)
# print(f'MAE: {mae:.5f}')

