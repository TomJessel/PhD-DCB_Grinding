#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@File    :   learn_mlp_window.py   
@Author  :   Tom Jessel
@Contact :   jesselt@cardiff.ac.uk
 
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
10/10/2022 10:51   tomhj      1.0         None
"""
from collections import deque

from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import resources
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
from sklearn.model_selection import train_test_split

VAL_FRAC = 0.2
SEQ_LEN = 10

exp = resources.load('Test5')
main_df = exp.features.drop(columns=['Runout', 'Form error'])


def pre_process(df):
    tr_x, v_x, tr_y, v_y = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.2)

    scaler = MinMaxScaler()
    tr_x = scaler.fit_transform(tr_x)
    v_x = scaler.transform(v_x)

    tr_y = np.array(tr_y).reshape(-1, 1)
    v_y = np.array(v_y).reshape(-1, 1)

    data = [np.concatenate((tr_x, tr_y), axis=1), np.concatenate((v_x, v_y), axis=1)]

    def sequence_data(d):
        seq_data = []
        prev_points = deque(maxlen=SEQ_LEN)

        for i in d:
            prev_points.append([n for n in i[:-1]])
            if len(prev_points) == SEQ_LEN:
                seq_data.append([np.array(prev_points), i[-1]])

        x = []
        y = []

        for seq, target in seq_data:
            x.append(seq)
            y.append(target)
        return x, y

    tr_x, tr_y = sequence_data(data[0])
    v_x, v_y = sequence_data(data[1])
    return np.array(tr_x), np.array(v_x), np.array(tr_y), np.array(v_y)


train_X, val_X, train_y, val_y = pre_process(main_df)
print(f"Train data shape:\t{train_X.shape}\nVal data shape:\t{val_X.shape}")


n_input = train_X.shape[1] * train_X.shape[2]
train_X = train_X.reshape((train_X.shape[0], n_input))
val_X = val_X.reshape((val_X.shape[0], n_input))

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=n_input))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mae')

model.fit(train_X, train_y, epochs=500, verbose=1)

y_pred = model.predict(val_X)

fig, ax = plt.subplots()
ax.plot(val_y, color='red', label='Real data')
ax.plot(y_pred, color='blue', ls='--', label='Predicted data')
ax.set_title('Model Predictions - Test Set')
ax.set_ylabel('Mean Radius (mm)')
ax.set_xlabel('Data Points')
ax.legend()
plt.show()
#  TODO optimise the model config for MLP with window
