"""
@File    :   autoencoder.py
@Author  :   Tom Jessel
@Contact :   jesselt@cardiff.ac.uk

@Creation Time    @Author    @Version    @Description
------------      -------    --------    -----------
27/02/2023 11:23   tomhj      1.0         N/A
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model
import tensorflow_addons as tfa
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import resources


def mp_rms_process(fno):
    avg_size = 100000
    sig = exp.ae.readAE(fno)
    sig = pd.DataFrame(sig)
    sig = sig.pow(2).rolling(500000).mean().apply(np.sqrt, raw=True)
    sig = np.array(sig)[500000-1:]
    avg_sig = np.nanmean(np.pad(sig.astype(float),
                                ((0, avg_size - sig.size%avg_size), (0,0)),
                                mode='constant',
                                constant_values=np.NaN
                                )
                         .reshape(-1, avg_size), axis=1)
    return avg_sig

def mp_get_rms(fnos):
    with mp.Pool(processes=20, maxtasksperchild=1) as pool:
        rms = list(tqdm(pool.imap(
            mp_rms_process,
            fnos),
            total=len(fnos),
            desc='RMS averaging'
        ))
        pool.close()
        pool.join()
    return rms

def pred_and_score(mod, input_data):
    pred = mod.predict(input_data, verbose=0)
    mae = mean_absolute_error(input_data.T, pred.T, multioutput='raw_values')
    mse = mean_squared_error(input_data.T, pred.T, multioutput='raw_values')
    r2 = r2_score(input_data.T, pred.T, multioutput='raw_values')

    print(f'MAE: {np.mean(mae):.5f}')
    print(f'MSE: {np.mean(mse):.5f}')
    print(f'R2: {np.mean(r2):.5f}')
    return pred, (mae, mse, r2)

def pred_plot(input, pred, no):
    pred_input =input[no,:].reshape(-1, n_inputs)
    x_pred = pred[no,:].reshape(-1, n_inputs)

    pred_input = scaler.inverse_transform(pred_input)
    x_pred = scaler.inverse_transform(x_pred)

    mse = mean_squared_error(pred_input, x_pred)
    mae = mean_absolute_error(pred_input, x_pred)

    fig, ax = plt.subplots()
    ax.plot(pred_input.T, label='Real')
    ax.plot(x_pred.T, label='Predicition')
    ax.legend()
    ax.set_title(f'MAE: {mae:.4f} MSE: {mse:.4f}')
    return fig, ax

def scatter_scores(scores):
    # presumes scores is a tuple with (mae, mse, r2)
    fig, ax = plt.subplots(1, 3)
    ax[0].scatter(x=range(len(unseen_scores[0])),
                  y=unseen_scores[0],
                  color='b',
                  label='mae'
                  )
    ax[1].scatter(x=range(len(unseen_scores[1])),
                  y=unseen_scores[1],
                  color='g',
                  label='mse'
                  )
    ax[2].scatter(x=range(len(unseen_scores[2])),
                  y=unseen_scores[2],
                  color='r',
                  label='r2'
                  )
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    fig.suptitle(f'Scores')
    return fig, ax

if __name__ == '__main__':
    rms = []
    for i in ['Test 5', 'Test 7', 'Test 8', 'Test 9']:
        #load in test file
        exp = resources.load(i)
    
        # list of fnos to load in
        # fnos = range(len(exp.ae._files)
        fnos = list(range(0, 100))
    
        r = mp_get_rms(fnos)
        rms.extend(r)

    m = min([r.shape[0] for r in rms])
    rms = [r[:m] for r in rms]
    rms = np.array(rms)

    # split dataset
    x_train, x_test = train_test_split(rms, test_size=0.1, random_state=1)
    print(f'X train shape: {x_train.shape}')
    print(f'X test shape: {x_test.shape}')
    del rms

    # scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    ## AUTOENCODER MODEL
    n_inputs = x_train.shape[1]
    # define encoder
    visible = Input(shape=(n_inputs, ))
    e = Dense(64, activation='relu')(visible)
    e = BatchNormalization()(e)

    e = Dense(64, activation='relu')(e)
    e = BatchNormalization()(e)

    # define bottleneck
    n_bottleneck = 10
    bottleneck = Dense(n_bottleneck, activation='relu')(e)

    # define decoder
    d = Dense(64, activation='relu')(bottleneck)
    d = BatchNormalization()(d)

    d = Dense(64, activation='relu')(d)
    d = BatchNormalization()(d)

    # output layer
    output = Dense(n_inputs, activation='relu')(d)

    # define autoencoder model
    model = Model(inputs=visible, outputs=output)

    # compile autoencoder model
    model.compile(
        optimizer='adam',
        loss='mse',
        )

    history = model.fit(x_train, x_train,
                        epochs=500,
                        batch_size=16,
                        verbose=0,
                        validation_data=(x_test, x_test),
                        callbacks=tfa.callbacks.TQDMProgressBar(
                            show_epoch_progress=False),
                        )

    # calc metrics
    print(f'\nTraining Scores:')
    pred, scores = pred_and_score(model, x_test)
   
    # plot loss
    fig1, ax1 = plt.subplots()
    ax1.plot(history.history['loss'], label='train')
    ax1.plot(history.history['val_loss'], label='test')
    ax1.legend()

    # predict test values
    fig, ax = pred_plot(x_test, pred, 0)
    ax.set_title(f'Training Dataset - {ax.get_title()}')
    
    # TEST ON OTHER EXPERIMENTS

    for i in ['Test 5', 'Test 7', 'Test 8', 'Test 9']:
        exp = resources.load(i)
        fnos = range(len(exp.ae._files))

        print('\n')
        rms = mp_get_rms(fnos)

        rms = [r[:m] for r in rms]
        unseen_rms = np.array(rms)

        unseen_rms = scaler.transform(unseen_rms)

        # calc metrics
        print(f'\nUNSEEN EXP DATA - {i}:')
        unseen_pred, unseen_scores = pred_and_score(model, unseen_rms)

        fig, ax = scatter_scores(unseen_scores)
        fig.suptitle(f'{i.upper()} - {fig._suptitle.get_text()}')

        # plot prediction from unseen data
        fig, ax = pred_plot(unseen_rms, unseen_pred, 0)
        ax.set_title(f'UNSEEN DATA {i.upper()} - {ax.get_title()}')


    plt.show(block=False)
