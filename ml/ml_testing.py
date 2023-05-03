import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import resources
from resources.ml_mlp import LSTM_Model, MLP_Model
from resources.surf_meas import SurfMeasurements
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import multiprocessing
__spec__ = None


def pred_plot(y: np.ndarray, y_pred: np.ndarray, title: str = ''):
    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(y, y_pred)

    # limits of max radius
    # xmax = main_df['Mean radius'].values.max()
    # xmin = main_df['Mean radius'].values.min()
    # xmax = 0.68
    # xmin = 0.6
    
    # ax[0].set_xlim([xmin, xmax])
    # ax[0].set_ylim([xmin, xmax])

    lims = [
        np.min([ax[0].get_xlim(), ax[0].get_ylim()]),
        np.max([ax[0].get_xlim(), ax[0].get_ylim()]),
    ]

    ax[0].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax[0].set_aspect('equal')
    ax[0].set_xlabel('Actual Y / mm')
    ax[0].set_ylabel('Predicted Y /mm')
    ax[0].set_title(f'{title} - Predictions')
    
    diff = (y - y_pred) * 1000

    ax[1].hist(diff, bins=30)
    ax[1].set_xlabel('Prediction Error / um')
    ax[1].set_ylabel('No Occurances')
    ax[1].set_title(f'{title} - Histogram')

    fig.tight_layout()
    fig.show()


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    exp5 = resources.load('Test 5')
    exp7 = resources.load('Test 7')
    exp8 = resources.load('Test 8')
    exp9 = resources.load('Test 9')

    dfs = [exp5.features.drop([23, 24]),
           exp7.features,
           exp8.features,
           exp9.features]

    # # surface measurement calculations for Ra, ... for each exp
    # surf5 = SurfMeasurements(exp5.nc4.radius, ls=45, lc=1250)
    # surf7 = SurfMeasurements(exp7.nc4.radius, ls=45, lc=1250)
    # surf8 = SurfMeasurements(exp8.nc4.radius, ls=45, lc=1250)
    # surf9 = SurfMeasurements(exp9.nc4.radius, ls=45, lc=1250)
    # # combine surface and AE dataframes
    # surfs = [surf5, surf7, surf8, surf9]
    # data = zip(dfs, surfs)
    # dfs = [pd.concat([df, surf.meas_df], axis=1) for df, surf in data]
    
    # combine all data into one dataframe
    main_df = pd.concat(dfs)

    # remove unwanted columns from dataframe to prevent model cheating
    # whichever non AE fatures are not missing are being predicted
    main_df = main_df.drop(columns=[
        # 'Mean radius', 'Pa', 'Pq', 'Psk', 'Wa', 'Wq', 'Wsk', 'Ra', 'Rsk',
        'Runout', 'Form error', 'Peak radius', 'Radius diff'
        ]).drop([0, 1, 2, 3])
    main_df.reset_index(drop=True, inplace=True)
    # print(main_df.head())

    lstm_reg = LSTM_Model(feature_df=main_df,
                          target='Mean radius',
                          tb=True,
                          tb_logdir='early-stopping',
                          params={'loss': 'mse',
                                  'epochs': 1500,
                                  'no_layers': 3,
                                  'no_nodes': 128,
                                  'batch_size': 10,
                                  'init_mode': 'glorot_uniform',
                                  'dropout': 0.01,
                                  'seq_len': 10,
                                  'no_dense': 1,
                                  # 'callbacks': [
                                      # tf.keras.callbacks.EarlyStopping(
                                      #     monitor='loss',
                                      #     patience=100,
                                      #     mode='min',
                                      #     start_from_epoch=100,
                                      # ),
                                      # tf.keras.callbacks.ReduceLROnPlateau(
                                      #     monitor='val_loss',
                                      #     mode='min',
                                      #     factor=0.8,
                                      #     verbose=1,
                                      #     cooldown=50,
                                      #     patience=100,
                                      # ),
                                  # ],
                                  },
                          )

    # lstm_reg.cv(n_splits=10)
    lstm_reg.fit(validation_split=0.2, verbose=0)
    lstm_reg.score(plot_fig=False)

    y = lstm_reg.val_data[1]
    y_pred = lstm_reg.model.predict(lstm_reg.val_data[0], verbose=0)
    pred_plot(y, y_pred, 'LSTM')
