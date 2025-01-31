import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import multiprocessing as mp
import time
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import tqdm
from tqdm.keras import TqdmCallback

import keras
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import tensorboard.plugins.hparams.api as hp

import src

HOME_DIR, BASE_DIR, CODE_DIR, TB_DIR, RMS_DATA_DIR = src.config.config_paths()


def _smooth(sig, win=11):
    """
    Smooth signal using a moving average filter.

    Replicates MATLAB's smooth function. (http://tinyurl.com/374kd3ny)

    Args:
        sig (np.array): Signal to smooth.
        win (int, optional): Window size. Defaults to 11.

    Returns:
        np.array: Smoothed signal.
    """
    out = np.convolve(sig, np.ones(win, dtype=int), 'valid') / win
    r = np.arange(1, win - 1, 2)
    start = np.cumsum(sig[:win - 1])[::2] / r
    stop = (np.cumsum(sig[:-win:-1])[::2] / r)[::-1]
    return np.concatenate((start, out, stop))


def add_freq_features(exps, freqs):
    """
    Add frequency features to experiments.

    Args:
        exps (list): List of experiments.
        freqs (list): List of frequencies to add.
    """
    for i, exp in enumerate(exps):
        f = np.array(exp.ae.fft[1000])
        # avg over 10kHz
        f = f.reshape(-1, 10).mean(axis=1).reshape(f.shape[0], -1)
        f = f.T
        for fr in freqs:
            fNew = np.concatenate(([np.NaN], f[fr]))
            exp.features[f'Freq {fr * 10} kHz'] = fNew
    return exps


def multiclass_categorise(df, doc, tol):
    x = np.arange(len(df))
    y = df['Probe diff'].values
    # is DOC within tolerance
    tol_bool = [0 if DOC - TOL <= yi <= DOC + TOL else 1 for yi in y]
    tol_bool = np.array(tol_bool, dtype=bool)

    # crossing points for transisitons between phases
    crossing = []
    for ix in x[tol_bool]:
        if ix - 1 not in x[tol_bool]:
            crossing.append(ix)
        elif ix + 1 not in x[tol_bool]:
            crossing.append(ix)
    # correct for wear in phase offset
    if crossing[1] > 10:
        crossing = [crossing[0], crossing[1]]
    else:
        crossing = [crossing[1], crossing[2]]

    # Multiclass classification
    # 0 -> Wear In
    # 1 -> Steady State
    # 2 -> Wear Out
    cat = np.ones(len(df), dtype=int)
    cat[:crossing[0] + 1] = 0
    cat[crossing[1]:] = 2
    df['Probe cat'] = cat
    return df


def drop_point_labels(df, drop_point):
    x = np.arange(len(df))
    tol_bool = [0 if xi < drop_point else 1 for xi in x]
    tol_bool = np.array(tol_bool).astype(bool)
    df['Probe cat'] = tol_bool
    return df


def create_maindf(dfs):
    # join all datasets
    main_df = pd.concat(dfs, ignore_index=True)
    # remove NaNs - mainly at start of test
    main_df = main_df.dropna()
    # reset dataframe index
    main_df = main_df.reset_index(drop=True)
    # remove unwanted features from dataframe
    main_df = main_df.drop(columns=['Runout',
                                    'Form error',
                                    'Peak radius',
                                    'Mean radius',
                                    'Radius diff',
                                    'Avg probe',
                                    'Probe diff',
                                    ])
    return main_df


def plot_scatter_cat(dfs, doc, tol):
    """
    Plot scatter of probe data with categorical labels.

    Args:
        dfs (list): List of dataframes, dfs must contain 'Probe cat' column.
        doc (float): Desired DOC.
        tol (float): Tolerance of DOC.
    
    """
    fig, ax = plt.subplots(4, 4,
                           figsize=(15, 8),
                           sharey='row',
                           constrained_layout=True,
                           )
    ax_pos = [[0, 0],
              [0, 1],
              [0, 2],
              [0, 3],
              [1, 0],
              [1, 1],
              [1, 2],
              [1, 3],
              [2, 0],
              [2, 1],
              [2, 2],
              [2, 3],
              [3, 0],
              [3, 1],
              [3, 2],
              [3, 3]]

    for i, (df, ax_idx) in enumerate(zip(dfs, ax_pos)):
        ax_row, ax_col = ax_idx

        x = np.arange(len(df))
        y = df['Probe diff']

        ax[ax_row, ax_col].plot(x, y, 'k--', alpha=0.5)
        ax[ax_row, ax_col].axhline(y=doc, color='g', ls='--', alpha=0.5)
        ax[ax_row, ax_col].axhline(y=doc + tol, color='r', ls='--', alpha=0.5)
        ax[ax_row, ax_col].axhline(y=doc - tol, color='r', ls='--', alpha=0.5)

        if 'Probe cat' in df.columns:
            cat = df['Probe cat']
            # c = ['g' if x == 0 else 'r' if x == 2 else 'C0' for x in cat]
            c = ['C0' if x == 0 else 'r' for x in cat]
        else:
            c = ['C0'] * len(df)
        ax[ax_row, ax_col].scatter(x, y, c=c, s=50, marker='x')
        ax[ax_row, ax_col].set_title(f'Test {i + 1}')
        
        if ax_col == 0:
            ax[ax_row, ax_col].set_ylabel('Probed DOC (mm)')
        if ax_row == 3:
            ax[ax_row, ax_col].set_xlabel('Cut No.')

    # # remove unused axes
    # for i in range(1, 3):
    #     for j in range(3, 7):
    #         if i == 0 and j == 0:
    #             continue
    #         ax[i, j].axis('off')
    return fig, ax


def plot_confusion_matrix(cm, std=None, classes=None, title=None, plt_ax=None):
    if plt_ax is None:
        fig, ax = plt.subplots()
    else:
        ax = plt_ax

    cm_div = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 3)
    if std is not None:
        std_div = np.around(
            std.astype('float') / cm.sum(axis=1)[:, np.newaxis], 3
        )

    ax.imshow(cm_div, cmap='Blues', alpha=0.9)
    for ix in range(cm.shape[0]):
        for jx in range(cm.shape[1]):
            if std is not None:
                txt = f'{cm_div[ix, jx]:.2f} +/- {std_div[ix, jx]:.2f}'
                txt += f'\n{round(cm[ix, jx])} +/- {round(std[ix, jx])}'
            else:
                txt = f'{cm_div[ix, jx]:.2f}\n({cm[ix, jx]})'
            ax.text(x=jx, y=ix,
                    s=txt,
                    ha='center',
                    va='center',
                    )
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[1]))
    if classes is not None:
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    if title is not None:
        ax.set_title(title)
    if plt_ax is None:
        return fig, ax
    else:
        return ax


def create_mlp_model(modelParams, nInputs: int, nOutputs: int):
    mlp_ = keras.models.Sequential()
    mlp_.add(keras.layers.Input(shape=(nInputs,)))

    if 'nLayers' not in modelParams:
        modelParams['nLayers'] = 2
    if 'nUnits' not in modelParams:
        modelParams['nUnits'] = 32
    if 'activation' not in modelParams:
        modelParams['activation'] = 'relu'
    if 'dropout' not in modelParams:
        modelParams['dropout'] = 0.01
    if 'initMode' not in modelParams:
        modelParams['initMode'] = 'glorot_uniform'
    if 'kernelReg' not in modelParams:
        modelParams['kernelReg'] = None
    
    if isinstance(modelParams['nUnits'], list):
        if len(modelParams['nUnits']) != modelParams['nLayers']:
            raise ValueError(
                "Number of layers and units do not match."
            )
        nUnits = modelParams['nUnits']
    elif isinstance(modelParams['nUnits'], int):
        nUnits = [modelParams['nUnits']] * modelParams['nLayers']

    for nUnit in nUnits:
        mlp_.add(
            keras.layers.Dense(nUnit,
                               activation=modelParams['activation'],
                               kernel_initializer=modelParams['initMode'],
                               use_bias=True,
                               kernel_regularizer=modelParams['kernelReg'],
                               )
        )
        mlp_.add(keras.layers.Dropout(rate=modelParams['dropout']))

    # Linear output layer for regression
    mlp_.add(keras.layers.Dense(nOutputs,
                                activation='softmax',
                                ))
    return mlp_


def _cv_model(model,
              tr_idx,
              va_idx,
              cvData,
              compileParams,
              fitParams,
              ):
    scaler = MinMaxScaler().fit(cvData[0][tr_idx])
    cvData[0] = scaler.transform(cvData[0])
    model.compile(optimizer=compileParams['optimizer'],
                  loss=compileParams['loss'],
                  metrics=compileParams['metrics'],
                  )
    model.fit(cvData[0][tr_idx],
              cvData[1][tr_idx],
              epochs=fitParams['epochs'],
              batch_size=fitParams['batch_size'],
              validation_data=(cvData[0][va_idx],
                               cvData[1][va_idx]
                               ),
              verbose=0,
              )
    sc = model.evaluate(cvData[0][va_idx],
                        cvData[1][va_idx],
                        return_dict=True,
                        batch_size=32,
                        verbose=0,
                        )
    y_pred = model.predict(cvData[0], verbose=0)
    y_pred_class = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(np.argmax(cvData[1], axis=1),
                          y_pred_class,
                          )
    return sc, cm


def _cv_model_star(args):
    return _cv_model(*args)


def cv_model(cvNSplits,
             cvNRepeats,
             cvData,
             modelParams,
             compileParams,
             fitParams,
             printout=True,
             TB=False,
             tbLogDir=None,
             random_state=None,
             ):
    if cvNRepeats > 1:
        cv = RepeatedStratifiedKFold(n_splits=cvNSplits,
                                     n_repeats=cvNRepeats,
                                     random_state=random_state,
                                     )
    else:
        cv = StratifiedKFold(n_splits=cvNSplits,
                             shuffle=True,
                             random_state=random_state,
                             )

    mlp_cv_models = [create_mlp_model(modelParams,
                                      nInputs=cvData[0].shape[1],
                                      nOutputs=cvData[1].shape[1]
                                      )
                     for _ in range(cv.get_n_splits())]
    cv_items = []
    for i, (cv_tr_idx, cv_va_idx) in enumerate(
        cv.split(X=np.zeros(len(idx_train)),
                 y=main_df.iloc[:, -1].copy()[idx_train],
                 )):
        cv_items.append((mlp_cv_models[i],
                         cv_tr_idx,
                         cv_va_idx,
                         cvData,
                         compileParams,
                         fitParams,
                         ))

    with mp.Pool() as pool:
        outputs = list(tqdm.tqdm(pool.imap(_cv_model_star,
                                           cv_items,
                                           chunksize=1,
                                           ),
                                 total=len(cv_items),
                                 desc='CV Model',
                                 ))
        pool.close()
        pool.join()

    cv_scores = [output[0] for output in outputs]
    cv_cm = [output[1] for output in outputs]

    # Average CV scores
    sc = {}
    for key in cv_scores[0].keys():
        if key == 'loss':
            continue
        if key == 'f1_score':
            sc[key] = [cv[key].numpy() for cv in cv_scores]
            sc[key] = np.array(sc[key])
        else:
            sc[key] = [cv[key] for cv in cv_scores]
            sc[key] = np.array(sc[key])

    cv_sc_mean = {key: np.mean(val, axis=0) for key, val in sc.items()}
    cv_sc_std = {key: np.std(val, axis=0) for key, val in sc.items()}
    # Average CV confusion matrix
    cm_shape = np.shape(cv_cm[0])
    cv_cm = np.concatenate([cm.reshape(-1, 1) for cm in cv_cm], axis=1)
    cv_cm_mean = np.mean(cv_cm, axis=1).reshape(cm_shape)
    cv_cm_std = np.std(cv_cm, axis=1).reshape(cm_shape)

    cv_cm_mean_div = np.around(
        cv_cm_mean.astype('float') / cv_cm_mean.sum(axis=1)[:, np.newaxis], 3
    )

    if printout:
        print('\nCross-Validation Evaluation:')
        for key, val in cv_sc_mean.items():
            print(f'{key.capitalize()}: {val} +/- {cv_sc_std[key]}')
        print('Cross-Validation Confusion Matrix:')
        print(cv_cm_mean_div)
        print()
        fig, ax = plot_confusion_matrix(cv_cm_mean,
                                        std=cv_cm_std,
                                        # classes=['WI', 'SS', 'WO'],
                                        classes=['SS', 'WO'],
                                        title='Cross-Validation',
                                        )
    
    if TB:
        tb_writer = tf.summary.create_file_writer(str(tbLogDir))
        md_scores = (
            f'### Scores - CV {cvNSplits}S{cvNRepeats}R\n'
            '| Metric | Mean | Std |\n'
            '|--------|------|-----|\n'
        )
        for key, val in cv_sc_mean.items():
            md_scores += f"| {key} | {val:.4f} | {cv_sc_std[key]:.4f} |\n"

        md_cm = (
            f'### Confusion Matrix - CV {cvNSplits}S{cvNRepeats}R\n'
            f'| True\Predicted | WI | SS | WO |\n'
            f'|----------------|----|----|----|\n'
            f'| WI | {cv_cm_mean.iloc[0, 0]:.4f} +- {cv_cm_std.iloc[0, 0]:.4f}'
            f' | {cv_cm_mean.iloc[0, 1]:.4f} +- {cv_cm_std.iloc[0, 1]:.4f}'
            f' | {cv_cm_mean.iloc[0, 2]:.4f} +- {cv_cm_std.iloc[0, 2]:.4f} |\n'
            f'| SS | {cv_cm_mean.iloc[1, 0]:.4f} +- {cv_cm_std.iloc[1, 0]:.4f}'
            f' | {cv_cm_mean.iloc[1, 1]:.4f} +- {cv_cm_std.iloc[1, 1]:.4f}'
            f' | {cv_cm_mean.iloc[1, 2]:.4f} +- {cv_cm_std.iloc[1, 2]:.4f} |\n'
            f'| WO | {cv_cm_mean.iloc[2, 0]:.4f} +- {cv_cm_std.iloc[2, 0]:.4f}'
            f' | {cv_cm_mean.iloc[2, 1]:.4f} +- {cv_cm_std.iloc[2, 1]:.4f}'
            f' | {cv_cm_mean.iloc[2, 2]:.4f} +- {cv_cm_std.iloc[2, 2]:.4f} |\n'
        )
        with tb_writer.as_default():
            tf.summary.text('Cross-Validation Scores:', md_scores, step=0)
            tf.summary.text('Cross-Validation Scores:', md_cm, step=1)
    return (cv_sc_mean, cv_sc_std), (cv_cm_mean, cv_cm_std)


def _tb_model_desc(params, tb_writer):

    hp = ('### Model parameters:\n'
          '___\n'
          '| Parameter | Value |\n'
          '|-----------|-------|\n'
          )

    for key, val in params.items():
        hp += f"| {key} | {val} |\n"

    with tb_writer.as_default():
        tf.summary.text('Model Parameters:', hp, step=0)


def _mlp_tb_desc_dict(modelParams, compileParams, fitParams):
    if isinstance(modelParams['nUnits'], int):
        units = modelParams['nLayers'] * [modelParams['nUnits']]
    else:
        units = modelParams['nUnits']

    params = {'epochs': fitParams['epochs'],
              'batch_size': fitParams['batch_size'],
              'units': units,
              'init_mode': modelParams['initMode'],
              'activation': modelParams['activation'],
              'dropout': modelParams['dropout'],
              'loss': compileParams['loss'],
              'optimizer': compileParams['optimizer'],
              }
    return params


def model_evaluate(model,
                   datasets,
                   idx_test,
                   printout=True,
                   tb=False,
                   tbLogDir=None,
                   ):
    sc = model.evaluate(datasets[0][idx_test],
                        datasets[1][idx_test],
                        return_dict=True,
                        batch_size=32,
                        verbose=0,
                        )
    # Predictions
    y_pred = model.predict(datasets[0], verbose=0)
    y_pred_class = np.argmax(y_pred, axis=1)

    # confusion matrix of test train and whole dataset

    cm = confusion_matrix(np.argmax(datasets[1], axis=1),
                          y_pred_class,
                          )

    cm_norm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 3)

    if printout:
        print('Test Set Evaluation:')
        for key, val in sc.items():
            if key == 'loss':
                continue
            try:
                print(f'{key.capitalize()}: {val:.4f}')
            except TypeError:
                print(f'{key.capitalize()}: {val}')
        print('Test Set Confusion Matrix:')
        print(cm_norm)
        fig, ax = plot_confusion_matrix(cm,
                                        # classes=['WI', 'SS', 'WO'],
                                        classes=['SS', 'WO'],
                                        title='Test Set',
                                        )
    
    if tb:
        tb_writer = tf.summary.create_file_writer(str(tbLogDir))
        md_scores = (
            '### Scores - Test Dataset\n'
            '| Metric | Score |\n'
            '|--------|-------|\n'
        )
        for key, val in sc.items():
            md_scores += f"| {key} | {val:.4f} |\n"

        md_cm = (
            '### Confusion Matrix - Test Dataset\n'
            f'| True\Predicted | WI | SS | WO |\n'
            f'|----------------|----|----|----|\n'
            f'| WI | {cm.iloc[0, 0]:.4f} | {cm.iloc[0, 1]:.4f} '
            f'| {cm.iloc[0, 2]:.4f} |\n'
            f'| SS | {cm.iloc[1, 0]:.4f} | {cm.iloc[1, 1]:.4f} '
            f'| {cm.iloc[1, 2]:.4f} |\n'
            f'| WO | {cm.iloc[2, 0]:.4f} | {cm.iloc[2, 1]:.4f} '
            f'| {cm.iloc[2, 2]:.4f} |\n'
        )

        with tb_writer.as_default():
            tf.summary.text('Test Dataset Scores:', md_scores, step=0)
            tf.summary.text('Test Dataset Scores:', md_cm, step=1)
    return sc, cm


if __name__ == "__main__":
    # Parameters
    DOC = 0.03          # Radial Depth of Cut (mm)
    TOL = 0.0015        # Tolerance of individual cuts (mm)
    OVERALL_TOL = 0.02  # Tolerance of cumulative cuts (mm)

    SMOOTH_WIN = 11     # Smoothing window size

    RANDOM_STATE = 42

    TB = False
    TB_LOG_DIR = TB_DIR / 'probeClassification/FailPoint/MLP'

    FITPARAMS = {
        'epochs': 1000,
        'batch_size': 128,
        'verbose': 0,
    }

    MODELPARAMS = {
        'modelFunc': create_mlp_model,
        'nLayers': 3,
        'nUnits': [16, 16, 16],
        'activation': 'relu',
        'dropout': 0.01,
        'initMode': 'glorot_uniform',
        'kernelReg': keras.regularizers.l2(0.001),
    }

    COMPILEPARAMS = {
        'loss': 'categorical_crossentropy',
        'optimizer': 'adam',
        'metrics': ['accuracy', 'recall', 'precision', 'auc'],
    }

    CV_NSPLITS = 5
    CV_NREPEATS = 5

    t = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    if isinstance(MODELPARAMS['nUnits'], int):
        units = (MODELPARAMS['nLayers'] *
                 [MODELPARAMS['nUnits']]
                 )
    else:
        units = MODELPARAMS['nUnits']

    if 'dropout' in MODELPARAMS:
        d = MODELPARAMS['dropout']
    else:
        d = 0.01
    run_name = f'MLP-{units}-D{d}-{t}'
    TB_LOG_DIR = TB_LOG_DIR / run_name

    #* Load Datasets
    expLabels = [
        'Test 11',
        'Test 14',
        'Test 15',
        'Test 16',
        'Test 17',
        'Test 18',
        'Test 19',
        'Test 21',
        'Test 22',
        'Test 23',
        'Test 24',
        'Test 25',
        'Test 26',
        'Test 28',
        'Test 30',
        'Test 31',
        'Test 32',
        'Test 34',
    ]

    print('Loading Datsets...')
    exps = [src.load(label) for label in expLabels]

    # Add in additional freq features
    freq = [35, 90]
    exps = add_freq_features(exps, freq)

    dfs = [exp.features.copy() for exp in exps]

    # Smooth probe data
    for df in dfs:
        df.loc[0, 'Probe diff'] = np.NaN
        df.loc[1:, 'Probe diff'] = _smooth(df.loc[1:, 'Probe diff'].values,
                                           win=SMOOTH_WIN,
                                           )

    # Determine categorical labels
    drop_points = [
        120,
        100,
        131,
        123,
        122,
        119,
        115,
        100,
        89,
        78,
        148,
        57,
        102,
        55,
        55,
        60,
        50,
        59,
    ]

    for i, df in enumerate(dfs):
        # df = multiclass_categorise(df, DOC, TOL)
        df = drop_point_labels(df, drop_points[i])

    #* Plot probe data scatter
    fig, ax = plot_scatter_cat(dfs, DOC, TOL)
    
    #* Dataset setup for ML
    # Joing all datasets
    main_df = create_maindf(dfs)
    print(f'Main DF: {main_df.shape[0]} rows x {main_df.shape[1]} columns')

    # Split into input and target data
    input_df = main_df.iloc[:, :-1].copy()

    target_df = main_df.iloc[:, -1].copy()
    target_df = keras.utils.to_categorical(target_df)

    # split of categories in each class
    percent = np.sum(target_df, axis=0) / target_df.shape[0] * 100
    print('Dataset Class Distribution:')
    for i, p in enumerate(percent):
        print(f'Class {i} : {p:.2f}%')

    # Split into training and testing data
    idx = np.arange(input_df.shape[0])
    idx_train, idx_test = train_test_split(idx,
                                           test_size=0.3,
                                           random_state=RANDOM_STATE,
                                           stratify=target_df,
                                           )
    print(f'Training Data: {len(idx_train)} - Testing Data: {len(idx_test)}\n')

    #* Model Setup
    # Define model
    mlp = create_mlp_model(MODELPARAMS,
                           nInputs=input_df.shape[1],
                           nOutputs=target_df.shape[1],
                           )

    if TB:
        tb_writer = tf.summary.create_file_writer(str(TB_LOG_DIR))
        _tb_model_desc(_mlp_tb_desc_dict(MODELPARAMS,
                                         COMPILEPARAMS,
                                         FITPARAMS,
                                         ),
                       tb_writer,
                       )
        hp_params = MODELPARAMS | COMPILEPARAMS | FITPARAMS
        pop_keys = ['modelFunc', 'metrics', 'verbose']
        for key in pop_keys:
            hp_params.pop(key, None)
        hp_params['nUnits'] = str(hp_params['nUnits'])
        with tb_writer.as_default():
            hp.hparams(hp_params,
                       trial_id=run_name,
                       )

    #* Cross-validation
    cvData = [input_df.iloc[idx_train].values, target_df[idx_train]]
    cv_model(cvNSplits=CV_NSPLITS,
             cvNRepeats=CV_NREPEATS,
             cvData=cvData,
             modelParams=MODELPARAMS,
             compileParams=COMPILEPARAMS,
             fitParams=FITPARAMS,
             printout=True,
             TB=TB,
             tbLogDir=TB_LOG_DIR,
             )
    
    # Scale/Normalise dataset based on training data
    scaler = MinMaxScaler().fit(input_df.iloc[idx_train])
    input_df = scaler.transform(input_df)
    
    #* MLP model
    # Compile model
    mlp.compile(optimizer=COMPILEPARAMS['optimizer'],
                loss=COMPILEPARAMS['loss'],
                metrics=COMPILEPARAMS['metrics'],
                )
    # Fit model
    callbacks = [TqdmCallback(verbose=0)]
    if TB:
        callbacks.append(keras.callbacks.TensorBoard(log_dir=TB_LOG_DIR,))

    history = mlp.fit(input_df[idx_train],
                      target_df[idx_train],
                      epochs=FITPARAMS['epochs'],
                      batch_size=FITPARAMS['batch_size'],
                      validation_data=(input_df[idx_test],
                                       target_df[idx_test]
                                       ),
                      verbose=FITPARAMS['verbose'],
                      callbacks=callbacks,
                      )

    #* Model Evaluation
    model_evaluate(mlp,
                   (input_df, target_df),
                   idx_test,
                   printout=True,
                   tb=TB,
                   tbLogDir=TB_LOG_DIR,
                   )
    
    plt.show()
