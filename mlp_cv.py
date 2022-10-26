import ml_reg_tb
from resources import *
import time
from typing import Any
import multiprocessing
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from scikeras.wrappers import KerasRegressor

EPOCHS = 500
BATCH_SIZE = 10
DROPOUT = 0.1
NO_LAYERS = 2
NO_NODES = 32
TEST_FRAC = 0.2
TARGET = 'Mean radius'

logdir = 'ml-results/logs/MLP/CV/'
run_name = f'{logdir}MLP-E-{EPOCHS}-B-{BATCH_SIZE}-L{np.full(NO_LAYERS, NO_NODES)}-D-{DROPOUT}' \
           f'-{time.strftime("%Y%m%d-%H%M%S", time.localtime())}'
exp = resources.load('Test 5')
main_df = exp.features.drop(columns=['Runout', 'Form error']).drop([0, 1, 23, 24])

X_train, X_val, y_train, y_val = ml_reg_tb.preprocess_df(main_df, TEST_FRAC)


def create_model():
    model: Any = KerasRegressor(
        model=ml_reg_tb.create_mlp,
        model__dropout=DROPOUT,
        model__activation='relu',
        model__no_nodes=NO_NODES,
        model__no_layers=NO_LAYERS,
        model__init_mode='glorot_normal',
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        loss='mae',
        metrics=['MSE', 'MAE', KerasRegressor.r_squared],
        optimizer='adam',
        optimizer__learning_rate=0.001,
        optimizer__decay=1e-6,
        verbose=2,
        callbacks=[tf.keras.callbacks.TensorBoard(log_dir=run_name, histogram_freq=1)],
    )
    return model


def train_model(run_no, tr_index, te_index):
    model = create_model()
    model.callbacks[0].log_dir = f'{model.callbacks[0].log_dir}/run_{run_no}'
    model.fit(X_train[tr_index], y_train[tr_index], validation_data=(X_train[te_index], y_train[te_index]))
    score = ml_reg_tb.score_test(model, X_train[te_index], y_train[te_index], plot_fig=False)
    return score


if __name__ == "__main__":
    cv = KFold(n_splits=5, shuffle=True, random_state=10)
    cv_items = [(i, train, test) for i, (train, test) in enumerate(cv.split(X_train))]

    tb_writer = tf.summary.create_file_writer(run_name)

    with multiprocessing.Pool() as pool:
        scores = pool.starmap(train_model, cv_items)

    mlp_reg = create_model()
    mlp_reg.fit(X_train, y_train)
    val_scores = ml_reg_tb.score_test(mlp_reg, X_val, y_val)

    mean_MAE = np.mean([score['MAE'] for score in scores])
    mean_MSE = np.mean([score['MSE'] for score in scores])
    mean_r2 = np.mean([score['r2'] for score in scores])

    std_MAE = np.std([score['MAE'] for score in scores])
    std_MSE = np.std([score['MSE'] for score in scores])
    std_r2 = np.std([score['r2'] for score in scores])

    print(f'Training Scores:')
    print(f'MAE: {mean_MAE * 1_000:.3f} ({std_MAE: .3f}) um')
    print(f'MSE: {mean_MSE * 1_000_000:.3f} ({std_MSE: .3f}) um^2')
    print(f'R^2: {mean_r2:.3f} ({std_MSE: .3f})')

    print(val_scores)
    # todo finish cross-validation with tensorboard
