# tf and keras autoencoder code without scikeras

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Dense, Input, Lambda, Dropout
from keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import src

# Get Project Paths
HOME_DIR, BASE_DIR, CODE_DIR, TB_DIR, RMS_DATA_DIR = src.config_paths()


class AutoEncoder(Model):
    def __init__(self,
                 input_dim=300,
                 latent_dim=16,
                 n_size=[64, 64],
                 dropout=0.01,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_size = n_size
        self.dropout = dropout
        self.encoder = self.get_encoder(input_dim, latent_dim, n_size)
        self.decoder = self.get_decoder(input_dim, latent_dim, n_size)

    def get_encoder(self, input_dim, latent_dim, n_size):
        inputs = Input(shape=(input_dim,), name='encoder_input')
        e = inputs

        for dim in n_size:
            e = Dense(dim, activation='relu')(e)
            e = Dropout(self.dropout)(e)

        encoder_out = Dense(latent_dim,
                            activation='relu',
                            )(e)

        encoder = Model(inputs, encoder_out, name='encoder')
        return encoder

    def get_decoder(self, input_dim, latent_dim, n_size):
        latent_inputs = Input(shape=(latent_dim,))
        d = latent_inputs

        for dim in n_size[::-1]:
            d = Dense(dim, activation='relu')(d)
            d = Dropout(self.dropout)(d)

        outputs = Dense(input_dim, activation='linear')(d)

        decoder = Model(latent_inputs, outputs, name='decoder')
        return decoder
    
    def call(self, inputs):
        z = self.encoder(inputs)
        out_decoder = self.decoder(z)
        self.add_loss(tf.losses.mean_squared_error(inputs, out_decoder))
        return out_decoder
        
    def build_graph(self):
        x = Input(shape=(self.input_dim,))
        return Model(inputs=[x], outputs=self.call(x))


class VariationalAutoEncoder(Model):
    def __init__(self,
                 input_dim=300,
                 latent_dim=16,
                 n_size=[64, 64],
                 r_loss_factor=1000,
                 dropout=0.01,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_size = n_size
        self.r_loss_factor = r_loss_factor
        self.dropout = dropout
        self.encoder = self.get_encoder(input_dim, latent_dim, n_size)
        self.decoder = self.get_decoder(input_dim, latent_dim, n_size)

    def get_encoder(self, input_dim, latent_dim, n_size):
        inputs = Input(shape=(input_dim,), name='encoder_input')
        e = inputs

        for dim in n_size:
            e = Dense(dim, activation='relu')(e)
            # e = BatchNormalization()(e)
            e = Dropout(self.dropout)(e)

        z_mean = Dense(latent_dim, name='z_mean')(e)
        z_log_sigma = Dense(latent_dim, name='z_log_sigma')(e)

        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                      mean=0., stddev=1.)
            return z_mean + K.exp(z_log_sigma / 2) * epsilon

        z = Lambda(sampling)([z_mean, z_log_sigma])

        # encoder mapping inputs to rthe latent space
        encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
        return encoder

    def get_decoder(self, input_dim, latent_dim, n_size):
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        d = latent_inputs

        for dim in n_size[::-1]:
            d = Dense(dim, activation='relu')(d)
            # d = BatchNormalization()(d)
            d = Dropout(self.dropout)(d)

        outputs = Dense(input_dim, activation='sigmoid')(d)

        decoder = Model(latent_inputs, outputs, name='decoder')
        return decoder
    
    @staticmethod
    def reconstruction_loss(y_true, y_pred):
        reconstruction_loss = tf.losses.mean_squared_error(
            y_true,
            y_pred,
        )
        return reconstruction_loss
    
    @staticmethod
    def kl_loss(z_mean, z_log_sigma):
        kl_loss = -0.5 * K.sum(
            1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma),
            axis=1
        )
        return kl_loss
    
    @staticmethod
    def total_loss(reconstruction_loss, kl_loss, r_loss_factor):
        return (r_loss_factor * reconstruction_loss) + kl_loss

    def call(self, inputs):
        z_mean, z_log_sigma, z = self.encoder(inputs)
        out_decoder = self.decoder(z)

        recon_loss = self.reconstruction_loss(inputs, out_decoder)
        recon_loss *= self.input_dim

        kl_loss = self.kl_loss(z_mean, z_log_sigma)

        vae_loss = self.total_loss(recon_loss, kl_loss, self.r_loss_factor)
        self.add_loss(vae_loss)
        self.add_metric(recon_loss, name='reconstruction_loss')
        self.add_metric(kl_loss, name='kl_loss')
        return out_decoder
        
    def build_graph(self):
        x = Input(shape=(self.input_dim,))
        return Model(inputs=[x], outputs=self.call(x))

    def plot_latent_space(self, data):
        z_mean, _, _ = self.encoder(data)
        
        fig, ax = plt.subplots()
        s = ax.scatter(z_mean[:, 0], z_mean[:, 1],
                       c=range(len(z_mean[:, 0])),
                       cmap=plt.get_cmap('jet'),
                       )
        cbar = plt.colorbar(s)
        cbar.set_label('Cut No.')
        return fig, ax


def pre_process(data: np.array, ind: (tuple, tuple)):
    ind_tr, ind_val = ind

    ind_tr, ind_te = train_test_split(ind_tr,
                                      test_size=0.2,
                                      random_state=42,
                                      )
    scaler = MinMaxScaler()
    scaler.fit(data[ind_tr, :].reshape(-1, 1))
    data_sc = scaler.transform(data.reshape(-1, 1)).reshape(data.shape)

    x_train = data_sc[ind_tr, :]
    x_val = data_sc[ind_val, :]
    x_test = data_sc[ind_te, :]
    return x_train, x_test, x_val, data_sc


if __name__ == '__main__':
    # Setup Tensorflow
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print('Num GPUs Available: ', len(gpus))
        try:
            # Limit memory usage to 5GB of the first GPU (if available)
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
            )
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(
                f'{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs'
            )
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    else:
        print('No GPUs Available')

    # tf.config.set_visible_devices([], 'GPU')
    exps = ['Test 8']
    rms = {}
    for test in exps:
        rms[test] = src.ae.RMS(test)
        rms[test].data.drop(['0', '1', '2'], axis=1, inplace=True)

    # remove outside triggers and DC offset
    def remove_dc(sig):
        return sig - np.nanmean(sig)

    for test in exps:
        rms[test]._data = rms[test].data.T.reset_index(drop=True).T
        rms[test]._data = rms[test].data.iloc[50:350, :].reset_index(drop=True)
        rms[test]._data = rms[test].data.apply(remove_dc, axis=0)

    dataset = rms[exps[0]].data.values.T

    TRAIN_STOP_IND = 50

    x_train, x_test, x_val, dataset_sc = pre_process(
        dataset,
        (range(0, TRAIN_STOP_IND),
         range(TRAIN_STOP_IND, dataset.shape[0])
         )
    )

    vae = VariationalAutoEncoder(x_train.shape[1],
                                 n_size=[16],
                                 latent_dim=2,
                                 r_loss_factor=1000,
                                 )
    vae.build_graph().summary()

    vae.compile(optimizer=keras.optimizers.Adam(),)

    history = vae.fit(x_train, x_train,
                      validation_data=(x_test, x_test),
                      epochs=500,
                      batch_size=32,
                      )

    fig, ax = vae.plot_latent_space(dataset)

    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='Training Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')

    pred = vae.predict(dataset_sc, verbose=0)

    recon_scores = vae.reconstruction_loss(dataset_sc, pred)
    kl_scores = vae.kl_loss(*vae.encoder(dataset_sc)[:2])
    total_scores = vae.total_loss(recon_scores, kl_scores, vae.r_loss_factor)

    exp = src.load(exps[0])
    runout = exp.features['Runout'].drop([0, 1, 2])

    # Scatter plot of reconstruction loss with runout overlayed
    fig, ax = plt.subplots(constrained_layout=True)
    ax.scatter(range(len(recon_scores)),
               recon_scores,
               s=5,
               )
    ax.axvline(TRAIN_STOP_IND,
               color='k',
               linestyle='--',
               alpha=0.5,
               )
    ax2 = ax.twinx()
    ax2.plot(runout, 'C1', label='Runout')
    ax.set_ylabel('Reconstruction Loss')
    ax2.set_ylabel('Runout')
    ax.set_xlabel('Cut No.')

    plt.show()
