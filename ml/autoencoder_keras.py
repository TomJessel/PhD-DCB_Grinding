# tf and keras autoencoder code without scikeras

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np

import tensorflow as tf
from keras.models import Model
from keras.layers import Layer, Dense
from keras import backend as K

import resources


# Get Project Paths
HOME_DIR, BASE_DIR, CODE_DIR, TB_DIR, RMS_DATA_DIR = resources.config_paths()

# Setup Tensorflow
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print('Num GPUs Available: ', len(gpus))
    try:
        # Limit memory usage to 5GB of the first GPU (if available)
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=5120)]
        )
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
else:
    print('No GPUs Available')


class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    

class Encoder(Layer):
    def __init__(self,
                 latent_dim=16,
                 inter_dim=64,
                 name='encoder',
                 **kwargs
                 ):
        super().__init__(name=name, **kwargs)
        self.dense_proj = Dense(inter_dim,
                                activation='relu',
                                )
        self.dense_mean = Dense(latent_dim)
        self.dense_log_var = Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(Layer):
    def __init__(self,
                 original_dim,
                 inter_dim=64,
                 name='decoder',
                 **kwargs
                 ):
        super().__init__(name=name, **kwargs)
        self.dense_proj = Dense(inter_dim, activation='relu')
        self.dense_out = Dense(original_dim, activation='sigmoid')

    def call(self, inputs):
        x = self.dense_proj(inputs)
        return self.dense_out(x)


class _VariationalAutoEncoder(Model):
    def __init__(self,
                 original_dim,
                 latent_dim=16,
                 inter_dim=64,
                 name='autoencoder',
                 **kwargs
                 ):
        super().__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim, inter_dim)
        self.decoder = Decoder(original_dim, inter_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)

        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )

        reconstruction_loss = tf.losses.mean_squared_error(
            inputs, reconstructed
        )

        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        self.add_loss(vae_loss)
        return reconstructed


if __name__ == '__main__':

    exps = ['Test 7']
    rms = {}
    for test in exps:
        rms[test] = resources.ae.RMS(test)
        rms[test].data.drop(['0', '1', '2'], axis=1, inplace=True)

    # remove outside triggers and DC offset
    def remove_dc(sig):
        return sig - np.nanmean(sig)

    for test in exps:
        rms[test]._data = rms[test].data.T.reset_index(drop=True).T
        rms[test]._data = rms[test].data.iloc[50:350, :].reset_index(drop=True)
        rms[test]._data = rms[test].data.apply(remove_dc, axis=0)

    x_train = rms['Test 7'].data.values.T
    print(x_train.shape)

    vae = _VariationalAutoEncoder(300)
    # vae.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),)
    vae.compile(optimizer=tf.optimizers.RMSprop())
    vae.fit(x_train, x_train, epochs=100, batch_size=32)
