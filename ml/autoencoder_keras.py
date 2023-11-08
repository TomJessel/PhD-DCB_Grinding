# tf and keras autoencoder code without scikeras

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Layer, Dense, Input, Lambda
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
            [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
        )
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
else:
    print('No GPUs Available')

# tf.config.set_visible_devices([], 'GPU')


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
        self.dense_1 = Dense(inter_dim,
                             activation='relu',
                             )
        self.dense_2 = Dense(inter_dim,
                             activation='relu',
                             )
        self.dense_mean = Dense(latent_dim)
        self.dense_log_var = Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
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
        self.dense_1 = Dense(inter_dim, activation='relu')
        self.dense_2 = Dense(inter_dim, activation='relu')
        self.dense_out = Dense(original_dim, activation='sigmoid')

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
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

    def build_graph(self):
        x = Input(shape=(self.original_dim,))
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


# class VariationalAutoEncoder(Model):
#     def __init__(self, input_dim=300, latent_dim=16, n_size=[64, 64]):
#         super().__init__()
#         self.input_dim = input_dim
#         self.latent_dim = latent_dim
#         self.n_size = n_size
#         self.encoder = self.get_encoder(input_dim, latent_dim, n_size)
#         self.decoder = self.get_decoder(input_dim, latent_dim, n_size)

#     def get_encoder(self, input_dim, latent_dim, n_size):
#         inputs = Input(shape=(input_dim,), name='encoder_input')
#         e = inputs

#         for dim in n_size:
#             e = Dense(dim, activation='relu')(e)

#         z_mean = Dense(latent_dim, name='z_mean')(e)
#         z_log_sigma = Dense(latent_dim, name='z_log_sigma')(e)

#         def sampling(args):
#             z_mean, z_log_sigma = args
#             epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
#                                       mean=0., stddev=0.1)
#             return z_mean + K.exp(z_log_sigma) * epsilon

#         z = Lambda(sampling)([z_mean, z_log_sigma])

#         # encoder mapping inputs to rthe latent space
#         encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
#         return encoder

#     def get_decoder(self, input_dim, latent_dim, n_size):
#         latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
#         d = latent_inputs

#         for dim in n_size[::-1]:
#             d = Dense(dim, activation='relu')(d)

#         outputs = Dense(input_dim, activation='sigmoid')(d)

#         decoder = Model(latent_inputs, outputs, name='decoder')
#         return decoder

#     def call(self, inputs):
#         out_encoder = self.encoder(inputs)
#         z_mean, z_log_sigma, z = out_encoder
#         out_decoder = self.decoder(z)

#         reconstruction_loss = tf.keras.metrics.mean_squared_error(
#             inputs,
#             out_decoder,
#         )
#         reconstruction_loss *= self.input_dim
#         kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
#         kl_loss = K.sum(kl_loss, axis=-1)
#         kl_loss *= -0.5
#         vae_loss = K.mean(reconstruction_loss + kl_loss)
#         self.add_loss(vae_loss)
#         return out_decoder


if __name__ == '__main__':

    exps = ['Test 8']
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

    dataset = rms[exps[0]].data.values.T
    x_train = dataset[:140, :]
    x_test = dataset[140:, :]
    print(f'x_train shape: {x_train.shape}')
    print(f'x_test shape: {x_test.shape}')

    vae = _VariationalAutoEncoder(300)
    vae.compile(optimizer=keras.optimizers.Adam(),
                metrics=[keras.metrics.RootMeanSquaredError(),
                         keras.metrics.MeanAbsoluteError(),
                         ],
                )
    history = vae.fit(x_train, x_train,
                      validation_data=(x_test, x_test),
                      epochs=200,
                      batch_size=32,
                      )
    vae.build_graph().summary()

    fig, ax = vae.plot_latent_space(dataset)

    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='Training Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    plt.show()

# TODO: add in pre-processing steps
