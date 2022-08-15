
import os, warnings, sys
from re import T
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Conv1D, Conv2D, \
    Activation, Flatten, Dense, Conv1DTranspose, Reshape, Layer, Input, Multiply, Add
from tensorflow.keras.models import Model
from tensorflow.keras.backend import random_normal
from tensorflow.keras.metrics import Mean, BinaryCrossentropy
from tensorflow.keras.losses import MeanSquaredError, binary_crossentropy
from tensorflow.keras.optimizers import Adam

# from .utils import get_mnist_data, draw_orig_and_post_pred_sample, plot_latent_space
from algorithm.model.vae_base import BaseVariationalAutoencoder


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon



class VariationalAutoencoderConv(BaseVariationalAutoencoder):
    def __init__(self, 
            hidden_layer_sizes, 
            **kwargs           
        ):
        super(VariationalAutoencoderConv, self).__init__(**kwargs)

        self.hidden_layer_sizes = hidden_layer_sizes

        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()


    def _get_encoder(self):
        encoder_inputs = Input(shape=(self.encode_len, self.feat_dim), name='encoder_input')
        x = encoder_inputs
               
        for i, num_filters in enumerate(self.hidden_layer_sizes):
            x = Conv1D(
                    filters = num_filters, 
                    kernel_size=3, 
                    strides=2, 
                    activation='relu', 
                    padding='same',
                    name=f'enc_conv_{i}')(x)

        x = Flatten(name='enc_flatten')(x)

        # save the dimensionality of this last dense layer before the hidden state layer. We need it in the decoder.
        self.encoder_last_dense_dim = x.get_shape()[-1]        

        z_mean = Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(x)

        encoder_output = Sampling()([z_mean, z_log_var])     
        self.encoder_output = encoder_output
        
        
        encoder = Model(encoder_inputs, [z_mean, z_log_var, encoder_output], name="encoder")
        # encoder.summary(); sys.exit()
        return encoder


    def _get_decoder(self):
        decoder_inputs = Input(shape=(self.latent_dim), name='decoder_input')
        
        x = decoder_inputs
        x = Dense(self.encoder_last_dense_dim, name="dec_dense", activation='relu')(x)
        x = Reshape(target_shape=(-1, self.hidden_layer_sizes[-1]), name="dec_reshape")(x)

        for i, num_filters in enumerate(reversed(self.hidden_layer_sizes[:-1])):
            x = Conv1DTranspose(
                filters = num_filters, 
                    kernel_size=3, 
                    strides=2, 
                    padding='same',
                    activation='relu', 
                    name=f'dec_deconv_{i}')(x)

        # last de-convolution
        x = Conv1DTranspose(
                filters = self.feat_dim, 
                    kernel_size=3, 
                    strides=2, 
                    padding='same',
                    activation='relu', 
                    name=f'dec_deconv__{i+1}')(x)

        x = Flatten(name='dec_flatten')(x)
        
        # x = Dense(self.seq_len * self.feat_dim, name="decoder_dense_final")(x)
        # self.decoder_outputs = Reshape(target_shape=(self.seq_len, self.feat_dim))(x)        
        
        self.decoder_outputs = Dense(self.decode_len, name="decoder_dense_final")(x)
        decoder = Model(decoder_inputs, self.decoder_outputs, name="decoder")
        
        return decoder


#####################################################################################################
#####################################################################################################

