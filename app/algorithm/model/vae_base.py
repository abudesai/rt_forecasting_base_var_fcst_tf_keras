
import os, warnings, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 

from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Mean



class BaseVariationalAutoencoder(Model, ABC):
    def __init__(self,  
            encode_len, 
            decode_len, 
            feat_dim, 
            latent_dim,
            reconstruction_wt = 5.0,
            **kwargs  ):
        super(BaseVariationalAutoencoder, self).__init__(**kwargs)
        self.encode_len = encode_len
        self.decode_len = decode_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.reconstruction_wt = reconstruction_wt
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean( name="reconstruction_loss" )
        self.kl_loss_tracker = Mean(name="kl_loss")

        self.encoder = None
        self.decoder = None 
 

    def call(self, X):
        z_mean, _, _ = self.encoder(X)
        x_decoded = self.decoder(z_mean)
        if len(x_decoded.shape) == 1: x_decoded = x_decoded.reshape((1, -1))
        return x_decoded


    def get_num_trainable_variables(self):
        trainableParams = int(np.sum([np.prod(v.get_shape()) for v in self.trainable_weights]))
        nonTrainableParams = int(np.sum([np.prod(v.get_shape()) for v in self.non_trainable_weights]))
        totalParams = trainableParams + nonTrainableParams
        return trainableParams, nonTrainableParams, totalParams


    def get_prior_samples(self, num_samples):
        Z = np.random.randn(num_samples, self.latent_dim)
        samples = self.decoder.predict(Z)
        return samples
    

    def get_prior_samples_given_Z(self, Z):
        samples = self.decoder.predict(Z)
        return samples


    @abstractmethod
    def _get_encoder(self, **kwargs):
        raise NotImplementedError


    @abstractmethod
    def _get_decoder(self, **kwargs):
        raise NotImplementedError

    
    def summary(self):
        self.encoder.summary()
        self.decoder.summary()


    def _get_reconstruction_loss(self, X, X_recons): 

        def get_reconst_loss_by_axis(X, X_c, axis): 
            x_r = tf.reduce_mean(X, axis = axis)
            x_c_r = tf.reduce_mean(X_c, axis = axis)
            err = tf.math.squared_difference(x_r, x_c_r)
            loss = tf.reduce_sum(err)
            return loss

        # overall    
        err = tf.math.squared_difference(X, X_recons)
        reconst_loss = tf.reduce_sum(err)
      
        reconst_loss += get_reconst_loss_by_axis(X, X_recons, axis=[1])     # by time axis        
        # reconst_loss += get_reconst_loss_by_axis(X, X_recons, axis=[1])    # by feature axis
        return reconst_loss


    def train_step(self, data): 
        X, Y = data
        Y, Y_missing = Y
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(X)
            reconstruction = self.decoder(z)
            # err = tf.math.squared_difference(Y, reconstruction)
            err = tf.math.abs(Y - reconstruction) * Y_missing
            
            reconstruction_loss = tf.reduce_sum(err)            
            # reconstruction_loss = self._get_reconstruction_loss(Y, reconstruction)

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_sum(tf.reduce_sum(kl_loss, axis=1))

            total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    
    def test_step(self, data): 
        X, Y = data      
        Y, Y_missing = Y  
        z_mean, z_log_var, z = self.encoder(X)
        reconstruction = self.decoder(z)
        
        err = tf.math.abs(Y - reconstruction) * Y_missing
        reconstruction_loss = tf.reduce_sum(err)  

        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_sum(tf.reduce_sum(kl_loss, axis=1))
        # kl_loss = kl_loss / self.latent_dim

        total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

#####################################################################################################
#####################################################################################################


if __name__ == '__main__':

    pass