
import numpy as np, pandas as pd
import math
import joblib
import sys
import os, warnings
os.environ['PYTHONHASHSEED']=str(2)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 
from sklearn.metrics import mean_squared_error


import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback

from tensorflow.keras.optimizers import Adam

# from algorithm.model.vae_dense_model import VariationalAutoencoderDense as VAE
from algorithm.model.vae_conv_model import VariationalAutoencoderConv as VAE

MODEL_NAME = "Forecaster_Base_VAE"


model_pred_pipeline_fname = "model_pred_pipeline.save"
model_params_fname = "model_params.save"
model_encoder_wts_fname = "model_encoder_wts.save"
model_decoder_wts_fname = "model_decoder_wts.save"
train_history_fname = "train_history.json"
train_data_fname = "train_data.csv"
train_data_fname_zip = "train_data.zip"


COST_THRESHOLD = float('inf')

class InfCostStopCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        loss_val = logs.get('loss')
        if(loss_val == COST_THRESHOLD or tf.math.is_nan(loss_val)):
            print("Cost is inf, so stopping training!!")
            self.model.stop_training = True


def get_data_based_model_params(train_data): 
    ''' 
        Set any model parameters that are data dependent. 
        For example, number of layers or neurons in a neural network as a function of data shape.
    '''     
    return {
        "feat_dim": train_data['X'].shape[2], 
        "encode_len": train_data['X'].shape[1],
        "decode_len": train_data['y'].shape[1]
        }

def get_patience_factor(N): 
    # magic number - just picked through trial and error
    patience = int(35 - math.log(N, 1.5))
    return patience


class Forecaster(): 
    MIN_HISTORY_LEN = 60        # in epochs

    def __init__(self, 
                encode_len, 
                decode_len, 
                feat_dim, 
                latent_dim,
                first_hidden_dim,
                second_hidden_dim,
                loss_decay_const = 0.99,
                reconstruction_wt=5.0, **kwargs ):
        
        self.encode_len = encode_len
        self.decode_len = decode_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.loss_decay_const = loss_decay_const
        self.first_hidden_dim = first_hidden_dim
        self.second_hidden_dim = second_hidden_dim
        self.hidden_layer_sizes = [int(first_hidden_dim), int(second_hidden_dim)]
        self.reconstruction_wt = reconstruction_wt


        self.vae_model = VAE(
            encode_len=encode_len,
            decode_len=decode_len,
            feat_dim=feat_dim,
            latent_dim = latent_dim,
            hidden_layer_sizes=self.hidden_layer_sizes,
            reconstruction_wt = reconstruction_wt
        )   

        self.vae_model.compile(optimizer=Adam()) 

    
    
    def fit(self, train_data, validation_split=0.1, verbose=0, max_epochs=1000):

        train_X, train_y, train_y_missing = train_data['X'], train_data['y'], train_data['y_missing']
        print("X/y shapes", train_X.shape, train_y.shape)
        # sys.exit()

        if train_X.shape[0] < 100:  validation_split = None
        
        patience = get_patience_factor(train_X.shape[0])
        # print("patience", patience)
        
        loss_to_monitor = 'loss' if validation_split is None else 'val_loss'
        early_stop_callback = EarlyStopping(monitor=loss_to_monitor, 
                                            min_delta = 1e-4, patience=patience) 

        learning_rate_reduction = ReduceLROnPlateau(monitor=loss_to_monitor, 
                                            patience=3, 
                                            factor=0.85, 
                                            min_lr=1e-5)

        inf_cost_callback = InfCostStopCallback()
        
        history = self.vae_model.fit(
            x=train_X, 
            y=[train_y, train_y_missing], 
            # y=train_y, 
            validation_split = validation_split,
            shuffle=False,
            verbose=verbose,
            epochs=max_epochs,  
            callbacks=[early_stop_callback, inf_cost_callback],
            batch_size=32)
        
        return history
    

    def predict(self, data):        
        X, y = data['X'], data['y']  
        # print(X.shape, E.shape, y.shape)  ; sys.exit()
        x_decoded = self.vae_model.predict(X)
        x_decoded = x_decoded[:, -self.decode_len:]
        
        return x_decoded
    

    def evaluate(self, data): 
        preds = self.predict(data)
        mse = mean_squared_error(data['y'].flatten(), preds.flatten())
        return mse
    

    def summary(self): 
        self.model.summary()
        
    
    
    def save(self, model_path):        
        encoder_wts = self.vae_model.encoder.get_weights()
        decoder_wts = self.vae_model.decoder.get_weights()
        joblib.dump(encoder_wts, os.path.join(model_path, model_encoder_wts_fname))
        joblib.dump(decoder_wts, os.path.join(model_path, model_decoder_wts_fname))
        
        dict_params = {
            'encode_len': self.encode_len,
            'decode_len': self.decode_len,
            'feat_dim': self.feat_dim,
            'latent_dim': self.latent_dim,
            'first_hidden_dim': self.first_hidden_dim,
            'second_hidden_dim': self.second_hidden_dim,
            'loss_decay_const': self.loss_decay_const,
            'reconstruction_wt': self.reconstruction_wt,
        }
        joblib.dump(dict_params, os.path.join(model_path, model_params_fname))
    
    @classmethod
    def load(cls, model_path):        
        dict_params = joblib.load(os.path.join(model_path, model_params_fname))
        model = cls(
            encode_len = dict_params['encode_len'], 
            decode_len = dict_params['decode_len'],        # 'auto' if auto-encoding, or int > 0 if forecasting
            feat_dim = dict_params['feat_dim'], 
            latent_dim = dict_params['latent_dim'], 
            first_hidden_dim=dict_params['first_hidden_dim'], 
            second_hidden_dim=dict_params['second_hidden_dim'], 
            loss_decay_const = dict_params['loss_decay_const'], 
            reconstruction_wt = dict_params['reconstruction_wt']
        )

        first_hidden_dim = int(dict_params['first_hidden_dim'])
        second_hidden_dim = int(dict_params['second_hidden_dim'])
        
        model.vae_model = VAE(
            encode_len = dict_params['encode_len'], 
            decode_len = dict_params['decode_len'],        # 'auto' if auto-encoding, or int > 0 if forecasting
            feat_dim = dict_params['feat_dim'], 
            latent_dim = dict_params['latent_dim'], 
            hidden_layer_sizes=[first_hidden_dim, second_hidden_dim ], 
            reconstruction_wt = dict_params['reconstruction_wt'], 
        )

        encoder_wts = joblib.load(os.path.join(model_path, model_encoder_wts_fname))
        decoder_wts = joblib.load(os.path.join(model_path, model_decoder_wts_fname))
        model.vae_model.encoder.set_weights(encoder_wts)
        model.vae_model.decoder.set_weights(decoder_wts)
        
        model.vae_model.compile(optimizer=Adam())
        return model
    

    def get_num_trainable_variables(self):
        return self.vae_model.get_num_trainable_variables()
      

def save_model_artifacts(train_artifacts, model_artifacts_path): 
    # save model
    save_model(train_artifacts["model"], model_artifacts_path)
    # save model-specific prediction pipeline
    save_model_pred_pipeline(train_artifacts["model_pred_pipeline"], model_artifacts_path)
    # save traiing history
    save_training_history(train_artifacts["train_history"], model_artifacts_path)
    # save training data
    save_training_data(train_artifacts["train_data"], model_artifacts_path)

def save_model(model, model_path):    
    model.save(model_path) 
    

def save_model_pred_pipeline(pipeline, model_path): 
    joblib.dump(pipeline, os.path.join(model_path, model_pred_pipeline_fname))


def save_training_history(history, f_path): 
    hist_df = pd.DataFrame(history.history) 
    hist_json_file = os.path.join(f_path, train_history_fname)
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)


def save_training_data(train_data, model_artifacts_path):
    compression_opts = { "method":'zip',  "archive_name": train_data_fname }      
    train_data.to_csv(os.path.join(model_artifacts_path, train_data_fname_zip), 
            index=False,  compression=compression_opts) 


