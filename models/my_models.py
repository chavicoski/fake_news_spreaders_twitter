import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Embedding, BatchNormalization, Dropout

def model_0(lang, trainable_embedding=True, downloaded=False):
    '''
    Model with pretrained embedding of size 50
    '''
    # Get the embedding for the selected language
    if lang == 'en':
        if downloaded:
            embedding_path = "models/tf_hub_modules/embedding_nnlm_en_50"
        else:
            embedding_path = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1"
    elif lang == 'es':
        if downloaded:
            embedding_path = "models/tf_hub_modules/embedding_nnlm_es_50"
        else:
            embedding_path = "https://tfhub.dev/google/nnlm-es-dim50/2"

    embedding_layer = hub.KerasLayer(embedding_path, output_shape=[50], input_shape=[], dtype=tf.string, trainable=trainable_embedding, name="Embedding_50")

    model = keras.Sequential()
    model.add(embedding_layer)
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    
    return model
    
def model_1(vocab_size):
    '''
    Model with new emmbedding to train
    '''
    model = keras.Sequential()
    model.add(Embedding(vocab_size, 64))
    model.add(Bidirectional(LSTM(32)))
    model.add(BatchNormalization())
    model.add(Dense(32, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(32, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(1, activation="sigmoid"))

    return model

def model_2(lang, trainable_embedding=True, downloaded=False):
    '''
    Model with pretrained embedding of size 128
    '''
    # Get the embedding for the selected language
    if lang == 'en':
        if downloaded:
            embedding_path = "models/tf_hub_modules/embedding_nnlm_en_128"
        else:
            embedding_path = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"
    elif lang == 'es':
        if downloaded:
            embedding_path = "models/tf_hub_modules/embedding_nnlm_es_128"
        else:
            embedding_path = "https://tfhub.dev/google/tf2-preview/nnlm-es-dim128/1"

    embedding_layer = hub.KerasLayer(embedding_path, output_shape=[128], input_shape=[], dtype=tf.string, trainable=trainable_embedding, name="Embedding_128")

    model = keras.Sequential()
    model.add(embedding_layer)
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(32, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(32, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    
    return model

def model_3(lang, trainable_embedding=True, downloaded=False):
    '''
    Model with pretrained embedding of size 128 and normalization
    '''
    # Get the embedding for the selected language
    if lang == 'en':
        if downloaded:
            embedding_path = "models/tf_hub_modules/embedding_nnlm_en_128_norm"
        else:
            embedding_path = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128-with-normalization/1"
    elif lang == 'es':
        if downloaded:
            embedding_path = "models/tf_hub_modules/embedding_nnlm_es_128_norm"
        else:
            embedding_path = "https://tfhub.dev/google/tf2-preview/nnlm-es-dim128-with-normalization/1"

    embedding_layer = hub.KerasLayer(embedding_path, output_shape=[128], input_shape=[], dtype=tf.string, trainable=trainable_embedding, name="Embedding_128_normalized")

    model = keras.Sequential()
    model.add(embedding_layer)
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(32, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(32, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    
    return model
 
