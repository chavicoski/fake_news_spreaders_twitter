import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Embedding, BatchNormalization

def model_0(lang):
    '''
    Model with pretrained embedding of size 50
    '''
    # Get the embedding for the selected language
    if lang == 'en':
        embedding_path = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1"
    elif lang == 'es':
        embedding_path = "https://tfhub.dev/google/nnlm-es-dim50/2"

    embedding_layer = hub.KerasLayer(embedding_path, output_shape=[50], input_shape=[], dtype=tf.string, trainable=True)

    model = keras.Sequential()
    model.add(embedding_layer)
    model.add(BatchNormalization())
    model.add(Dense(32, activation="relu"))
    model.add(BatchNormalization())
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


