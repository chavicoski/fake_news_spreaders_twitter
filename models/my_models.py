import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Embedding, BatchNormalization, Dropout, Activation, Concatenate, Lambda
import tensorflow_text

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
    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(32)))
    model.add(BatchNormalization())
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
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

def model_4(lang, trainable_embedding=True, downloaded=False):
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

    embedding = hub.KerasLayer(embedding_path, output_shape=[128], input_shape=[], dtype=tf.string, trainable=trainable_embedding, name="Embedding_128_normalized")

    x_input = Input(shape=[], dtype=tf.string)
    x_embed = embedding(x_input)
    x1 = Dense(32, activation="relu")(x_embed)
    x1 = BatchNormalization()(x1)
    x2 = Concatenate()([x_embed, x1])
    x2 = Dropout(0.5)(x2)
    x3 = Dense(64, activation="relu")(x2)
    x3 = BatchNormalization()(x3)
    x4 = Concatenate()([x2, x3])
    x4 = Dropout(0.5)(x4)
    x5 = Dense(128, activation="relu")(x4)
    x5 = BatchNormalization()(x5)
    x6 = Concatenate()([x4, x5])
    x6 = Dropout(0.5)(x6)
    x7 = Dense(256, activation="relu")(x6)
    x7 = BatchNormalization()(x7)
    out = Dense(1, activation="sigmoid")(x7)

    model = keras.Model(inputs=[x_input], outputs=[out])

    return model

def model_5(downloaded=False):
    '''
    Model with pretrained convolutional encoder for multilingual sentences(to 512 vector encoding)
    '''
    # Get the embedding for the selected language
    if downloaded:
        embedding_path = "models/tf_hub_modules/embedding_cnn_multilingual_512"
    else:
        embedding_path = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3"

    embedding = hub.load(embedding_path)

    def embedding_func(x):
        return embedding(x)

    x_input = Input(shape=[], dtype=tf.string)
    x_embed = Lambda(embedding_func, name="multilingual_encoder")(x_input)
    x1 = Dense(32, activation="relu")(x_embed)
    x1 = BatchNormalization()(x1)
    x2 = Concatenate()([x_embed, x1])
    x2 = Dropout(0.5)(x2)
    x3 = Dense(64, activation="relu")(x2)
    x3 = BatchNormalization()(x3)
    x4 = Concatenate()([x2, x3])
    x4 = Dropout(0.5)(x4)
    x5 = Dense(128, activation="relu")(x4)
    x5 = BatchNormalization()(x5)
    x6 = Concatenate()([x4, x5])
    x6 = Dropout(0.5)(x6)
    x7 = Dense(256, activation="relu")(x6)
    x7 = BatchNormalization()(x7)
    out = Dense(1, activation="sigmoid")(x7)

    model = keras.Model(inputs=[x_input], outputs=[out])

    return model

def model_6(downloaded=False):
    '''
    Model with pretrained convolutional encoder for multilingual sentences(to 512 vector encoding).
    A bigger version of model 5
    '''
    # Get the embedding for the selected language
    if downloaded:
        embedding_path = "models/tf_hub_modules/embedding_cnn_multilingual_512"
    else:
        embedding_path = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3"

    embedding = hub.load(embedding_path)

    def embedding_func(x):
        return embedding(x)

    x_input = Input(shape=[], dtype=tf.string)
    x_embed = Lambda(embedding_func, name="multilingual_encoder")(x_input)
    x1 = Dense(64, activation="relu")(x_embed)
    x1 = BatchNormalization()(x1)
    x2 = Concatenate()([x_embed, x1])
    x2 = Dropout(0.5)(x2)
    x3 = Dense(128, activation="relu")(x2)
    x3 = BatchNormalization()(x3)
    x4 = Concatenate()([x2, x3])
    x4 = Dropout(0.5)(x4)
    x5 = Dense(256, activation="relu")(x4)
    x5 = BatchNormalization()(x5)
    x6 = Concatenate()([x4, x5])
    x6 = Dropout(0.5)(x6)
    x7 = Dense(512, activation="relu")(x6)
    x7 = BatchNormalization()(x7)
    out = Dense(1, activation="sigmoid")(x7)

    model = keras.Model(inputs=[x_input], outputs=[out])

    return model

def model_7(en_path="", es_path="", downloaded=False, frozen=False):
    '''
    Model with pretrained convolutional encoder for multilingual sentences(to 512 vector encoding)
    Uses a bilinear style architecture with the models 5_en and 5_es as the two branches.
    '''
    # Get the embedding for the selected language
    if downloaded:
        embedding_path = "models/tf_hub_modules/embedding_cnn_multilingual_512"
    else:
        embedding_path = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3"

    embedding = hub.load(embedding_path)

    def embedding_func(x):
        return embedding(x)

    # Create the models 5_en and 5_es to load the weights 
    en_model = model_5(downloaded=True)  
    es_model = model_5(downloaded=True)  

    if en_path != "" and es_path != "":
        # Get the pretrained weights 
        en_model.load_weights(en_path)
        es_model.load_weights(es_path)

    # Create the input and embedding of the net
    x_input = Input(shape=[], dtype=tf.string)
    x_embed = Lambda(embedding_func, name="multilingual_encoder")(x_input)

    '''
    Take the layers of interest of each loaded model to create the chains of the bilinear model
    '''
    # English chain
    x = x_embed
    first_concat = True
    for i, layer in enumerate(en_model.layers[2:-2]):
        # We have to handle concat layers a bit different beacuse of the two inputs
        if layer.__class__.__name__ == "Concatenate":
            # Get the concat inputs
            bn_out = en_model.layers[2:-2][i-1].get_output_at(1)
            if first_concat:
                # For the first concat we take the embedding output
                skip_out = x_embed
                first_concat= False
            else:
                skip_out = en_model.layers[2:-2][i-4].get_output_at(1)

            # Set the concat inputs 
            x = layer([bn_out, skip_out])

        else:
            # For the normal layers we just fordward the previous layer output
            x = layer(x)

        # Freeze the weights
        if frozen: layer.trainable = False

    # Spanish chain
    x = x_embed
    first_concat = True
    for i, layer in enumerate(es_model.layers[2:-2]):
        # We have to handle concat layers a bit different beacuse of the two inputs
        if layer.__class__.__name__ == "Concatenate":
            # Get the concat inputs
            bn_out = es_model.layers[2:-2][i-1].get_output_at(1)
            if first_concat:
                # For the first concat we take the embedding output
                skip_out = x_embed
                first_concat= False
            else:
                skip_out = es_model.layers[2:-2][i-4].get_output_at(1) 

            # Set the concat inputs 
            x = layer([bn_out, skip_out])

        else:
            # For the normal layers we just fordward the previous layer output
            x = layer(x)

        # Freeze the weights
        if frozen: layer.trainable = False

    '''
    Connect the chains and add the new layers
    '''
    # Conect chains
    x_concat = Concatenate()([en_model.layers[-3].get_output_at(1), es_model.layers[-3].get_output_at(1)])
    x1 = BatchNormalization()(x_concat)
    x2 = Dense(128, activation="relu")(x1)
    x3 = BatchNormalization()(x2)
    out = Dense(1, activation="sigmoid")(x3)

    # Build the model
    model = keras.Model(inputs=[x_input], outputs=[out])

    return model
