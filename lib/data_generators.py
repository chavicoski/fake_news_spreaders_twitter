import sys
import os
import math
import numpy as np
import xml.etree.ElementTree as ET
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_datasets as tfds

'''
Data generator that returns the tweets encoded word by word
Params:
    data_file -> Is a path to a txt file with N rows where each row has a tweet and the label (0 or 1) separated by a space.
    lang -> "en" or "es" string with the laguage of the data
    batch_size -> Number of samples per batch created by the generator
    shuffle_at_end -> To shuffle the samples at the end of each epoch
    buffer_size -> buffer size for shuffling the data
    prefetch_batches -> max number of batches to prefetch. set to 'tf.data.experimental.AUTOTUNE' for automatic selection
'''
def encoded_datagen(data_file, lang, batch_size=64, shuffle_at_end=True, buffer_size=100, prefetch_batches=tf.data.experimental.AUTOTUNE):

    # Function so separate the tweet string from the label
    @tf.function
    def labeler(line):
        # Split the line by tokens
        splited_tensor = tf.strings.split(line)
        # Get the tokens of the tweet
        tweet = tf.strings.reduce_join(splited_tensor[:-1], separator=' ')
        # Get the label token. The last one
        label = splited_tensor[-1]
        return tf.cast(tweet, tf.string), tf.strings.to_number(label, out_type=tf.int64) 

    # Get the data from a text file with N rows where each row has a tweet and his label separated by a space
    lines_dataset = tf.data.TextLineDataset(data_file)
    # Separate the tweet and the label with the previous function 'labeler'
    lines_dataset = lines_dataset.map(lambda line: labeler(line), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    '''
    Sentence tokenization
    '''
    # Build the tokenizer
    tokenizer = tfds.features.text.Tokenizer()
    # Build the vocabulary from the data
    vocabulary_set = set()
    for text_tensor, _ in lines_dataset:
        some_tokens = tokenizer.tokenize(text_tensor.numpy())
        vocabulary_set.update(some_tokens)
	
    # Build the encoder from the vocabulary
    encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)
    # Store the encoder for testing
    encoder.save_to_file(f"models/encoder_{lang}")
    
    # Encoder fuction to apply to the samples
    def encode(text_tensor, label):
        encoded_text = encoder.encode(text_tensor.numpy())
        return encoded_text, label

    # Encoder function wraper
    def encode_map_fn(text, label):
        # py_func doesn't set the shape of the returned tensors.
        encoded_text, label = tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))

        # `tf.data.Datasets` work best if all components have a shape set
        #  so set the shapes manually: 
        encoded_text.set_shape([None])
        label.set_shape([])

        return encoded_text, label

    # Apply the encoder to the data
    lines_dataset = lines_dataset.map(encode_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # To save transformations and not repeat them
    lines_dataset = lines_dataset.cache()
    '''
    Batch preparation
    '''
    # Shuffle the data. Can be set to shuffle at the end of each epoch
    lines_dataset = lines_dataset.shuffle(buffer_size, reshuffle_each_iteration=shuffle_at_end)
    # Get the number of batches per epoch
    num_batches = len(list(lines_dataset)) // batch_size
    # Prepare padded batches in order to have sentences of the same length
    lines_dataset = lines_dataset.padded_batch(batch_size=batch_size, padded_shapes=([None], []))
    # Repeat dataset for each epoch
    lines_dataset = lines_dataset.repeat()
    # Prefetch buffer for the batches to improve speed
    lines_dataset = lines_dataset.prefetch(prefetch_batches)

    return lines_dataset, vocabulary_set, num_batches


'''
Data generator that returns the tweets encoded word by word
Params:
    tweets -> list with the tweet's strings
    lang -> "en" or "es" string with the laguage of the data
'''
def encoded_datagen_inference(tweets, lang):

    # Create the Dataset with the tweets
    tweets_dataset = tf.data.Dataset.from_tensor_slices(tweets)

    '''
    Sentence tokenization
    '''
    # Build the encoder from the vocabulary
    encoder = tfds.features.text.TokenTextEncoder.load_from_file(f"models/encoder_{lang}")
    
    # Encoder fuction to apply to the samples
    def encode(text_tensor):
        encoded_text = encoder.encode(text_tensor.numpy())
        return [encoded_text]

    # Encoder function wraper
    def encode_map_fn(text):
        # py_func doesn't set the shape of the returned tensors.
        encoded_text = tf.py_function(encode, inp=[text], Tout=(tf.int64))

        # `tf.data.Datasets` work best if all components have a shape set
        #  so set the shapes manually: 
        encoded_text.set_shape([None])

        return encoded_text

    # Apply the encoder to the data
    tweets_dataset = tweets_dataset.map(encode_map_fn)
    '''
    Batch preparation
    '''
    # Batch_size is one for later calculation fo the author label
    tweets_dataset = tweets_dataset.batch(1)

    return tweets_dataset


'''
Data generator that returns the tweets in a string
Params:
    data_file -> Is a path to a txt file with N rows where each row has a tweet and the label (0 or 1) separated by a space.
    batch_size -> Number of samples per batch created by the generator
    shuffle_at_end -> To shuffle the samples at the end of each epoch
    buffer_size -> buffer size for shuffling the data
    prefetch_batches -> max number of batches to prefetch. set to 'tf.data.experimental.AUTOTUNE' for automatic selection
'''
def text_datagen(data_file, batch_size=64, shuffle_at_end=True, buffer_size=100, prefetch_batches=tf.data.experimental.AUTOTUNE):

    # Function so separate the tweet string from the label
    @tf.function
    def labeler(line):
        # Split the line by tokens
        splited_tensor = tf.strings.split(line)
        # Get the tokens of the tweet
        tweet = tf.strings.reduce_join(splited_tensor[:-1], separator=' ')
        # Get the label token. The last one
        label = splited_tensor[-1]
        return tf.cast(tweet, tf.string), tf.strings.to_number(label, out_type=tf.int64) 

    # Get the data from a text file with N rows where each row has a tweet and his label separated by a space
    lines_dataset = tf.data.TextLineDataset(data_file)
    # Separate the tweet and the label with the previous function 'labeler'
    lines_dataset = lines_dataset.map(lambda line: labeler(line), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # To save transformations and not repeat them
    lines_dataset = lines_dataset.cache()
    '''
    Batch preparation
    '''
    # Shuffle the data. Can be set to shuffle at the end of each epoch
    lines_dataset = lines_dataset.shuffle(buffer_size, reshuffle_each_iteration=shuffle_at_end)
    # Get the number of batches per epoch
    num_batches = len(list(lines_dataset)) // batch_size
    # Prepare padded batches in order to have sentences of the same length
    lines_dataset = lines_dataset.batch(batch_size=batch_size)
    # Repeat dataset for each epoch
    lines_dataset = lines_dataset.repeat()
    # Prefetch buffer for the batches to improve speed
    lines_dataset = lines_dataset.prefetch(prefetch_batches)

    return lines_dataset, num_batches


class Test_datagen(keras.utils.Sequence):

    def __init__(self, samples_path, batch_size):
        self.batch_size = batch_size
        self.files = [os.path.join(samples_path, f) for f in os.listdir(samples_path) if f.endswith(".xml")]

    def __len__(self):
        return math.ceil(len(self.files)/self.batch_size)

    def __getitem__(self, idx):
        files_batch = self.files[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        X = []
        for f_path in files_batch:
            for tweet in get_tweets(f_path):
                X.append(tweet)

        return np.array(X)
