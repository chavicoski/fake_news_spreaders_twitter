import os
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from lib.data_generators import text_datagen, encoded_datagen
from models.my_models import *

# Data paths
data_path = "../data"
en_train_data = os.path.join(data_path, "en", "train_tweets.txt")
en_dev_data = os.path.join(data_path, "en", "dev_tweets.txt")
es_train_data = os.path.join(data_path, "es", "train_tweets.txt")
es_dev_data = os.path.join(data_path, "es", "dev_tweets.txt")

# Select language for training
lang = 'es'

if lang == 'en':
    train_data, dev_data = en_train_data, en_dev_data
elif lang == 'es':
    train_data, dev_data = es_train_data, es_dev_data

batch_size = 256
epochs = 1000
# Select model architecture
model_number = 0

# Build the selected model
print(f"Building model {model_number}...")
if model_number == 0:
    # Build the data generators for train and dev
    train_datagen = text_datagen(train_data, batch_size=batch_size)
    dev_datagen = text_datagen(dev_data, batch_size=batch_size)
    # Create the model
    model = model_0(lang)

elif model_number == 1:
    # Build the data generators for train and dev
    train_datagen, vocab_train = encoded_datagen(train_data, batch_size=batch_size)
    dev_datagen, vocab_dev = encoded_datagen(dev_data, batch_size=batch_size)
    # Compute the vocabulary size for the embedding layer. We add 1 because of the padding character
    vocab_size = len(vocab_train.union(vocab_dev)) + 1
    # Create the model
    model = model_1(vocab_size)

# Print the model 
model.summary()

# Optimizer 
opt = SGD(learning_rate=0.01, momentum=0.9)
#opt = Adam(learning_rate=0.001)

# Compile the model
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

# Train
history = model.fit(train_datagen, epochs=epochs, validation_data=dev_datagen)

