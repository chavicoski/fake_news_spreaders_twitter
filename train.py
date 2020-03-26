import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam
from lib.data_generators import text_datagen, encoded_datagen
from models.my_models import *

# Paths
data_path = "../data"
en_train_data = os.path.join(data_path, "en", "train_tweets.txt")
en_dev_data = os.path.join(data_path, "en", "dev_tweets.txt")
es_train_data = os.path.join(data_path, "es", "train_tweets.txt")
es_dev_data = os.path.join(data_path, "es", "dev_tweets.txt")
saved_models_path = "models/checkpoints"

# Select language for training
lang = 'es'

if lang == 'en':
    train_data, dev_data = en_train_data, en_dev_data
elif lang == 'es':
    train_data, dev_data = es_train_data, es_dev_data

batch_size = 256
epochs = 200
# Select model architecture
model_number = 2

# Build the selected model
print(f"Building model {model_number}...")
if model_number == 0:
    # To enable the fine tuning of the pretrained embedding
    trainable_embedding = True
    # Build the data generators for train and dev
    train_datagen, num_batches_train = text_datagen(train_data, batch_size=batch_size)
    dev_datagen, num_batches_dev = text_datagen(dev_data, batch_size=batch_size)
    # Create the model
    model = model_0(lang, trainable_embedding)

elif model_number == 1:
    # Build the data generators for train and dev
    train_datagen, vocab_train, num_batches_train = encoded_datagen(train_data, batch_size=batch_size)
    dev_datagen, vocab_dev, num_batches_dev = encoded_datagen(dev_data, batch_size=batch_size)
    # Compute the vocabulary size for the embedding layer. We add 1 because of the padding character
    vocab_size = len(vocab_train.union(vocab_dev)) + 1
    # Create the model
    model = model_1(vocab_size)

elif model_number == 2:
    # To enable the fine tuning of the pretrained embedding
    trainable_embedding = True
    # Build the data generators for train and dev
    train_datagen, num_batches_train = text_datagen(train_data, batch_size=batch_size)
    dev_datagen, num_batches_dev = text_datagen(dev_data, batch_size=batch_size)
    # Create the model
    model = model_2(lang, trainable_embedding)

# Print the model 
model.summary()

# Optimizer 
opt = SGD(learning_rate=0.01, momentum=0.9)
#opt = Adam(learning_rate=0.001)

# Callbacks
ckpt_path = os.path.join(saved_models_path, f"model_{model_number}-{lang}.ckpt")
checkpoint = keras.callbacks.ModelCheckpoint(
        ckpt_path, 
        save_weights_only=True, 
        save_best_only=True,
        monitor="val_loss",
        verbose=1)

# Compile the model
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"]) 

# Train
history = model.fit(train_datagen, epochs=epochs, validation_data=dev_datagen, callbacks=[checkpoint], steps_per_epoch=num_batches_train, validation_steps=num_batches_dev)

# Load the model from the best epoch
print(f"Loading best model from training...")
# Create the model architecture to load the saved weights in it
if model_number == 0:
    loaded_model = model_0(lang)
elif model_number == 1:
    loaded_model = model_1(vocab_size)
elif model_number == 2:
    loaded_model = model_2(lang)
# Compile the model
loaded_model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"]) 
# Load the trained weights
loaded_model.load_weights(ckpt_path)

# Evaluate with development partition
print(f"Evaluating the model...")
loss, acc = loaded_model.evaluate(dev_datagen, steps=num_batches_dev)
print(f"Best model results: loss: {loss:.3f}, acc: {acc:.3f}")
