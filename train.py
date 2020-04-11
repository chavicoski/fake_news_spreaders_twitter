import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="1";  
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import LearningRateScheduler
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
lang = 'en'

if lang == 'en':
    train_data, dev_data = en_train_data, en_dev_data
elif lang == 'es':
    train_data, dev_data = es_train_data, es_dev_data

batch_size = 256
epochs = 10000
# Select model architecture
model_number = 4

# Build the selected model
print(f"Building model {model_number} for language {lang}...")
if model_number in [0, 2, 3, 4]:
    # To enable the fine tuning of the pretrained embedding
    trainable_embedding = True
    # Build the data generators for train and dev
    train_datagen, num_batches_train = text_datagen(train_data, batch_size=batch_size)
    dev_datagen, num_batches_dev = text_datagen(dev_data, batch_size=batch_size)
    # Create the model
    if model_number == 0:
        model = model_0(lang, trainable_embedding, downloaded=True)
    elif model_number == 2:
        model = model_2(lang, trainable_embedding, downloaded=True)
    elif model_number == 3:
        model = model_3(lang, trainable_embedding, downloaded=True)
    elif model_number == 4:
        model = model_4(lang, trainable_embedding, downloaded=True)

elif model_number == 1:
    # Build the data generators for train and dev
    train_datagen, vocab_train, num_batches_train = encoded_datagen(train_data, batch_size=batch_size)
    dev_datagen, vocab_dev, num_batches_dev = encoded_datagen(dev_data, batch_size=batch_size)
    # Compute the vocabulary size for the embedding layer. We add 1 because of the padding character
    vocab_size = len(vocab_train.union(vocab_dev)) + 1
    # Create the model
    model = model_1(vocab_size)

# Print the model 
model.summary()

# Optimizer 
opt = SGD(learning_rate=0.0002, momentum=0.9)
#opt = Adam(learning_rate=0.0002)

# Learning rate scheduler
def lr_scheduler(epoch):
    if epoch < 1000:
        return 0.0002
    elif epoch < 5000:
        return 0.0001
    else:
        return 0.00005

# Callbacks
callbacks = []

callbacks.append(LearningRateScheduler(lr_scheduler))

ckpt_path = os.path.join(saved_models_path, f"model_{model_number}-{lang}_slowlr_BN.ckpt")
callbacks.append(keras.callbacks.ModelCheckpoint(
        ckpt_path, 
        save_weights_only=(model_number != 1), 
        save_best_only=True,
        monitor="val_loss",
        verbose=1))

# Compile the model
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"]) 

# Train
history = model.fit(train_datagen, epochs=epochs, validation_data=dev_datagen, callbacks=callbacks, steps_per_epoch=num_batches_train, validation_steps=num_batches_dev)

# Load the model from the best epoch
print(f"Loading best model from training...")
# Check the type of saved model (weights or full model)
if model_number in [0, 2, 3, 4]:
    # Create the model architecture to load the saved weights in it
    if model_number == 0:
        loaded_model = model_0(lang, downloaded=True)
    elif model_number == 2:
        loaded_model = model_2(lang, downloaded=True)
    elif model_number == 3:
        loaded_model = model_3(lang, downloaded=True)
    elif model_number == 4:
        loaded_model = model_4(lang, downloaded=True)

    # Compile the model
    loaded_model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"]) 
    # Load the trained weights
    loaded_model.load_weights(ckpt_path)

elif model_number == 1:
    loaded_model = keras.models.load_model(ckpt_path)

# Evaluate with development partition
print(f"Evaluating the model...")
loss, acc = loaded_model.evaluate(dev_datagen, steps=num_batches_dev)
print(f"Best model results: loss: {loss:.3f}, acc: {acc:.3f}")
