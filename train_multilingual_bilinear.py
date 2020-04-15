import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0";  
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from lib.data_generators import text_datagen, encoded_datagen
from models.my_models import *

# Paths
data_path = "../data"
# Get the data from the en+es folder
train_data = os.path.join(data_path, "en+es", "train_tweets.txt")
dev_data = os.path.join(data_path, "en+es", "dev_tweets.txt")
saved_models_path = "models/checkpoints"

lang = "en+es"
batch_size = 256
epochs = 5000
model_number = 7

#################
# TRAIN PHASE 1 #
#################

# Build the selected model
print(f"Building model {model_number} for language {lang}...")
# Build the data generators for train and dev
train_datagen, num_batches_train = text_datagen(train_data, batch_size=batch_size)
dev_datagen, num_batches_dev = text_datagen(dev_data, batch_size=batch_size)
# Trained models paths
en_ckpt_path = "models/checkpoints/model_5-en.ckpt"
es_ckpt_path = "models/checkpoints/model_5-es.ckpt"

# Create the model
if model_number == 7:
    # Create a model with the pretrained part frozen to train the new layers first
    frozen_model = model_7(en_ckpt_path, es_ckpt_path, downloaded=True, frozen=True)
else:
    print("The model {model_number} is not a valid model.")

# Print the model 
frozen_model.summary()

# Optimizer 
opt = SGD(learning_rate=0.001, momentum=0.9)
#opt = Adam(learning_rate=0.0002)

# Learning rate scheduler
def lr_scheduler(epoch):
    if epoch < 100:
        return 0.001
    elif epoch < 500:
        return 0.0005
    else:
        return 0.0001

# Callbacks
callbacks = []

callbacks.append(LearningRateScheduler(lr_scheduler))

frozen_ckpt_name = f"model_{model_number}-{lang}_frozen.ckpt"
frozen_ckpt_path = os.path.join(saved_models_path, frozen_ckpt_name)
callbacks.append(keras.callbacks.ModelCheckpoint(
        frozen_ckpt_path, 
        save_weights_only=True, 
        save_best_only=True,
        monitor="val_loss",
        verbose=1))

# Compile the model
frozen_model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"]) 

# Train
frozen_history = frozen_model.fit(train_datagen, epochs=epochs, validation_data=dev_datagen, callbacks=callbacks, steps_per_epoch=num_batches_train, validation_steps=num_batches_dev)

#################
# TRAIN PHASE 2 #
#################

# Load the model from the best epoch
print(f"Loading best model from frozen training phase for fine dotuning...")
# Create the model architecture to load the saved weights in it
if model_number == 7:
    fine_model = model_7(en_ckpt_path, es_ckpt_path, downloaded=True, frozen=False)

# Compile the model
fine_model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"]) 
# Load the trained weights
fine_model.load_weights(frozen_ckpt_path)
# Plot the model
fine_model.summary()

# Optimizer 
opt = SGD(learning_rate=0.001, momentum=0.9)
#opt = Adam(learning_rate=0.0002)

# Learning rate scheduler
def lr_scheduler(epoch):
    if epoch < 100:
        return 0.001
    elif epoch < 500:
        return 0.0005
    else:
        return 0.0001

# Callbacks
callbacks = []

callbacks.append(LearningRateScheduler(lr_scheduler))

fine_ckpt_name = f"model_{model_number}-{lang}_fine.ckpt"
fine_ckpt_path = os.path.join(saved_models_path, fine_ckpt_path)
callbacks.append(keras.callbacks.ModelCheckpoint(
        fine_ckpt_path, 
        save_weights_only=True, 
        save_best_only=True,
        monitor="val_loss",
        verbose=1))

# Train
fine_history = fine_model.fit(train_datagen, epochs=epochs, validation_data=dev_datagen, callbacks=callbacks, steps_per_epoch=num_batches_train, validation_steps=num_batches_dev)

############
# EVALUATE #
############

# Load the model from the best epoch
print(f"Loading best model from training...")
# Create the model architecture to load the saved weights in it
if model_number == 7:
    loaded_model = model_7(en_ckpt_path, es_ckpt_path, downloaded=True)

# Compile the model
loaded_model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"]) 
# Load the trained weights
loaded_model.load_weights(fine_ckpt_path)

# Evaluate with development partition
print(f"Evaluating the model...")
loss, acc = loaded_model.evaluate(dev_datagen, steps=num_batches_dev)
print(f"Best model results: loss: {loss:.3f}, acc: {acc:.3f}")
