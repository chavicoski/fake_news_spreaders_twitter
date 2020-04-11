import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="1";  
import tensorflow as tf
from tensorflow import keras
from models.my_models import *
from lib.data_generators import encoded_datagen_inference
from lib.data_generators import Test_datagen
from lib.utils import get_tweets
import pickle

# Paths
test_data_path = "../data/test/"
en_test_data = os.path.join(test_data_path, "en")
es_test_data = os.path.join(test_data_path, "es")
saved_models_path = "models/checkpoints"

# Select language to test
lang = "es"
# Select model to test
model_number = 4
# Get the path to the trained model
ckpt_path = os.path.join(saved_models_path, f"model_{model_number}-{lang}_slowlr_BN.ckpt")

# Build the selected model to load the weights
if model_number in [0, 2, 3, 4]:
    if model_number == 0:
        model = model_0(lang, downloaded=True)
    elif model_number == 2:
        model = model_2(lang, downloaded=True)
    elif model_number == 3:
        model = model_3(lang, downloaded=True)
    elif model_number == 4:
        model = model_4(lang, downloaded=True)
    
    # Compile the model
    model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"]) 
    # Load the trained weights in the model
    model.load_weights(ckpt_path)

if model_number == 1:
    # Load the trained model
    model = keras.models.load_model(ckpt_path)
    # Compile the model
    model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"]) 

# Select language for data
if lang == "es": test_data = es_test_data
elif lang == "en": test_data = en_test_data

# Load labels
with open(os.path.join(test_data, "labels_dict.pickle"), "rb") as handle:
    labels_dict = pickle.load(handle)

'''
Prediction
'''
classification_threshold = 0.5
total_authors = 0
fake_miss = 0
true_miss = 0
for f_name in os.listdir(test_data):
    if f_name.endswith(".xml"):
        # Get data from file
        author_id = f_name[:-4]
        f_path = os.path.join(test_data, f_name)

        # Build mini-dataset for the author
        tweets = list(get_tweets(f_path))
        if model_number == 1:
            tweets_dataset = encoded_datagen_inference(tweets)
        else:
            tweets_dataset = tf.data.Dataset.from_tensor_slices(tweets)
            tweets_dataset = tweets_dataset.batch(1)
 
        #print("TESTING DATAGEN")
        #for elem in tweets_dataset:
        #    print(elem.numpy())

        # Predict
        results = model.predict(tweets_dataset, steps=len(tweets))
        fake_prob = results.mean()
        true_label = labels_dict[author_id]
        if fake_prob > classification_threshold: 
            if true_label != 1: fake_miss += 1
        else:
            if true_label != 0: true_miss += 1

        total_authors += 1

hits = total_authors - (fake_miss+true_miss)
acc = hits / total_authors
print(f"Accuracy: {acc:.2f} -> true_misses: {true_miss}, fake_misses: {fake_miss}, hits: {hits}, total: {total_authors}")
