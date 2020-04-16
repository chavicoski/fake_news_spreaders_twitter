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
from tqdm import tqdm
from matplotlib import pyplot as plt

# Paths
test_data_path = "../data/test/"
en_test_data = os.path.join(test_data_path, "en")
es_test_data = os.path.join(test_data_path, "es")
en_es_test_data = os.path.join(test_data_path, "en+es")
saved_models_path = "models/checkpoints"

# Select language to test
lang = "en+es"
# Select model to test
model_number = 7
# Get the path to the trained model
slow_name = f"model_{model_number}-{lang}_slowlr_BN"
fast_name = f"model_{model_number}-{lang}_fastlr"
adam_name = f"model_{model_number}-{lang}_Adam"
base_name = f"model_{model_number}-{lang}"
fine_name = f"model_{model_number}-{lang}_fine"

ckpt_name = fine_name
ckpt_path = os.path.join(saved_models_path, ckpt_name + ".ckpt")

# Build the selected model to load the weights
if model_number in [0, 2, 3, 4, 5, 6, 7]:
    if model_number == 0:
        model = model_0(lang, downloaded=True)
    elif model_number == 2:
        model = model_2(lang, downloaded=True)
    elif model_number == 3:
        model = model_3(lang, downloaded=True)
    elif model_number == 4:
        model = model_4(lang, downloaded=True)
    elif model_number == 5:
        model = model_5(downloaded=True)
    elif model_number == 6:
        model = model_6(downloaded=True)
    elif model_number == 7:
        # Trained models paths
        en_ckpt_path = "models/checkpoints/model_5-en_slowlr_BN.ckpt"
        es_ckpt_path = "models/checkpoints/model_5-es_slowlr_BN.ckpt"
        model = model_7(en_ckpt_path, es_ckpt_path, downloaded=True)
    
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
elif lang == "en+es": test_data = en_es_test_data

# Load labels
with open(os.path.join(test_data, "labels_dict.pickle"), "rb") as handle:
    labels_dict = pickle.load(handle)


print(f"Going to run test with model {model_number} and language {lang}")

'''
Prediction
'''
classification_threshold = 0.5
total_authors = 0
fake_miss_prod = 0
true_miss_prod = 0
fake_miss_avg = 0
true_miss_avg = 0
# Lists to store all the predictions and labels to make plots
preds = []
labels = []
for f_name in tqdm(os.listdir(test_data)):
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
        # Get the label
        true_label = labels_dict[author_id]

        '''
        Compute voting by product
        '''
        # Take the probs to make the product voting
        fake_probs = results.flatten()
        true_probs = 1 - fake_probs
        # Multiply the probs and take the class that maximizes the prob
        fake_prob = fake_probs.prod()
        true_prob = true_probs.prod()
        # Check voting result
        if true_prob > fake_prob:
            if true_label != 0: true_miss_prod += 1
        else:
            if true_label != 1: fake_miss_prod += 1
       
        
        # Store results for plots
        preds += list(fake_probs)
        labels += [labels_dict[author_id]] * len(tweets)

        '''
        Compute voting by average
        '''
        # Store the voting results
        fake_prob = results.mean()
        if fake_prob > classification_threshold: 
            if true_label != 1: fake_miss_avg += 1
        else:
            if true_label != 0: true_miss_avg += 1

        total_authors += 1

# Show average voting restuls
hits = total_authors - (fake_miss_avg+true_miss_avg)
acc = hits / total_authors
print(f"Average voting: acc = {acc:.2f} -> true_misses: {true_miss_avg}, fake_misses: {fake_miss_avg}, hits: {hits}, total: {total_authors}")

# Show prodcut voting restuls
hits = total_authors - (fake_miss_prod+true_miss_prod)
acc = hits / total_authors
print(f"Product voting: acc = {acc:.2f} -> true_misses: {true_miss_prod}, fake_misses: {fake_miss_prod}, hits: {hits}, total: {total_authors}")

# Make plots
preds = np.array(preds)
labels = np.array(labels)
# Take the preds for each class
preds_true = preds[labels == 0]
preds_fake = preds[labels == 1]
# Make the histogram
plt.hist([preds_true, preds_fake], bins=30, range=(0, 1), label=["true", "fake"])
plt.legend(loc="upper right")
plt.xlabel("Fake score")
plt.ylabel("Count")
plt.title("Histogram of predicted probabilities for Fake and not Fake tweets")
# Save histogram
plt.savefig(f"plots/{ckpt_name}.png")
