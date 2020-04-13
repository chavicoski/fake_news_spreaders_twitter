import os
from shutil import copy
import xml.etree.ElementTree as ET
from lib.utils import get_tweets
import pickle

'''
The data in en+es folder has been moved manually from the en and es folders. In this script
we are building the binary of the labels dict for the en+es partition in test folder
'''

# Path to the folder with all the data subfolders for each language
full_data_path = "../data"
# Path to the folder with the data
en_es_data_path = os.path.join(full_data_path, "en+es")
# Path to the new folder with test data
test_data_path = "../data/test"
# Path to the target folder for test data
en_es_test_path = os.path.join(test_data_path, "en+es")

# Create a dict with the labels for each user
labels_path = os.path.join(en_es_data_path, "truth.txt")
labels_dict = {}
with open(labels_path) as labels:
    for line in labels:
        user_id, label = line.split(":::")
        labels_dict[user_id] = int(label)

# Save the dict in disk
with open(os.path.join(en_es_test_path, "labels_dict.pickle"), "wb") as handle:
    pickle.dump(labels_dict, handle)
