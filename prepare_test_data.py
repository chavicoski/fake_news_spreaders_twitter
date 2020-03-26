import os
from shutil import copy
import xml.etree.ElementTree as ET
from lib.utils import get_tweets
import pickle

'''
This code takes the xml files for the authors in the development partition made by "prepare_data.py"
and store them in a new folder "test" separated by "es" and "en". This is made for simulating the 
clasification of the authors directly from the xml, simulating the use case of the evaluation of the 
competition.
'''

# Path to the folder with all the data subfolders for each language
full_data_path = "../data"
# Path to each laguage folder in test data
es_data_path = os.path.join(full_data_path, "es")
en_data_path = os.path.join(full_data_path, "en")
# Path to the new folder with test data
test_data_path = "../data/test"
# Path to each laguage folder in test data
es_test_path = os.path.join(test_data_path, "es")
en_test_path = os.path.join(test_data_path, "en")

# Percentaje for the development set
dev_partition = 0.2  # This must be the same value that appears in the "prepare_data.py" script
# List of all the laguage folders with data and the corresponding test path
dataset = [(es_data_path, es_test_path), (en_data_path, en_test_path)]


for data_path, test_path in dataset:
    # Check and create the test folder for this language
    if not os.path.exists(test_path):
        os.makedirs(test_path, exist_ok=True)

    # Create a dict with the labels for each user
    labels_path = os.path.join(data_path, "truth.txt")
    labels_dict = {}
    with open(labels_path) as labels:
        for line in labels:
            user_id, label = line.split(":::")
            labels_dict[user_id] = int(label)

    # Save the dict in disk
    with open(os.path.join(test_path, "labels_dict.pickle"), "wb") as handle:
        pickle.dump(labels_dict, handle)

    # Get the list of all the author xml files
    all_authors = [author_xml for author_xml in os.listdir(data_path) if author_xml.endswith(".xml")]
    # Get the total number of samples for the dev set
    n_dev_samples = int(len(all_authors)*dev_partition)
    # Take the dev partition
    dev_authors = all_authors[-n_dev_samples:]
   
    # Copy the xml files
    for f_name in dev_authors:
        copy(os.path.join(data_path, f_name), os.path.join(test_path, f_name))
