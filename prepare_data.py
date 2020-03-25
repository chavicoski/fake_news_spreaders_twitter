import os
import xml.etree.ElementTree as ET

'''
This code 'prepares' the data by taking the xml files with the tweets for each author
and creates the train and dev partitions by taking the tweets and putting them in two
text files with a row for each tweet. Where each row has the tweet text and the label
(0 or 1) separated by a space. Note that the tweets for each author can only be in one
partition because the partitioning is made at author level.
'''

# Path to the folder with all the data subfolders for each language
data_path = "../data"
# Path to each laguage folder
es_data_path = os.path.join(data_path, "es")
en_data_path = os.path.join(data_path, "en")

# Percentaje for the development set
dev_partition = 0.2
# List of all the laguage folders with data to prepare
dataset = [es_data_path, en_data_path]

for data_path in dataset:
    labels_path = os.path.join(data_path, "truth.txt")
    labels_dict = {}
    with open(labels_path) as labels:
        for line in labels:
            user_id, label = line.split(":::")
            labels_dict[user_id] = int(label)

    # Get the list of all the author xml files
    all_authors = os.listdir(data_path)
    # Get the total number of samples for the dev set
    n_dev_samples = int(len(all_authors)*dev_partition)
    # Make the partitioning in train and dev
    train_authors = all_authors[:-n_dev_samples]
    dev_authors = all_authors[-n_dev_samples:]
    '''
    Create train partition
    '''
    with open(os.path.join(data_path, "train_tweets.txt"), "w+") as preprocessed_f:
        for f_name in train_authors:
            if f_name.endswith(".xml"):
                tree = ET.parse(os.path.join(data_path, f_name)) 
                root = tree.getroot()
                for tweet in root.iter("document"):
                    tweet_text = tweet.text.replace('\n', ' ').lower()
                    label = labels_dict[f_name[:-4]]
                    preprocessed_f.write(f"{tweet_text} {label}\n")

    '''
    Create development partition
    '''
    with open(os.path.join(data_path, "dev_tweets.txt"), "w+") as preprocessed_f:
        for f_name in dev_authors:
            if f_name.endswith(".xml"):
                tree = ET.parse(os.path.join(data_path, f_name)) 
                root = tree.getroot()
                for tweet in root.iter("document"):
                    tweet_text = tweet.text.replace('\n', ' ').lower()
                    label = labels_dict[f_name[:-4]]
                    preprocessed_f.write(f"{tweet_text} {label}\n")
