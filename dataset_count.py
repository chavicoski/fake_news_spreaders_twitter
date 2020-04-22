import os
import sys

# Path to the folder with all the data subfolders for each language
full_data_path = "../data"
# Path to each laguage folder
es_data_path = os.path.join(full_data_path, "es")
en_data_path = os.path.join(full_data_path, "en")

#####################################
# COUNT OF LABELS FOR SINGLE TWEETS #
#####################################

tr_labels_count = {"es": [0, 0], "en": [0, 0]}
dev_labels_count = {"es": [0, 0], "en": [0, 0]}

for lang, data_path in [("es", es_data_path), ("en", en_data_path)]:
    with open(os.path.join(data_path, "train_tweets.txt"), "r") as tr_tweets:
        for tweet in tr_tweets.readlines():
            label = int(tweet.split(" ")[-1])
            if label:
                tr_labels_count[lang][1] += 1
            else:
                tr_labels_count[lang][0] += 1

    with open(os.path.join(data_path, "dev_tweets.txt"), "r") as tr_tweets:
        for tweet in tr_tweets.readlines():
            label = int(tweet.split(" ")[-1])
            if label:
                dev_labels_count[lang][1] += 1
            else:
                dev_labels_count[lang][0] += 1


print("Single tweet counts:")
for split, labels_count in [("tr", tr_labels_count), ("dev", dev_labels_count)]:
    print(split, ":")
    for lang, count in labels_count.items():
        total = sum(count)
        print(f"\t{lang} -> true: {count[0]}({count[0]/total:.2f}), fake: {count[1]}({count[1]/total:.2f})")

###############################
# COUNT OF LABELS FOR AUTHORS #
###############################

labels_count = {"es": [0, 0], "en": [0, 0]}

for lang, data_path in [("es", es_data_path), ("en", en_data_path)]:
    with open(os.path.join(data_path, "truth.txt"), "r") as authors:
        for author in authors.readlines():
            label = int(author.split(":::")[-1])
            if label:
                labels_count[lang][1] += 1
            else:
                labels_count[lang][0] += 1


print("Authors counts:")
for lang, count in labels_count.items():
    total = sum(count)
    print(f"\t{lang} -> true: {count[0]}({count[0]/total:.2f}), fake: {count[1]}({count[1]/total:.2f})")
