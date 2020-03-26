import os
import xml.etree.ElementTree as ET

def get_tweets(author_xml_file):
    tree = ET.parse(author_xml_file) 
    root = tree.getroot()
    for tweet in root.iter("document"):
        tweet_text = tweet.text.replace('\n', ' ').lower()
        yield tweet_text

# Test functions
if __name__ == "__main__":
    
    for tweet in get_tweets("../data/es/139a0345f3a3a1922cc08b887e5cc49.xml"):
        print(tweet)
