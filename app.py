import nltk
import os
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle

# here we simply LOAD the data of our JSON using json.load()
with open("intents.json") as file:
    data=json.load(file)


# empty lists to store tokens
words = []
labels = []
docs_x = []
docs_y = []


# Loop through the loaded json file named data. The bracket specifies what exactly in that file we are interested in exploring.
for intents in data["intents"]:
    # The following loops through 'patterns' in intents.json
    # 'Patterns' are words identifiable by the machine and
    # these words are to interpretable by the machine.
    # The machine responds to these patterns according to
    # the tag they belong to. 
    for pattern in intents['patterns']:
        pattern = pattern.lower()
    		#creating a list of words
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intents['tag'])
        
    if intents['tag'] not in labels:
        labels.append(intents['tag'])