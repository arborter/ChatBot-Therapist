import nltk
import os
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
#import tensorflow as tf
import random
import json
import pickle

with open("intents.json") as file:
    data=json.load(file)
    
