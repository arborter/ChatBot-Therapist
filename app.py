from flask import Flask, render_template, request, jsonify
import nltk
import datetime
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle

stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)
with open("data.pickle","rb") as f:
    words, labels, training, output = pickle.load(f)

#Function to process input
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i,w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

tf.reset_default_graph()

net = tflearn.input_data(shape = [None, len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]), activation = "softmax")
net = tflearn.regression(net)

#Loading existing model from disk
model = tflearn.DNN(net)
model.load("model.tflearn")


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get')
def get_bot_response():
    message = request.args.get('msg')
    if message:
        message = message.lower()
        results = model.predict([bag_of_words(message,words)])[0]
        result_index = np.argmax(results)
        tag = labels[result_index]
        if results[result_index] > 0.5:
            if tag == "help":
                response = "i will do the best i can. tell me what emøcean you feel right now in one word"
                return str(response)
            else:
                for any in data['intents']:
                    if tag == message:
                        responses = any['responses']
                        response = random.choice(responses)
                        return str(response)
        else:
            response = "I didn't quite get that, please try again."
        return str(response)
    return "i need more clarity, too."


if __name__ == "__main__":
    app.run()

