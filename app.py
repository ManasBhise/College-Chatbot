from flask import Flask, render_template, request, jsonify
import numpy as np
import random
import pickle
import json
from tensorflow.keras.models import load_model
from preprocessing.preprocessing import preprocess_text

app = Flask(__name__)

# Load trained stuff
model = load_model("model/chatbot_model.h5")
with open("model/vocab.pkl", 'rb') as f:
    vocab = pickle.load(f)
with open("model/tags.pkl", 'rb') as f:
    tags = pickle.load(f)
with open("intents.json") as f:
    intents = json.load(f)

# Predict function
def predict_class(sentence):
    tokens = preprocess_text(sentence)
    bag = [1 if word in tokens else 0 for word in vocab]
    res = model.predict(np.array([bag]))[0]
    threshold = 0.7
    results = [(i, r) for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return tags[results[0][0]] if results else "noanswer"

# Get response
def get_response(tag):
    for intent in intents['intents']:
        if intent['intent'] == tag:
            return random.choice(intent['responses'])
    return "I'm not sure I understand."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    user_input = request.form["msg"]
    tag = predict_class(user_input)
    response = get_response(tag)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
