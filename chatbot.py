# Import libraries
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import random
from preprocessing.preprocessing import preprocess_text

# Load the trained model
model = load_model('model/chatbot_model.h5')

# Load vocab and tags
with open('model/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

with open('model/tags.pkl', 'rb') as f:
    tags = pickle.load(f)

# Load intents
with open('intents.json') as f:
    intents = json.load(f)

# Function to convert input to a bag-of-words vector
def bag_of_words(sentence, vocab):
    tokens = preprocess_text(sentence)
    bag = [1 if word in tokens else 0 for word in vocab]
    return np.array(bag)

# Function to get chatbot response
def get_response(user_input):
    bow = bag_of_words(user_input, vocab)
    prediction = model.predict(np.array([bow]))[0]
    index = np.argmax(prediction)
    tag = tags[index]

    # Confidence check
    if prediction[index] > 0.7:
        for intent in intents['intents']:
            if intent['intent'] == tag:
                return random.choice(intent['responses'])
    else:
        return "Sorry, I didn't get that. Can you rephrase?"

# Chat loop
print("Chatbot is ready! Type 'quit' to stop.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Chatbot: Bye! 👋")
        break
    response = get_response(user_input)
    print("Chatbot:", response)
