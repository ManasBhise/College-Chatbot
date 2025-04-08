# Importing modules
import json  # loads the intents file
import numpy as np  # for numerical operations
import tensorflow as tf  # to build and train model
from tensorflow.keras.models import Sequential  # model for training
from tensorflow.keras.layers import Dense, Dropout  # layers for neural network
from sklearn.preprocessing import LabelEncoder  # to convert text labels into numbers
import nltk  # natural language toolkit
from nltk.tokenize import word_tokenize

# Download the tokenizer
nltk.download('punkt')

from preprocessing.preprocessing import preprocess_text  # custom text preprocessing

# Load the intents JSON data
with open("intents.json") as file:
    data = json.load(file)

all_words = []
tags = []
xy = []

# Extract data from intents
for intent in data['intents']:
    tag = intent['intent']  # changed from 'tag' to 'intent' to match your JSON structure
    tags.append(tag)  # FIXED: was `tags.append(tags)` which caused list-of-lists
    for pattern in intent['text']:  # input phrases
        tokens = preprocess_text(pattern)
        all_words.extend(tokens)
        xy.append((tokens, tag))

# Remove duplicates and sort
vocab = sorted(set(all_words))
tags = sorted(set(tags))

# Create training data
X_train = []
Y_train = []

for (pattern_tokens, tag) in xy:  # ❗ FIXED: changed second var from `tags` to `tag`
    bag = [1 if word in pattern_tokens else 0 for word in vocab]
    X_train.append(bag)
    label = tags.index(tag)
    Y_train.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

# Build the model
model = Sequential()
model.add(Dense(128, input_shape=(len(X_train[0]),), activation='relu'))  # ❗ FIXED: missing comma tuple
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(tags), activation='softmax'))

# ❗ FIXED: 'complie' → 'compile'
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train, epochs=200, batch_size=8)

# Save model and metadata
model.save('model/chatbot_model.h5')

# Save vocab and tags using pickle
import pickle
with open("model/vocab.pkl", 'wb') as f:
    pickle.dump(vocab, f)

with open('model/tags.pkl', 'wb') as f:
    pickle.dump(tags, f)

print("Model training is complete!")
