import nltk, os
nltk.download('punkt')

from nltk import word_tokenize,sent_tokenize
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import random
import json
import pickle
from datetime import datetime
from flask import Flask, render_template, request, session, jsonify
import os

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Required for session management

# Load intents
with open("intents.json") as file:
    data = json.load(file)

# Always retrain the model to ensure it's up to date with intents.json
words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
        
    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))
labels = sorted(labels)

training = []
output = []
out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []
    wrds = [stemmer.stem(w.lower()) for w in doc]
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1
    
    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

# Save the processed data
with open("data.pickle","wb") as f:
    pickle.dump((words, labels, training, output), f)

# Create model using Keras
model = Sequential()
model.add(Dense(16, input_shape=(len(training[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(output[0]), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(training, output, epochs=1000, batch_size=8, verbose=1)
model.save_weights('model.weights.h5')

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

def get_context_aware_response(user_input, context):
    # Add context to the input
    context_input = f"{context} {user_input}" if context else user_input
    
    result = model.predict(np.array([bag_of_words(context_input, words)]))[0]
    result_index = np.argmax(result)
    tag = labels[result_index]

    if result[result_index] > 0.7:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        return random.choice(responses), tag
    else:
        return "I didnt get that. Can you explain or try again.", None

@app.route('/')
def index():
    # Initialize session variables if they don't exist
    if 'conversation_history' not in session:
        session['conversation_history'] = []
    if 'current_context' not in session:
        session['current_context'] = ""
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    user_input = data.get('message', '').strip()
    
    # Get response with context
    response, tag = get_context_aware_response(user_input, session.get('current_context', ''))
    
    # Update context based on the current interaction
    if tag:
        session['current_context'] = tag
    else:
        session['current_context'] = ""
    
    # Add to conversation history
    conversation_entry = {
        'user_input': user_input,
        'bot_response': response,
        'timestamp': datetime.now().strftime("%H:%M")
    }
    
    if 'conversation_history' not in session:
        session['conversation_history'] = []
    
    session['conversation_history'].append(conversation_entry)
    
    # Keep only the last 10 messages in history
    if len(session['conversation_history']) > 10:
        session['conversation_history'] = session['conversation_history'][-10:]
    
    # Save the session
    session.modified = True
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
