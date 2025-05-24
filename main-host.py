import nltk, os
from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env file
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
from flask import Flask, render_template, request, session, jsonify, current_app
import google.generativeai as genai

import os

# Configure confidence thresholds from environment variables
NN_CONFIDENCE_THRESHOLD_HIGH = float(os.getenv('NN_CONFIDENCE_THRESHOLD_HIGH', '0.7'))
NN_CONFIDENCE_THRESHOLD_LOW = float(os.getenv('NN_CONFIDENCE_THRESHOLD_LOW', '0.4'))

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

# Load Gemini API key from environment variables
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY environment variable not set.")
    # In a production environment, you might want to raise an exception or handle this more gracefully

# Configure Gemini API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def format_history_for_gemini(history):
    """Formats the conversation history into a list of messages with roles for Gemini API."""
    formatted_history = []
    for entry in history:
        formatted_history.append({'role': 'user', 'parts': [entry['user_input']]})
        formatted_history.append({'role': 'model', 'parts': [entry['bot_response']]})
    return formatted_history

def get_gemini_response(user_input, context):
    if not GEMINI_API_KEY:
        return "I'm currently unable to connect to my external knowledge base. Please try again later."
    model = genai.GenerativeModel(model_name="gemini-pro", system_instruction="You are NeuralNexus, an AI chatbot powered by neural networks and natural language processing. You're intelligent, helpful, and slightly technical in your responses. You enjoy discussing AI concepts and maintain a friendly, professional tone. Keep responses conversational and not too long.")
    conversation = model.start_chat(history=[])
    # Optionally, add context to the history for Gemini
    if context:
        conversation.send_message(f"Previous context: {context}")

    try:
        response = conversation.send_message(user_input)
        return response.text
    except Exception as e:
        current_app.logger.error(f"Gemini API call failed in get_gemini_response: {e}")
        # Fallback: a generic error message or a simple NN response if possible
        return "I'm currently experiencing some difficulties. Could you please rephrase or try again later."

# New function for post-processing responses
def postprocess_response(response_text, source):
    """Applies formatting and ensures consistent personality."""
    # Simple example: Add a prefix based on the source
    # You can expand this later with more sophisticated logic
    if source == 'neural_network':
        return "I'm currently experiencing some difficulties. Could you please rephrase or try again later?"

# New function for Gemini verification/enhancement
def get_verified_response(user_input, nn_response, context):
    """
    Uses Gemini to verify or enhance the neural network response.
    Includes error handling for Gemini API calls.
    """
    if not GEMINI_API_KEY:
        return nn_response # Fallback if API key is not set

    try:
        model = genai.GenerativeModel(model_name="gemini-pro", system_instruction="You are NeuralNexus, an AI chatbot powered by neural networks and natural language processing. You're intelligent, helpful, and slightly technical in your responses. You enjoy discussing AI concepts and maintain a friendly, professional tone. You have received a potential response from another system based on user input, and your task is to verify its relevance or provide a better response if needed. Keep responses conversational and not too long.")
        
        conversation = model.start_chat(history=[])
        # No need to explicitly add context here, the prompt guides Gemini

        verification_response = conversation.send_message(f"User input: {user_input}\nPotential previous response: {nn_response}\nPlease refine or provide a response.")
        return verification_response.text
    except Exception as e:
        current_app.logger.error(f"Gemini verification failed: {e}")
        # In a production environment, you might want to return a more informative message
        return nn_response # Fallback to NN response if Gemini verification fails




def get_context_aware_response(user_input, context):
    # We use the user input directly for NN prediction as it's trained on patterns,
    # context is used to potentially influence the response selection or Gemini call later
    context_input = user_input
    
    result = model.predict(np.array([bag_of_words(context_input, words)]))[0]
    result_index = np.argmax(result)
    tag = labels[result_index]

    if result[result_index] > NN_CONFIDENCE_THRESHOLD_LOW: # Use the lower threshold for initial check
        for tg in data["intents"]:
            if tg['tag'] == tag and result[result_index] > NN_CONFIDENCE_THRESHOLD_HIGH:
                # High confidence: Use pure neural network response
                responses = tg['responses']
                raw_response = random.choice(responses)
                return postprocess_response(raw_response, 'neural_network'), tag
            elif tg['tag'] == tag and result[result_index] >= NN_CONFIDENCE_THRESHOLD_LOW and result[result_index] <= NN_CONFIDENCE_THRESHOLD_HIGH:
                # Medium confidence: Use neural network result + Gemini verification
                nn_response = random.choice(tg['responses'])
                raw_response = get_verified_response(user_input, nn_response, context)
                return postprocess_response(raw_response, 'gemini_verified'), tag # Indicate it was Gemini-verified
    else:
        # Low confidence: Use pure Gemini API response
        raw_response = get_gemini_response(user_input, context)
        return postprocess_response(raw_response, 'gemini'), None # Indicate it was a pure Gemini response

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
