import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the pre-trained LSTM model for next-word prediction
model = load_model('next_word_lstm.h5')

# Load the tokenizer used during training
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word based on the input text
def predict_next_word(model, tokenizer, text, max_sequence_len):
    # Convert the input text into a sequence of tokens
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    # Ensure the token list doesn't exceed the model's maximum input sequence length
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]  # Retain only the last (max_sequence_len - 1) tokens
    
    # Pad the sequence to match the model's expected input shape
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    
    # Predict the next word using the model
    predicted = model.predict(token_list, verbose=0)
    
    # Get the index of the predicted word and map it back to the word
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Streamlit App Setup
st.title("Next Word Prediction with LSTM")
input_text = st.text_input("Enter a sequence of words", "To be or not to")
if st.button("Predict Next Word"):
    # Retrieve the maximum sequence length from the model's input shape
    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    
    # Display the predicted next word
    if next_word:
        st.write(f"The next word is: **{next_word}**")
    else:
        st.write("Sorry, I couldn't predict the next word.")