# Next Word Prediction Using LSTM

This repository contains Python scripts for developing a Long Short-Term Memory (LSTM) based model to predict the next word in a sequence. The project leverages Shakespeare's *Hamlet* as the dataset and demonstrates preprocessing, model training, and deployment in a user-friendly Streamlit web application.

### Dataset Source:

The dataset for this project is the text of Shakespeare's *Hamlet*, sourced from the NLTK Gutenberg Corpus. It is tokenized and preprocessed to create input sequences for training the LSTM model.

### Key Features:

- **Data Collection**: 
  - Extracted *Hamlet* text from NLTK's Gutenberg Corpus.
  - Saved the text into a file `hamlet.txt` for further processing.

- **Data Preprocessing**:
  - Tokenized the text and created n-gram sequences.
  - Padded the sequences for uniform input length and split the data into training and testing sets.

- **Model Development**:
  - Built an LSTM model with:
    - An embedding layer for word vector representation.
    - Two LSTM layers for sequence prediction.
    - Dropout layers for regularization.
    - A dense output layer with a softmax activation function to predict the next word.

- **Model Training**:
  - Trained the model with categorical cross-entropy loss and Adam optimizer.
  - Implemented early stopping to prevent overfitting by monitoring validation loss.

- **Next Word Prediction**:
  - Developed a function to predict the next word in a given sequence using the trained model.
  - Ensured compatibility with varying input lengths by padding sequences dynamically.

- **Model Deployment**:
  - Created a Streamlit web application where users can input a sequence of words and receive real-time predictions for the next word.

### Technologies Used:

- **Python**: Used for data preprocessing, model training, and deployment.
- **TensorFlow & Keras**: Employed for building and training the LSTM model.
- **NLTK**: Used for text data collection and tokenization.
- **Scikit-learn**: Applied for splitting the dataset into training and testing sets.
- **Streamlit**: Used to build a user-friendly web application for next-word prediction.
- **Pickle**: Saved the tokenizer for future use alongside the trained model.

### Project Workflow:

1. **Data Collection**:
   - Loaded Shakespeare's *Hamlet* using the NLTK Gutenberg Corpus.
   - Saved the text into a file named `hamlet.txt`.

2. **Data Preprocessing**:
   - Tokenized the text to create a word index.
   - Generated input sequences using n-grams and padded them for consistent length.
   - Split the sequences into predictors (`x`) and labels (`y`) and performed one-hot encoding.

3. **Model Implementation**:
   - Constructed an LSTM model with an embedding layer, two LSTM layers, and a dense output layer.
   - Trained the model with early stopping to optimize performance.

4. **Prediction Function**:
   - Built a function to accept a word sequence and return the predicted next word.
   - Utilized the trained model and tokenizer for prediction.

5. **Deployment**:
   - Created a Streamlit web app for user interaction and real-time next-word prediction.
![Next Word Prediction](https://github.com/AashishSaini16/Next-Word-Predictor-using-LSTM/blob/main/output.JPG)
