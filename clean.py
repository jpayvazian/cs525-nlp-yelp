import os
import utils
import string
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras import layers
from keras.utils.np_utils import to_categorical

REVIEW_SAMPLE = 'yelp_academic_dataset_review_sample.json'
REVIEW_FILES = ['one_star.json', 'two_star.json', 'three_star.json', 'four_star.json', 'five_star.json']

# HYPERPARAMETERS
EMBEDDING_DIM = 32 # Embedding vetor dimensionality
LSTM_NEURONS = 128
DROPOUT_RATE = 0.1 # Prevents overfitting
SEQ_LEN = 6 # N-gram sequence length
BATCH_SIZE = 32
EPOCHS = 5

# Cleans review text
def clean_text(text):
    # Get rid of punctuation
    text = text.replace("\n", " ")
    text = "".join(w for w in text if w not in string.punctuation).lower()

    return text

# Creates RNN/LSTM Model
def create_model(input_dim):
    model = Sequential([
        layers.Embedding(input_dim, EMBEDDING_DIM, input_length=SEQ_LEN-1),
        # Experiment with different layers/values
        layers.LSTM(LSTM_NEURONS, return_sequences=True),
        layers.LSTM(LSTM_NEURONS),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(input_dim, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

# Generates the review one word at a time until review length reached
def generate_review(seed_text, model, tokenizer, review_len):
    for _ in range(review_len):
        # Convert last SEQ_LEN-1 words of seed text into tokenized value
        seed_text_input = [' '.join(seed_text.split()[-(SEQ_LEN-1):])]
        seed_sequence = np.array(tokenizer.texts_to_sequences(seed_text_input)[0])
        seed_sequence = np.expand_dims(seed_sequence, axis=0)

        # Predict the next word based on previous words
        yhat = np.argmax(model.predict(seed_sequence), axis=1)

        # Locate the word cooresponding to the index outputted from model
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                # Append the word to seed_text
                seed_text += " " + word
                break

    return seed_text

if __name__ == "__main__":
    # Clean and split reviews
    # Create separate models for each set of star reviews
    for file in REVIEW_FILES:
        reviews = utils.load_data(file)

        # Initialize tokenizer
        rnn_tokenizer = Tokenizer()
        rnn_tokenizer.fit_on_texts(reviews['text'])
        # Vocabulary size
        vocab_size = len(rnn_tokenizer.word_index) + 1

        # Create subsequences of words (N-gram) as features
        features = []
        for r in reviews['text']:
            # Tokenize text into sequence of integers
            sequence = rnn_tokenizer.texts_to_sequences([r])[0]
            # Sequences will overlap in all but 1 word
            for i in range(0,len(sequence)-(SEQ_LEN-1)):
                features.append(sequence[i:i+SEQ_LEN])

        # Create RNN Model
        rnn_model = create_model(vocab_size)

        # Create labels from last word of sequence
        features = np.array(features)
        x, y = features[:,:-1], features[:,-1]
        y = to_categorical(y, num_classes=vocab_size)

        # Train model
        rnn_model.fit(x, y, epochs=EPOCHS, batch_size=BATCH_SIZE)

        # Input seed text of length SEQ_LEN
        print(generate_review("If I could give it", rnn_model, rnn_tokenizer, 20))

        break # Just trying 1 model for now

        # results: If I could give it
        # If I could give it back to the table and the food was not good and the food was not good and the food was