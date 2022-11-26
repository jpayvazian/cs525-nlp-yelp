import os
import utils
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
DROPOUT_RATE = 0.2 # Prevents overfitting
SEQ_LEN = 5 # N-gram sequence length
WINDOW = 1 # amount the sliding window moves each time
BATCH_SIZE = 32
EPOCHS = 5
GREEDY = False
TEMP = 1.0

# Generates the review one word at a time until review length reached
def generate_review(seed_text, model, tokenizer, review_len):
    for _ in range(review_len):
        # Convert last SEQ_LEN-1 words of seed text into tokenized value
        seed_text_input = [' '.join(seed_text.split()[-SEQ_LEN:])]
        seed_sequence = np.array(tokenizer.texts_to_sequences(seed_text_input)[0])
        seed_sequence = np.expand_dims(seed_sequence, axis=0)

        # Predict the next word based on previous words
        # If greedy approach, choose highest val
        # Otherwise, sample the next word for more variation
        pred = model.predict(seed_sequence)[0]
        if GREEDY:
            yhat = np.argmax(pred)
        else:
            yhat = utils.sample_pred(pred, TEMP)

        # Locate the word cooresponding to the index outputted from model
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                # Append the word to seed_text
                seed_text += " " + word
                break

    return seed_text

def create_word_model(reviews):
    # Initialize tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(reviews['text'])

    # Vocabulary size
    vocab_size = len(tokenizer.word_index) + 1

    # Create RNN Model
    model = Sequential([
        layers.Embedding(vocab_size, EMBEDDING_DIM, input_length=SEQ_LEN),
        # Experiment with different layers/values
        layers.LSTM(LSTM_NEURONS, return_sequences=True),
        layers.LSTM(LSTM_NEURONS),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(vocab_size, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Create subsequences of words (N-gram) as features
    x,y = [], []
    for r in reviews['text']:
        # Tokenize text into sequence of integers
        sequence = tokenizer.texts_to_sequences([r])[0]

        # Sequences will overlap in SEQ_LEN-WINDOW words
        for i in range(0,len(sequence)-SEQ_LEN,WINDOW):
            # Label is the next word following the sequence
            x.append(sequence[i:i+SEQ_LEN])
            y.append(sequence[i+SEQ_LEN])

    # Create labels from last word/char of sequence
    x = np.array(x)
    y = np.array(y)
    y = to_categorical(y, num_classes=vocab_size)

    # Train model
    model.fit(x, y, epochs=EPOCHS, batch_size=BATCH_SIZE)

    return model, tokenizer
