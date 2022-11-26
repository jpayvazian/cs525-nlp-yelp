import numpy as np
import utils
from keras.models import Sequential
from keras import layers

REVIEW_SAMPLE = 'yelp_academic_dataset_review_sample.json'
REVIEW_FILES = ['one_star.json', 'two_star.json', 'three_star.json', 'four_star.json', 'five_star.json']

# HYPERPARAMETERS
LSTM_NEURONS = 128
DROPOUT_RATE = 0.2 # Prevents overfitting
SEQ_LEN = 30 # char sequence len
WINDOW = 5 # amount the sliding window moves each time
BATCH_SIZE = 32
EPOCHS = 5
GREEDY = False
TEMP = 1.0

# Generates the review one char at a time until review length reached
def generate_review(seed_text, model, chars_map, length):
    for _ in range(length):
        # Convert seed text into one hot encoding
        x = np.zeros((1, len(seed_text), len(chars_map)))
        for t, char in enumerate(seed_text):
            x[0, t, chars_map[char]] = 1

        # Predict the next char based on previous chars
        # If greedy approach, choose highest val
        # Otherwise, sample the next char for more variation
        pred = model.predict(x)[0]
        if GREEDY:
            yhat = np.argmax(pred)
        else:
            yhat = utils.sample_pred(pred, TEMP)

        # Locate the char cooresponding to the index outputted from model
        for char, index in chars_map.items():
            if index == yhat:
                # Append the word to seed_text
                seed_text += char
                break

    return seed_text

def create_char_model(reviews):
    # Unique characters
    chars = sorted(list(set(' '.join(reviews['text']))))
    chars_map = dict((char, chars.index(char)) for char in chars)

    # Create RNN Model
    model = Sequential([
        # Experiment with different layers/values
        layers.LSTM(LSTM_NEURONS, input_shape=(None,len(chars)), return_sequences=True),
        layers.LSTM(LSTM_NEURONS),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(len(chars), activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Create fixed length char sections as features
    sequences, labels = [], []
    for r in reviews['text']:
        for i in range(0,len(r)-SEQ_LEN,WINDOW):
            # Label is the next char following the sequence
            sequences.append(r[i:i+SEQ_LEN])
            labels.append(r[i+SEQ_LEN])

    # Swap characters with indexes to create features as one hot encodings
    x = np.zeros((len(sequences), SEQ_LEN, len(chars)), dtype=np.bool)
    y = np.zeros((len(sequences), len(chars)), dtype=np.bool)
    for i, sequence in enumerate(sequences):
        for t, char in enumerate(sequence):
            x[i, t, chars_map[char]] = 1
            y[i, chars_map[labels[i]]] = 1

    # Train model
    model.fit(x, y, epochs=EPOCHS, batch_size=BATCH_SIZE)

    return model, chars_map