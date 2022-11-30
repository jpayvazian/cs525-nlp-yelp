from utils import *
from keras.models import Sequential
from keras import layers

# Generates the review one char at a time until review length reached
def generate_review(model, chars_map, seed_text, length):
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
            yhat = sample_pred(pred, TEMP)

        # Locate the char cooresponding to the index outputted from model
        for char, index in chars_map.items():
            if index == yhat:
                # Append the word to seed_text
                seed_text += char
                break

    return seed_text

def create_char_model(reviews):
    # Unique characters
    chars = sorted(list(set(' '.join(reviews))))
    chars_map = dict((char, chars.index(char)) for char in chars)

    # Create fixed length char sections as features
    sequences, labels = [], []
    for r in reviews:
        for i in range(0,len(r)-SEQ_LEN_CHAR,WINDOW_CHAR):
            # Label is the next char following the sequence
            sequences.append(r[i:i+SEQ_LEN_CHAR])
            labels.append(r[i+SEQ_LEN_CHAR])

    # Randomize order of training data
    sequences, labels = shuffle_data(sequences, labels)

    # Swap characters with indexes to create features as one hot encodings
    x = np.zeros((len(sequences), SEQ_LEN_CHAR, len(chars)), dtype=np.bool)
    y = np.zeros((len(sequences), len(chars)), dtype=np.bool)
    for i, sequence in enumerate(sequences):
        for t, char in enumerate(sequence):
            x[i, t, chars_map[char]] = 1
            y[i, chars_map[labels[i]]] = 1

    # Create and train LSTM Model
    model = Sequential([
        # Experiment with different layers/values
        layers.LSTM(LSTM_NEURONS, input_shape=(None, len(chars)), return_sequences=True),
        layers.LSTM(LSTM_NEURONS),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(len(chars), activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(x, y, epochs=EPOCHS, batch_size=BATCH_SIZE)

    return model, chars_map