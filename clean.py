import os
import utils
import string
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras import layers

DATA_DIR = 'data'
REVIEW_SAMPLE = 'yelp_academic_dataset_review_sample.json'
REVIEW_FILES = ['one_star.json', 'two_star.json', 'three_star.json', 'four_star.json', 'five_star.json']

# MODEL ARCHITECTURE
EMBEDDING_DIM = 32 # Embedding vetor dimensionality, increase?
LSTM_NEURONS = 128
DROPOUT_RATE = 0.1 # Prevents overfitting

# N-GRAM SLIDING WINDOW
SEQ_LEN = 5

# Cleans review text
def clean_text(text):
    # To preserve commas and periods, replace them and revert on ouput
    text = text.replace("\n", " ")
    text = text.replace(".", " periodtoken")
    text = text.replace(",", " commatoken")
    # Get rid of other punctuation
    text = "".join(w for w in text if w not in string.punctuation).lower()

    return text

# Creates RNN/LSTM Model
def create_model(input_dim):
    model = Sequential([
        layers.Embedding(input_dim, EMBEDDING_DIM, input_length=SEQ_LEN),
        # TODO: Experiment with different layers/values
        layers.LSTM(LSTM_NEURONS, return_sequences=True),
        layers.LSTM(LSTM_NEURONS),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(input_dim, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

if __name__ == "__main__":
    # Clean reviews
    # reviews = utils.load_data(REVIEW_SAMPLE)
    # reviews['text'] = reviews['text'].apply(clean_text)

    # Create separate models for each set of star reviews
    for file in REVIEW_FILES:
        reviews = utils.load_data(file)

        # Tokenize text in sequence of integers
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(reviews['text'])
        sequences = tokenizer.texts_to_sequences(reviews['text'])
        # Vocabulary size
        vocab_size = len(tokenizer.word_index) + 1

        # Create subsequences of words (N-gram) as features
        features = []
        for s in sequences:
            # Sequences will overlap in all but 1 word
            for i in range(0,len(s)-1):
                features.append(s[i:i+SEQ_LEN])

        # Create RNN Model
        rnn_model = create_model(vocab_size)
