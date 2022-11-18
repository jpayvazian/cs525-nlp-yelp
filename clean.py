import os
import utils
import string
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras import layers

DATA_DIR = 'data'
REVIEW_SAMPLE = 'yelp_academic_dataset_review_sample.json'

# HYPERPARAMETERS
EMBEDDING_DIM = 32 # Embedding vetor dimensionality, increase?
MAX_LEN = 150 # Pad/Truncate to this size, otherwise the max review may be too much padding on average

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
    return Sequential([
        layers.Embedding(input_dim, EMBEDDING_DIM, input_length=MAX_LEN),
        # TODO: Finish model and experiment with different layers/values
        layers.LSTM(),
        layers.LSTM(),
        layers.Dropout(),
        layers.Dense()
    ])

if __name__ == "__main__":
    # Clean reviews
    # reviews = utils.load_data(REVIEW_SAMPLE)
    # reviews['text'] = reviews['text'].apply(clean_text)

    # Separate reviews by star rating
    reviews_1star = utils.load_data('one_star.json')
    reviews_2star = utils.load_data('two_star.json')
    reviews_3star = utils.load_data('three_star.json')
    reviews_4star = utils.load_data('four_star.json')
    reviews_5star = utils.load_data('five_star.json')

    # Tokenize reviews
    tokenizer = Tokenizer()

    # Calculate vocabulary size
    tokenizer.fit_on_texts(reviews_1star['text'])
    # reviews_1star['sequences'] = tokenizer.texts_to_sequences(reviews_1star['text'])
    vocab_size_1star = len(tokenizer.word_index) + 1

    tokenizer.fit_on_texts(reviews_2star['text'])
    # reviews_2star['sequences'] = tokenizer.texts_to_sequences(reviews_2star['text'])
    vocab_size_2star = len(tokenizer.word_index) + 1

    tokenizer.fit_on_texts(reviews_3star['text'])
    # reviews_3star['sequences'] = tokenizer.texts_to_sequences(reviews_3star['text'])
    vocab_size_3star = len(tokenizer.word_index) + 1

    tokenizer.fit_on_texts(reviews_4star['text'])
    # reviews_4star['sequences'] = tokenizer.texts_to_sequences(reviews_4star['text'])
    vocab_size_4star = len(tokenizer.word_index) + 1

    tokenizer.fit_on_texts(reviews_5star['text'])
    # reviews_5star['sequences'] = tokenizer.texts_to_sequences(reviews_5star['text'])
    vocab_size_5star = len(tokenizer.word_index) + 1

    # TODO: Pad/truncate reviews to MAX_LEN

    # Create RNN Models for each star category
    model_1star = create_model(vocab_size_1star)
    model_2star = create_model(vocab_size_2star)
    model_3star = create_model(vocab_size_3star)
    model_4star = create_model(vocab_size_4star)
    model_5star = create_model(vocab_size_5star)