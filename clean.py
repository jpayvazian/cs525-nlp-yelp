import os
import utils
import string
from keras.preprocessing.text import Tokenizer

DATA_DIR = 'data'
REVIEW_SAMPLE = 'yelp_academic_dataset_review_sample.json'
REVIEW_TOKENS = 'yelp_academic_dataset_review_tokens.json'

# Cleans review text
def clean_text(text):
    # To preserve commas and periods, replace them and revert on ouput
    text = text.replace("\n", " ")
    text = text.replace(".", " periodtoken")
    text = text.replace(",", " commatoken")
    # Get rid of other punctuation
    text = "".join(w for w in text if w not in string.punctuation).lower()

    return text

if __name__ == "__main__":
    # Clean reviews
    reviews = utils.load_data(REVIEW_SAMPLE)
    reviews['text'] = reviews['text'].apply(clean_text)

    # Tokenize reviews
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(reviews['text'])
    reviews['sequences'] = tokenizer.texts_to_sequences(reviews['text'])
    reviews.to_json(os.path.join(DATA_DIR, REVIEW_TOKENS))