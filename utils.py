import os
import pandas as pd
import json
import re
import numpy as np
import random

DATA_DIR = 'data'
BUSINESS = 'yelp_academic_dataset_business.json'
REVIEW = 'yelp_academic_dataset_review.json'
REVIEW_SAMPLE = 'yelp_academic_dataset_review_sample.json'
REVIEW_FILES = ['one_star.json', 'two_star.json', 'three_star.json', 'four_star.json', 'five_star.json']
TEXT_FILES = ['one_star.txt', 'two_star.txt', 'three_star.txt', 'four_star.txt', 'five_star.txt']
TRAIN_SIZE = 2000 # amount of reviews for training
CATEGORY = "Food"

# HYPERPARAMETERS
EMBEDDING_DIM = 32 # Embedding vetor dimensionality
LSTM_NEURONS = 128 # LSTM Hidden layer neurons
DROPOUT_RATE = 0.2 # Prevents overfitting
SEQ_LEN_WORD = 5 # word sequence len
SEQ_LEN_CHAR = 30 # char sequence length
WINDOW_WORD = 1 # amount the sliding window moves each time for word model
WINDOW_CHAR = 5 # amount the sliding window moves each time for char model
BATCH_SIZE = 32
EPOCHS = 10
GREEDY = False # If true, take best next word each time
TEMP = 1.0 # Changes amount of variation in sampling

# Shuffles order of training data to prevent bias in model
def shuffle_data(x,y):
    data = list(zip(x, y))
    random.shuffle(data)
    return zip(*data)

# Loads json data
# lines=True for kaggle data since formatted as separate objects
def load_data(file):
    return pd.read_json(os.path.join(DATA_DIR, file))

# Creates a file for each star rating with just the review text
def create_text_files():
    for i in range(5):
        reviews = load_data(REVIEW_FILES[i])['text'].to_list()
        with open(os.path.join(DATA_DIR, TEXT_FILES[i]), 'w') as f:
            f.writelines(reviews)

# Cleans review text
def clean_text(text):
    # Get rid of punctuation and other special characters
    text = text.replace("\n", " ")
    text = re.sub(r'[^a-z0-9 ]+', '', text.lower())

    return text

# Samples prediction values for more variation
# https://keras.io/examples/generative/lstm_character_level_text_generation/
def sample_pred(pred, temp=1.0):
    preds = np.asarray(pred).astype('float64')
    preds = np.log(preds) / temp
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Samples subset of reviews from kaggle data
def sample_data():
    # Keep track of reviews per rating
    star_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    sample = []

    # Read business data
    business = pd.read_json(os.path.join(DATA_DIR, BUSINESS), lines=True)

    # Load one json review at a time
    with open(os.path.join(DATA_DIR, REVIEW), 'r', encoding='utf8') as f:
        for line in f:
            review = json.loads(line)
            # Iterate until found enough reviews for each rating
            if min(star_dict.values()) == TRAIN_SIZE:
                break

            # Skip review if already have enough for that star rating
            stars = int(review['stars'])
            if star_dict[stars] == TRAIN_SIZE:
                continue

            # Skip review if business not desired category
            categories = str(business.loc[business['business_id'] == review['business_id']]['categories'])
            if CATEGORY not in categories:
                continue

            # Add review to sample
            star_dict[stars] += 1
            sample.append(review)

            # Print progress
            print(f'\r{star_dict}', end='')

    # Write sample data to new file
    with open(os.path.join(DATA_DIR, REVIEW_SAMPLE), 'w', encoding='utf8') as f:
        f.write(json.dumps(sample))
