import os
import pandas as pd

DATA_DIR = 'data'
BUSINESS = 'yelp_academic_dataset_business.json'
REVIEW = 'yelp_academic_dataset_review.json'
REVIEW_SAMPLE = 'yelp_academic_dataset_review_sample.json'

# Loads json data
# lines=True for kaggle data since formatted as separate objects
def load_data(file):
    return pd.read_json(os.path.join(DATA_DIR, file))
