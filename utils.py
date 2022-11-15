import os
import pandas as pd
import json

DATA_DIR = 'data'
BUSINESS = 'yelp_academic_dataset_business.json'
REVIEW = 'yelp_academic_dataset_review.json'
REVIEW_SAMPLE = 'yelp_academic_dataset_review_sample.json'
SIZE = 2000
CATEGORY = "Food"

# Loads json data
# lines=True for kaggle data since formatted as separate objects
def load_data(file):
    return pd.read_json(os.path.join(DATA_DIR, file))

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
            if min(star_dict.values()) == SIZE:
                break

            # Skip review if already have enough for that star rating
            stars = int(review['stars'])
            if star_dict[stars] == SIZE:
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
