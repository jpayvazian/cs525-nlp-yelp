import os
import pandas as pd

DATA_DIR = 'data'
BUSINESS = 'yelp_academic_dataset_business.json'
CHECKIN = 'yelp_academic_dataset_checkin.json'
REVIEW = 'yelp_academic_dataset_review.json'
TIP = 'yelp_academic_dataset_tip.json'
USER = 'yelp_academic_dataset_user.json'

# Loads specific file (business, checkin, review, tip, or user)
def load_data(file):
    return pd.read_json(os.path.join(DATA_DIR, file), lines=True)
