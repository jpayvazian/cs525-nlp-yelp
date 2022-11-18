import os
import utils
from keras.preprocessing.text import Tokenizer
import numpy as np
import matplotlib.pyplot as plt

PLOT_DIR = 'plots'
REVIEW_FILES = ['one_star.json', 'two_star.json', 'three_star.json', 'four_star.json', 'five_star.json']

if __name__ == "__main__":
    # Metrics for data exploration
    avg_review_len = []
    vocab_size = []

    for file in REVIEW_FILES:
        reviews = utils.load_data(file)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(reviews['text'])

        # Calculate vocabulary size and average review length
        avg_review_len.append(np.mean([len(w.split()) for w in reviews['text']]))
        vocab_size.append(len(tokenizer.word_index) + 1)

    # Make graphs with data
    plt.bar([1,2,3,4,5], vocab_size)
    plt.title('Yelp reviews: Star rating vs. Vocabulary size')
    plt.xlabel("Stars")
    plt.ylabel("Vocabulary size")
    plt.savefig(os.path.join(PLOT_DIR, 'vocab_size.png'))
    plt.clf()
    plt.bar([1,2,3,4,5], avg_review_len)
    plt.title('Yelp reviews: Average length of review')
    plt.xlabel("Stars")
    plt.ylabel("Average word count of review")
    plt.savefig(os.path.join(PLOT_DIR, 'review_len.png'))