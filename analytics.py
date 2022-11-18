import os
import utils
from keras.preprocessing.text import Tokenizer
import numpy as np
import matplotlib.pyplot as plt

PLOT_DIR = 'plots'
if __name__ == "__main__":
    # Separate reviews by star rating
    reviews_1star = utils.load_data('one_star.json')
    reviews_2star = utils.load_data('two_star.json')
    reviews_3star = utils.load_data('three_star.json')
    reviews_4star = utils.load_data('four_star.json')
    reviews_5star = utils.load_data('five_star.json')

    # Tokenize reviews
    tokenizer = Tokenizer()

    avg_review_len = []
    vocab_size = []
    # Calculate vocabulary size and truncate/pad reviews
    tokenizer.fit_on_texts(reviews_1star['text'])
    avg_review_len.append(np.mean([len(s) for s in reviews_1star['sequences']]))
    vocab_size.append(len(tokenizer.word_index) + 1)

    tokenizer.fit_on_texts(reviews_2star['text'])
    avg_review_len.append(np.mean([len(s) for s in reviews_2star['sequences']]))
    vocab_size.append(len(tokenizer.word_index) + 1)

    tokenizer.fit_on_texts(reviews_3star['text'])
    avg_review_len.append(np.mean([len(s) for s in reviews_3star['sequences']]))
    vocab_size.append(len(tokenizer.word_index) + 1)

    tokenizer.fit_on_texts(reviews_4star['text'])
    avg_review_len.append(np.mean([len(s) for s in reviews_4star['sequences']]))
    vocab_size.append(len(tokenizer.word_index) + 1)

    tokenizer.fit_on_texts(reviews_5star['text'])
    avg_review_len.append(np.mean([len(s) for s in reviews_5star['sequences']]))
    vocab_size.append(len(tokenizer.word_index) + 1)

    # Make graphs with data
    plt.bar([1,2,3,4,5], vocab_size)
    plt.title('Yelp reviews: Star rating vs. Vocabulary size')
    plt.xlabel("Stars")
    plt.ylabel("Vocabulary size")
    plt.savefig(os.path.join(PLOT_DIR, 'vocab_size.png'))

    plt.bar([1,2,3,4,5], avg_review_len)
    plt.title('Yelp reviews: Average length of review')
    plt.xlabel("Stars")
    plt.ylabel("Average word count of review")
    plt.savefig(os.path.join(PLOT_DIR, 'review_len.png'))