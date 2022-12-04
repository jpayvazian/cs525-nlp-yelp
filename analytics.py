from utils import *
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# LSA Analysis on 5 star reviews
def LSA():
    real = load_data(REVIEW_FILES[4])['text'].to_list()[:GEN_SIZE]
    fake = load_json(FAKE_REVIEW_FILES[4])

    # Add labels and merge data
    real = pd.DataFrame(real, columns=['text'])
    fake = pd.DataFrame(fake, columns=['text'])
    real['label'] = 1
    fake['label'] = 0
    both = pd.concat([fake, real])

    # Use Count Vectorizer to turn text in numerical data
    cv = CountVectorizer()
    review_matrix = cv.fit_transform(both['text'])

    # Latent Semantic Analysis
    lsa = TruncatedSVD(n_components=2)
    lsa_matrix = lsa.fit_transform(review_matrix)

    # t-SNE Clustering of predicted (real or fake)
    tsne = TSNE(n_components=2, perplexity=50)
    tsne_vectors = tsne.fit_transform(lsa_matrix)

    # Visualize
    plt.scatter(tsne_vectors[:, 0], tsne_vectors[:, 1], c=both['label'])
    plt.title('t-SNE Clustering of LSA: Real vs Generated YELP Reviews')
    plt.savefig(os.path.join(PLOT_DIR, 'LSA.png'))

if __name__ == "__main__":
    # Metrics for data exploration
    avg_review_len = []
    vocab_size = []

    for file in REVIEW_FILES:
        reviews = load_data(file)
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