from utils import *
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from transformers import TFAutoModel, AutoTokenizer, pipeline
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

def data_exploration():
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

# LSA Analysis on 5 star reviews
def LSA(star_num):
    # Load data for specified star rating
    real = load_data(REVIEW_FILES[star_num])['text'].to_list()[:GEN_SIZE]
    fake = load_json(FAKE_REVIEW_FILES[star_num])

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
    plt.title(f't-SNE Clustering of LSA: Real vs Generated YELP Reviews {star_num+1} stars')
    plt.savefig(os.path.join(PLOT_DIR, f'LSA_{star_num}.png'))

# Analyzes separability of real and fake data using a binary classifier
def simple_machine_evaluation(star_num, show_stats=True):
    # Load data for specified star rating
    real = load_data(REVIEW_FILES[star_num])['text'].to_list()[:GEN_SIZE]
    fake = load_json(FAKE_REVIEW_FILES[star_num])

    # Add labels and split data
    real = pd.DataFrame(real, columns=['text'])
    fake = pd.DataFrame(fake, columns=['text'])
    real['label'] = 1
    fake['label'] = 0
    both = pd.concat([fake, real])

    # Use Count Vectorizer to turn text in numerical data
    cv = CountVectorizer()
    vec = cv.fit_transform(both['text'])

    train_x, test_x, train_y, test_y = train_test_split(vec, both['label'], test_size=.3)
    
    # Define and train model on combined data
    model_real = RandomForestClassifier(max_depth=10)
    model_real.fit(train_x, train_y)

    acc = model_real.score(test_x, test_y)

    if show_stats:
        # Evaluate classifier
        pred = model_real.predict(test_x)
        print(f'Separability of {star_num+1}-star data')
        print(confusion_matrix(test_y, pred))
        print('Accuracy\t' + str(acc))

    return acc

# Analyzes distribution of classes for real and fake data.  A classifier trained on real data to classify
# a star rating 1-5 should (ideally) perform equally well on real and fake data, and vice versa
def class_machine_evaluation():
    # Load data for specified star rating
    all_real_reviews = []
    all_fake_reviews = []
    fake_labels = []
    real_labels = []

    for i in range(len(REVIEW_FILES)):
        real = load_data(REVIEW_FILES[star_num])['text'].to_list()[:GEN_SIZE]
        fake = load_json(FAKE_REVIEW_FILES[star_num])
        all_real_reviews += real
        all_fake_reviews += fake
        fake_labels += [i+1] * len(fake)
        real_labels += [i+1] * len(real)

    # Combine in order for embeddings to be same dimensionality
    combined_data = all_real_reviews + all_fake_reviews

    # Use Count Vectorizer to turn text in numerical data
    cv = TfidfVectorizer()
    vec = cv.fit_transform(combined_data)
    real_vec = vec[:len(all_real_reviews)]
    fake_vec = vec[len(all_real_reviews):]

    real_train_x, real_test_x, real_train_y, real_test_y = train_test_split(real_vec, real_labels, test_size=.3)
    fake_train_x, fake_test_x, fake_train_y, fake_test_y = train_test_split(fake_vec, fake_labels, test_size=.3)

    # Define and train model on real data
    model_real = RandomForestClassifier(max_depth=10)
    model_real.fit(real_train_x, real_train_y)

    # Define and train model on fake data
    model_fake = RandomForestClassifier(max_depth=10)
    model_fake.fit(fake_train_x, fake_train_y)

    # Evaluate classifier trained on real data
    print('Evaluating classifier trained on real data')
    # Real Data Evaluatation
    pred = model_real.predict(real_test_x)
    print('Real Data Evaluation')
    print(confusion_matrix(real_test_y, pred))
    print('Accuracy\t' + str(model_real.score(real_test_x, real_test_y)))

    # Fake Data Evaluatation
    pred = model_real.predict(fake_test_x)
    print('Fake Data Evaluation')
    print(confusion_matrix(fake_test_y, pred))
    print('Accuracy\t' + str(model_real.score(fake_test_x, fake_test_y)))

    # Evaluate classifier trained on fake data
    print('Evaluating classifier trained on fake data')
    # Real Data Evaluatation
    pred = model_fake.predict(real_test_x)
    print('Real Data Evaluation')
    print(confusion_matrix(real_test_y, pred))
    print('Accuracy\t' + str(model_fake.score(real_test_x, real_test_y)))

    # Fake Data Evaluatation
    pred = model_fake.predict(fake_test_x)
    print('Fake Data Evaluation')
    print(confusion_matrix(fake_test_y, pred))
    print('Accuracy\t' + str(model_fake.score(fake_test_x, fake_test_y)))

def gpt_detector():
    # Load data for specified star rating
    reviews = []
    labels = []

    for i in range(len(REVIEW_FILES)):
        real = load_data(REVIEW_FILES[star_num])['text'].to_list()[:GEN_SIZE]
        fake = load_json(FAKE_REVIEW_FILES[star_num])
        reviews += [r[:512] for r in real]
        reviews += [f[:512] for f in fake]
        labels += ["LABEL_1"] * len(real)
        labels += ["LABEL_0"] * len(fake)

    classifier = pipeline(model="roberta-base-openai-detector")
    results = classifier(reviews)
    labels = np.array(labels)
    pred = np.array([y.get('label') for y in results])

    print(f'Acc: {np.mean(labels == pred)}')

def word_frequencies(star):
    # Load data for specified star rating
    real = load_data(REVIEW_FILES[star])['text'].to_list()[:GEN_SIZE]
    fake = load_json(FAKE_REVIEW_FILES[star])

    # Add labels and split data
    real = pd.DataFrame(real, columns=['text'])
    fake = pd.DataFrame(fake, columns=['text'])

    # Tokenize each review and add to list
    real_tokens, fake_tokens = [], []
    for real_review, fake_review in zip(real['text'], fake['text']):
        real_tokens += word_tokenize(real_review)
        fake_tokens += word_tokenize(fake_review)

    # Get frequency distribution object
    real_dist = FreqDist(real_tokens)
    fake_dist = FreqDist(fake_tokens)

    # Convert object to dataframe and save to csv
    real_words, real_freq = zip(*real_dist.most_common())
    fake_words, fake_freq = zip(*fake_dist.most_common())
    real_df = pd.DataFrame(zip(real_words, real_freq), columns=['word', 'frequency'])
    fake_df = pd.DataFrame(zip(fake_words, fake_freq), columns=['word', 'frequency'])
    real_df.to_csv(f'./plots/real_{star}star_frequencies.csv')
    fake_df.to_csv(f'./plots/fake_{star}star_frequencies.csv')

if __name__ == "__main__":

    star_num = 0        # The star rating 1-5 indexed at 0-4
    # data_exploration()
    # LSA(0)
    # simple_machine_evaluation(star_num)
    # class_machine_evaluation()
    # gpt_detector()
    # word_frequencies(star_num)

    # Run simple machine eval multiple times and save to csv for averaging classifier accuracy
    # reps = 20
    # s = []
    # for star in range(5):
    #     s_star = []
    #     for i in range(reps):
    #         acc = simple_machine_evaluation(star, show_stats=False)
    #         s_star.append(acc)
    #     s.append(s_star)

    # df = pd.DataFrame(zip(s[0], s[1], s[2], s[3], s[4]), columns=['1star', '2star', '3star', '4star', '5star'])
    # df.to_csv('./results/separability.csv')