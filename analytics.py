from collections import Counter
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
import nltk
from transformers import TFAutoModel, AutoTokenizer, pipeline
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import itertools

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
        real = load_data(REVIEW_FILES[i])['text'].to_list()[:GEN_SIZE]
        fake = load_json(FAKE_REVIEW_FILES[i])
        all_real_reviews += real
        all_fake_reviews += fake
        fake_labels += [i+1] * len(fake)
        real_labels += [i+1] * len(real)

    # Combine in order for embeddings to be same dimensionality
    combined_data = all_real_reviews + all_fake_reviews

    # Use TF-IDF Vectorizer to turn text in numerical data
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
    rr_acc = model_real.score(real_test_x, real_test_y)
    print('Accuracy\t' + str(rr_acc))

    # Fake Data Evaluatation
    pred = model_real.predict(fake_test_x)
    print('Fake Data Evaluation')
    print(confusion_matrix(fake_test_y, pred))
    rf_acc = model_real.score(fake_test_x, fake_test_y)
    print('Accuracy\t' + str(rf_acc))

    # Evaluate classifier trained on fake data
    print('Evaluating classifier trained on fake data')
    # Real Data Evaluatation
    pred = model_fake.predict(real_test_x)
    print('Real Data Evaluation')
    print(confusion_matrix(real_test_y, pred))
    fr_acc = model_fake.score(real_test_x, real_test_y)
    print('Accuracy\t' + str(fr_acc))

    # Fake Data Evaluatation
    pred = model_fake.predict(fake_test_x)
    print('Fake Data Evaluation')
    print(confusion_matrix(fake_test_y, pred))
    ff_acc = model_fake.score(fake_test_x, fake_test_y)
    print('Accuracy\t' + str(ff_acc))

    return rr_acc, rf_acc, fr_acc, ff_acc

def sentiment_machine_evaluation():
    # Load data for specified star rating
    all_real_reviews = []
    all_fake_reviews = []
    fake_labels = []
    real_labels = []

    for i in range(len(REVIEW_FILES)):
        real = load_data(REVIEW_FILES[i])['text'].to_list()[:GEN_SIZE]
        fake = load_json(FAKE_REVIEW_FILES[i])
        all_real_reviews += real
        all_fake_reviews += fake
        fake_labels += [i+1] * len(fake)
        real_labels += [i+1] * len(real)

    # Combine in order for embeddings to be same dimensionality
    combined_data = all_real_reviews + all_fake_reviews
    combined_labels = real_labels + fake_labels

    data = pd.DataFrame(zip(combined_data, combined_labels), columns=['reviews', 'star_rating'])

    data['label'] = data['star_rating'].apply(lambda x: 'negative' if x < 4 else 'positive')

    # Use TF-IDF Vectorizer to turn text in numerical data
    cv = TfidfVectorizer()
    vec = cv.fit_transform(combined_data)
    real_vec = vec[:len(all_real_reviews)]
    fake_vec = vec[len(all_real_reviews):]
    real_labels = data['label'][:len(all_real_reviews)]
    fake_labels = data['label'][len(all_fake_reviews):]

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
    rr_acc = model_real.score(real_test_x, real_test_y)
    print('Accuracy\t' + str(rr_acc))

    # Fake Data Evaluatation
    pred = model_real.predict(fake_test_x)
    print('Fake Data Evaluation')
    print(confusion_matrix(fake_test_y, pred))
    rf_acc = model_real.score(fake_test_x, fake_test_y)
    print('Accuracy\t' + str(rf_acc))

    # Evaluate classifier trained on fake data
    print('Evaluating classifier trained on fake data')
    # Real Data Evaluatation
    pred = model_fake.predict(real_test_x)
    print('Real Data Evaluation')
    print(confusion_matrix(real_test_y, pred))
    fr_acc = model_fake.score(real_test_x, real_test_y)
    print('Accuracy\t' + str(fr_acc))

    # Fake Data Evaluatation
    pred = model_fake.predict(fake_test_x)
    print('Fake Data Evaluation')
    print(confusion_matrix(fake_test_y, pred))
    ff_acc = model_fake.score(fake_test_x, fake_test_y)
    print('Accuracy\t' + str(ff_acc))

    return rr_acc, rf_acc, fr_acc, ff_acc

def gpt_detector():
    # Load data for specified star rating
    reviews = []
    labels = []

    for s in range(len(REVIEW_FILES)):
        real = load_data(REVIEW_FILES[s])['text'].to_list()[:GEN_SIZE]
        fake = load_json(FAKE_REVIEW_FILES[s])
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

def pos_tagging_analytics(star_num):

    # Load data for specified star rating
    real = load_data(REVIEW_FILES[star_num])['text'].to_list()[:GEN_SIZE]
    fake = load_json(FAKE_REVIEW_FILES[star_num])

    # Add labels and split data
    real = pd.DataFrame(real, columns=['text'])
    fake = pd.DataFrame(fake, columns=['text'])
    real['label'] = 1
    fake['label'] = 0
    both = pd.concat([fake, real])

    real['pos-tagging'] = real['text'].apply(lambda x: pos_tag(x))
    fake['pos-tagging'] = fake['text'].apply(lambda x: pos_tag(x))


    dict_pos_tagging = Counter()

    for pos_dict in fake['pos-tagging']:
        dict_pos_tagging = dict_pos_tagging + pos_dict
        for key, value in dict_pos_tagging.items():
            if key in dict_pos_tagging and key in pos_dict:
               dict_pos_tagging[key] = value + pos_dict[key]       

   
    new_dict = dict(itertools.islice(dict_pos_tagging.items(),10))

    data = (new_dict)
    names = list(data.keys())
    values = list(data.values())

    plt.bar(range(len(data)), values, tick_label=names)
    plt.title('Top 12 POS tags in fake reviews (' + star_num + ' star reviews)')
    plt.show()    




def pos_tag(text):

    tokens = word_tokenize(text)
    tags = nltk.pos_tag(tokens)

    #find the number of each tags
    counts = Counter( tag for word, tag in tags)

    return counts

if __name__ == "__main__":

    star_num = 0        # The star rating 1-5 indexed at 0-4
    # data_exploration()
    # LSA(0)
    # simple_machine_evaluation(star_num)
    # class_machine_evaluation()
    # sentiment_machine_evaluation()
    # gpt_detector()
    # word_frequencies(star_num)


    # pos analysis
    pos_tagging_analytics(0)
    pos_tagging_analytics(1)
    pos_tagging_analytics(2)
    pos_tagging_analytics(3)
    pos_tagging_analytics(4)


    # Run machine eval multiple times and save to csv for averaging classifier accuracy
    # reps = 20
    # ac1, ac2, ac3, ac4 = [], [], [], []
    # for i in range(reps):
    #     a1, a2, a3, a4 = class_machine_evaluation()
    #     ac1.append(a1)
    #     ac2.append(a2)
    #     ac3.append(a3)
    #     ac4.append(a4)

    # df = pd.DataFrame(zip(ac1, ac2, ac3, ac4), columns=['real_trained_on_real', 'fake_trained_on_real', 'real_trained_on_fake', 'fake_trained_on_fake'])
    # df.to_csv('./results/multiclass.csv')