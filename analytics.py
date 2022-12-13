from utils import *
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from transformers import TFAutoModel, AutoTokenizer, pipeline
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
import itertools
from sklearn.cluster import KMeans

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
    both = pd.DataFrame()
    # Combine all reviews for real and fake
    for k in range(5):
        real = load_data(REVIEW_FILES[k])['text'].to_list()[:GEN_SIZE]
        fake = load_json(FAKE_REVIEW_FILES[k])

        # Add labels and merge data
        real = pd.DataFrame(real, columns=['text'])
        fake = pd.DataFrame(fake, columns=['text'])
        real['label'] = 1
        fake['label'] = 0
        both = pd.concat([both, fake, real])

    # Use Tf-idf Vectorizer to turn text in numerical data
    tf = TfidfVectorizer()
    review_matrix = tf.fit_transform(both['text'])

    # LSA/LDA/Kmeans
    # lsa = TruncatedSVD(n_components=5)
    # lda = LatentDirichletAllocation(n_components=5)
    kmeans = KMeans(n_clusters=2)
    kmeans_matrix = kmeans.fit_transform(review_matrix)

    # t-SNE Clustering of predicted (real or fake)
    tsne = TSNE(n_components=2, perplexity=50)
    tsne_vectors = tsne.fit_transform(kmeans_matrix)

    # Visualize
    plt.scatter(tsne_vectors[:, 0], tsne_vectors[:, 1], c=both['label'])
    plt.title(f't-SNE KMeans Clustering: Real vs Generated YELP Reviews')
    plt.savefig(os.path.join(PLOT_DIR, f'kmeans.png'))

# Gets the accuracy per class given true and predicted labels.  To be used with machine evaluation
# Returns a list of accuracies for each class
def get_class_stats(labels, pred):
    totals = [0, 0, 0, 0, 0]
    accuracies = [0, 0, 0, 0, 0]
    for guess, label in zip(pred, labels):
        if guess == label:
            accuracies[label-1] += 1
        totals[label-1] += 1
    accuracies = np.divide(np.array(accuracies), np.array(totals))

    return accuracies

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
    # Real Data Evaluatation
    pred = model_real.predict(real_test_x)
    rr_acc = get_class_stats(real_test_y, pred)

    # Fake Data Evaluatation
    pred = model_real.predict(fake_test_x)
    rf_acc = get_class_stats(fake_test_y, pred)

    # Evaluate classifier trained on fake data
    # Real Data Evaluatation
    pred = model_fake.predict(real_test_x)
    fr_acc = get_class_stats(real_test_y, pred)

    # Fake Data Evaluatation
    pred = model_fake.predict(fake_test_x)
    ff_acc = get_class_stats(fake_test_y, pred)

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
    classifier = pipeline(model="roberta-base-openai-detector")
    for s in range(len(REVIEW_FILES)):
        reviews = []
        labels = []
        real = load_data(REVIEW_FILES[s])['text'].to_list()[:GEN_SIZE]
        fake = load_json(FAKE_REVIEW_FILES[s])
        reviews += [r[:512] for r in real]
        reviews += [f[:512] for f in fake]
        labels += ["LABEL_1"] * len(real)
        labels += ["LABEL_0"] * len(fake)

        results = classifier(reviews)
        pred = np.array([y.get('label') for y in results])

        print(f'Acc star {s+1}: {np.mean(np.array(labels) == pred)}')

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

def bleu_score(star):
    # Load data for specified star rating
    real = load_data(REVIEW_FILES[star])['text'].to_list()[:GEN_SIZE]
    fake = load_json(FAKE_REVIEW_FILES[star])

    ref_bleu = []
    gen_bleu = []
    for l in fake:
        gen_bleu.append(l.split())
    for i,l in enumerate(real):
        ref_bleu.append([l.split()])

    cc = SmoothingFunction()
    score_bleu = corpus_bleu(ref_bleu, gen_bleu, weights=(0, 1, 0, 0), smoothing_function=cc.method4)
    print(f'BLEU:\t{score_bleu}')

    return score_bleu

#rouge scores for a reference/generated sentence pair
#source google seq2seq source code.
#supporting function
def _split_into_words(sentences):
    """Splits multiple sentences into words and flattens the result"""
    return list(itertools.chain(*[_.split(" ") for _ in sentences]))

#supporting function
def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    words = _split_into_words(sentences)
    return _get_ngrams(n, words)

#supporting function
def _get_ngrams(n, text):
    """Calcualtes n-grams.
    Args:
    n: which n-grams to calculate
    text: An array of tokens
    Returns:
    A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set

def rouge_n(star, n=2):
    """
    Computes ROUGE-N of two text collections of sentences.
    Source: http://research.microsoft.com/en-us/um/people/cyl/download/
    papers/rouge-working-note-v1.3.1.pdf
    Args:
    evaluated_sentences: The sentences that have been picked by the summarizer
    reference_sentences: The sentences from the referene set
    n: Size of ngram.  Defaults to 2.
    Returns:
    recall rouge score(float)
    Raises:
    ValueError: raises exception if a param has len <= 0
    """
    # Load data for specified star rating
    reference_sentences = load_data(REVIEW_FILES[star])['text'].to_list()[:GEN_SIZE]
    evaluated_sentences = load_json(FAKE_REVIEW_FILES[star])

    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")

    evaluated_ngrams = _get_word_ngrams(n, evaluated_sentences)
    reference_ngrams = _get_word_ngrams(n, reference_sentences)
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    # Gets the overlapping ngrams between evaluated and reference
    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    # Handle edge case. This isn't mathematically correct, but it's good enough
    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    print(f'ROUGE:\t{recall}')

    return recall

if __name__ == "__main__":

    star_num = 0        # The star rating 1-5 indexed at 0-4
    # data_exploration()
    # LSA(0)
    # simple_machine_evaluation(star_num)
    # class_machine_evaluation()
    # sentiment_machine_evaluation()
    # gpt_detector()
    # word_frequencies(star_num)
    bleu_score(star_num)
    rouge_n(star_num)

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
    # df.to_csv('./results/multiclass_class_stats.csv')