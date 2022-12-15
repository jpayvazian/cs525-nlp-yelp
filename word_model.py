from utils import *
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras import layers
from keras.utils.np_utils import to_categorical
import tensorflow as tf

# Generates the review one word at a time until review length reached
def generate_review(model, tokenizer, seed_text, review_len):
    for _ in range(review_len):
        # Convert last SEQ_LEN-1 words of seed text into tokenized value
        seed_text_input = [' '.join(seed_text.split()[-SEQ_LEN_WORD:])]
        seed_sequence = np.array(tokenizer.texts_to_sequences(seed_text_input)[0])
        seed_sequence = np.expand_dims(seed_sequence, axis=0)

        # Predict the next word based on previous words
        # If greedy approach, choose highest val
        # Otherwise, sample the next word for more variation
        pred = model.predict(seed_sequence)[0]
        if GREEDY:
            yhat = np.argmax(pred)
        else:
            yhat = sample_pred(pred, TEMP)

        # Locate the word cooresponding to the index outputted from model
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                # Append the word to seed_text
                seed_text += " " + word
                break

    return seed_text

def create_word_model(reviews, star):
    tf.compat.v1.experimental.output_all_intermediates(True)
    # Initialize tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(reviews)

    # Vocabulary size
    vocab_size = len(tokenizer.word_index) + 1

    # Create subsequences of words (N-gram) as features
    x,y = [], []
    for r in reviews:
        # Tokenize text into sequence of integers
        sequence = tokenizer.texts_to_sequences([r])[0]

        # Sequences will overlap in SEQ_LEN-WINDOW words
        for i in range(0,len(sequence)-SEQ_LEN_WORD,WINDOW_WORD):
            # Label is the next word following the sequence
            x.append(sequence[i:i+SEQ_LEN_WORD])
            y.append(sequence[i+SEQ_LEN_WORD])

    # Randomize order of training data
    x, y = shuffle_data(x, y)

    # Create labels from last word/char of sequence
    y = to_categorical(np.array(y), num_classes=vocab_size)

    # # Create callback to save training checkpoint
    # checkpoint_path = f'checkpoint/{star}_word_model.ckpt'
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    # Create and train LSTM Model
    model = Sequential([
        layers.Embedding(vocab_size, EMBEDDING_DIM, input_length=SEQ_LEN_WORD),
        # Experiment with different layers/values
        layers.LSTM(LSTM_NEURONS, return_sequences=True),
        layers.LSTM(LSTM_NEURONS),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(vocab_size, activation ='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    model.load_weights('./checkpoint/word_model/weights.h5')

    model.fit(np.array(x), y, epochs=EPOCHS, batch_size=BATCH_SIZE)

    model.save_weights('./checkpoint/word_model/weights.h5')

    return model, tokenizer

if __name__ == "__main__":
    star = int(input("Enter star rating for review (1-5)"))

    # Names for data text file and model save dir
    file_name = os.path.join(DATA_DIR, TEXT_FILES[star - 1])
    run_name = f"{star}star_word_model_{EPOCHS}epoch"
    data_file_name = os.path.join(DATA_DIR, f'{star}star_word_model.json')

    # Load reviews
    reviews = load_data(REVIEW_FILES[star-1])['text'].to_list()[:GEN_SIZE]

    # Finetune/load model
    model, tokenizer = create_word_model(reviews, star)

    # Generate reviews with same length and prefix as real ones
    fake_reviews = []
    for i in range(GEN_SIZE):
        # Split review into list of words
        review_words = reviews[i].split()
        seed = ' '.join(review_words[:PREFIX_SIZE])
        fake_reviews.append(generate_review(model, tokenizer, seed, len(review_words)))
        print(f'\r{i+1}/{GEN_SIZE}', end= '')

    with open(data_file_name, 'w', encoding='utf8') as f:
        f.write(json.dumps(fake_reviews))

