from utils import *
import word_model
import char_model
import gpt
STARS = ['1', '2', '3', '4', '5']
MODEL_TYPES = ['w', 'c', 'g']

if __name__ == "__main__":
    # 1. User input for star rating
    reviews, model, tokenizer = None, None, None
    file_name, run_name = '', ''
    while True:
        star = input("Enter star rating for review (1-5)")
        if star not in STARS:
            print("Invalid star rating")
            continue

        print("Loading data...")
        reviews = load_data(REVIEW_FILES[int(star)-1])['text'].to_list()
        file_name = os.path.join(DATA_DIR, TEXT_FILES[int(star) - 1])
        run_name = f"{star}star_{EPOCHS}epoch"
        break

    # 2. User input for model type
    while True:
        model_type = input("Model type: word, char, or GPT2? [w/c/g]").lower()
        if model_type not in MODEL_TYPES:
            print("Invalid model type")
            continue

        print("Training model...")
        break

    # 3. Build model
    if model_type == 'g':
        model = gpt.finetune_gpt2(file_name, run_name)

    elif model_type == 'c':
        model, tokenizer = char_model.create_char_model(reviews)

    elif model_type == 'w':
        model, tokenizer = word_model.create_word_model(reviews, star)

    # 4. Generate reviews
    while True:
        review_len = int(input("Enter review length (# words/chars):"))
        seed = input("Enter text sample to start the review:").lower()

        if model_type == 'g':
            print(gpt.generate_review(model, review_len, run_name, seed))

        elif model_type == 'c':
            print(char_model.generate_review(model, tokenizer, seed, review_len))

        elif model_type == 'w':
            print(word_model.generate_review(model, tokenizer, seed, review_len))
