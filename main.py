import utils
import word_model
import char_model
REVIEW_FILES = ['one_star.json', 'two_star.json', 'three_star.json', 'four_star.json', 'five_star.json']
STARS = ['1', '2', '3', '4', '5']
MODEL_TYPES = ['w', 'c']

if __name__ == "__main__":
    # 1. User input for star rating
    while True:
        star = input("Enter star rating for review (1-5)")
        if star not in STARS:
            print("Invalid star rating")
            continue

        print("Loading data...")
        reviews = utils.load_data(REVIEW_FILES[int(star)-1])
        break

    # 2. User input for model type
    while True:
        model_type = input("Model type: word or char? [w/c]").lower()
        if model_type not in MODEL_TYPES:
            print("Invalid model type")
            continue

        print("Training model...")
        break

    # 3. Build word model
    if model_type == 'w':
        model, tokenizer = word_model.create_word_model(reviews)
        # 4. Generate reviews
        while True:
            try:
                review_len = int(input("Enter review length (# words):"))
                seed = input("Enter text sample to start the review:").lower()

                print(word_model.generate_review(seed, model, tokenizer, review_len))
            # Type a non-number for review len to quit
            except:
                break

    # 3. Build char model
    elif model_type == 'c':
        model, chars_map = char_model.create_char_model(reviews)
        # 4. Generate reviews
        while True:
            try:
                review_len = int(input("Enter review length (# characters):"))
                seed = input("Enter text sample to start the review:").lower()

                print(char_model.generate_review(seed, model, chars_map, review_len))
            # Type a non-number for review len to quit
            except:
                break
