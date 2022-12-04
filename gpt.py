from utils import *
import gpt_2_simple as gpt2

# Generates a review
def generate_review(session, review_len, run_name, seed_text=None):
    return gpt2.generate(session, length=review_len, prefix=seed_text, return_as_list=True, run_name=run_name)[0]

# Finetunes GPT2 model
def finetune_gpt2(file_name, run_name):
    # Download 124M param (small) GPT2 model if not already there
    if not os.path.isdir(os.path.join("models", "124M")):
        print(f"Downloading GPT2 124M model...")
        gpt2.download_gpt2(model_name="124M")

    # Start finetuning session
    session = gpt2.start_tf_sess()

    # Finetune model if not already there, otherwise load checkpoint
    if not os.path.isdir(os.path.join("checkpoint", run_name)):
        gpt2.finetune(session, file_name, model_name="124M", steps=EPOCHS, run_name=run_name)
    else:
        gpt2.load_gpt2(session, run_name=run_name)

    return session

if __name__ == "__main__":
    # Generate a bunch of reviews for evaluation
    star = int(input("Enter star rating for review (1-5)"))

    # Names for data text file and model save dir
    file_name = os.path.join(DATA_DIR, TEXT_FILES[star - 1])
    run_name = f"{star}star_{EPOCHS}epoch"

    # Finetune/load model
    model = finetune_gpt2(file_name, run_name)

    # Get average lengths of reviews
    reviews = load_data(REVIEW_FILES[star-1])['text'].to_list()[:GEN_SIZE]
    avg_len = int(np.mean([len(x) for x in reviews]))

    fake_reviews = []
    # Generate reviews with same length and prefix as real ones
    for i in range(GEN_SIZE):
        fake_reviews.append(generate_review(model, avg_len, run_name, ' '.join(reviews[i].split()[:PREFIX_SIZE])))
        print(f'\r{i+1}/{GEN_SIZE}', end= '')

    with open(os.path.join(DATA_DIR, FAKE_REVIEW_FILES[star-1]), 'w', encoding='utf8') as f:
        f.write(json.dumps(fake_reviews))
