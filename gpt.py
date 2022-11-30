from utils import *
import gpt_2_simple as gpt2

# Generates a review
def generate_review(session, seed_text, review_len):
    gpt2.generate(session, length=review_len, prefix=seed_text)

# Finetunes GPT2 model
def finetune_gpt2(file_name):
    # Download 124M param (small) GPT2 model if not already there
    if not os.path.isdir(os.path.join("models", "124M")):
        print(f"Downloading GPT2 124M model...")
        gpt2.download_gpt2(model_name="124M")

    # Start finetuning session
    session = gpt2.start_tf_sess()
    gpt2.finetune(session, file_name, model_name="124M", steps=EPOCHS)

    return session