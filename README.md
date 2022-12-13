# cs525-nlp-yelp
Final project for CS525: Natural Language Processing

Authors: Jack Ayvazian, Roopsa Ghosh, Dillon McCarthy

### Code Files
* `main.py`: Prompt user can use to choose star rating and model type to generate reviews
* Models: `word_model.py` (Word based LSTM) , `char_model.py` (character based LSTM), `gpt.py` (Finetuned GPT2)
* Evaluation: `analytics.py` contains functions for evaluation including data exploration, clustering, classifiers for real/fake and 1-5 stars, word frequency, bleu/rouge, etc.
* `utils.py`: Functions for loading and cleaning data, global constants/hyperparameters

### Motivation
How can we generate realistic reviews for restaurants given a rating (1-5 stars), text seed (to begin the review), and word/char limit?

### Models
* Word and Character based keras LSTM
* Fine tuned GPT2

### Evaluation
* Testing the accuracy of a classifier (e.g Logistic Regression, SVM) with real and generated reviews
* LSA Analysis on real/fake reviews to visualize how linearly separable the features are
* POS Tagging on real/fake to see if there is a large difference in the frequency of certain POS
