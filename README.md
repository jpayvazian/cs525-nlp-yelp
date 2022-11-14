# cs525-nlp-yelp
Final project for CS525: Natural Language Processing

Authors: Jack Ayvazian, Roopsa Ghosh, Dillon McCarthy

1. Proposed question
How can we generate realistic reviews for restaurants given a rating (1-5 stars), text seed (to begin the review), and word limit?

2. Proposed method/algorithm/model (in general)
LSTM/RNN, GANs, Markov Chains

3. Evaluation method
Testing a classifier (real/fake and trained on real, test on fake), analyzing cosine similarities

4. Any preliminary findings?
Loading and cleaning data
- Yelp data set is very large (> 4 mil)
- Load subset of data into a new file for use
