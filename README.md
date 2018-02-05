# Matrix Factorization Recommender System
This was a part of my homework assignment for UOZP at FRI, University of Ljubljana. The idea was to take artist ratings that users gad given on last.fm, and use that data to recommend new artists to users.

## Fatorization
I implemented the iterative stochastic matrix factorization algorithm to handle the actual factorization.

## Bias
Rating bias is accounted for by appending a row and column of 1s to get the average.

## Additional data
By computing the average distance between ratings users give I found ratings friends give to the same artist are tightly correlated. I used the average friend score for the same artist when predicting a rating to improve Recommendation accuracy.
