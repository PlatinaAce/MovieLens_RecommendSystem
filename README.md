# MovieLens_RecommendSystem
This project implements, evaluates, and compares various movie recommendation algorithms using the MovieLens (ml-latest-small) dataset. It covers both major recommendation approaches: Collaborative Filtering (CF) and Content-Based Filtering (CBF).

## Key Features
* Data Preprocessing & EDA: Analyzes and visualizes the characteristics of the ratings, movies, and tags data.
* Content-Based Filtering (CBF): Implements a recommender using TF-IDF vectorization on movie genres and tags, generating user profiles and calculating Cosine Similarity to recommend items.
* Collaborative Filtering (CF): Implements and compares 6 major CF algorithms using the surprise and implicit libraries.
* Model Evaluation & Hyperparameter Tuning: Performs automated tuning using GridSearchCV and detailed performance evaluation using RMSE and MAE metrics.
* Result Visualization: Generates matplotlib charts to visually compare the performance (RMSE, MAE, execution time) of each CF model.

## Implemented Algorithms
This project implements and evaluates the following algorithms:

### Content-Based Filtering (CBF)
* TF-IDF (Term Frequency-Inverse Document Frequency)
* Cosine Similarity

### Collaborative Filtering (CF)
* Item-Item kNN
* User-User kNN
* Slope-One
* Bias-SVD (Biased Singular Value Decomposition)
* NMF (Non-negative Matrix Factorization)
* ALS (Alternating Least Squares)

## Dataset
* Dataset: MovieLens Latest Small (ml-latest-small)
* Contents: Comprises 100,836 ratings and 3,683 tag applications across 9,742 movies by 610 users.
* Source Files: movies.csv, ratings.csv, tags.csv
