# Learning Model Evaluation and Analysis
# Collaborative Filtering (Bias-SVD) + Content-Based Filtering (TF-IDF)

import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# -----------------------------------------------------------
# Collaborative Filtering Evaluation (Bias-SVD)

print("\n===== Collaborative Filtering (Bias-SVD) Evaluation =====")

# Load ratings data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ratings_path = os.path.join(BASE_DIR, "ml-latest-small", "ratings.csv")
ratings_df = pd.read_csv(ratings_path)

# Prepare dataset for Surprise
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

# Train/Test Split
trainset, testset = train_test_split(data, test_size=0.2)

# Train SVD model
model = SVD()
model.fit(trainset)
predictions = model.test(testset)

# Evaluate RMSE / MAE
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)
accuracy_percent = (1 - (rmse / 5)) * 100

print(f"RMSE = {rmse:.4f}")
print(f"MAE  = {mae:.4f}")
print(f"Approximate Prediction Accuracy: {accuracy_percent:.2f}%")

# -----------------------------------------------------------
# Content-Based Filtering Evaluation (TF-IDF + Cosine)

print("\n===== Content-Based Filtering (TF-IDF + Cosine Similarity) =====")

cbf_path = os.path.join(BASE_DIR, "MovieLens_CBF_preprocessed.csv")
movies = pd.read_csv(cbf_path)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
movies['content'] = movies['content'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies['content'])

# Choose user for recommendation
USER_ID = 600
user_ratings = ratings_df[ratings_df['userId'] == USER_ID]
liked_movies = user_ratings[user_ratings['rating'] >= 4.0]

if liked_movies.empty:
    print(f"User {USER_ID} has no high-rated movies (rating >= 4.0)")
else:
    # Build user profile vector (weighted mean of liked movies)
    liked_indices = movies[movies['movieId'].isin(liked_movies['movieId'])].index
    user_vector = tfidf_matrix[liked_indices].T.dot(liked_movies['rating'])
    user_vector = user_vector / liked_movies['rating'].sum()

    # Compute cosine similarity between user and all movies
    cos_sim = cosine_similarity(user_vector.T.reshape(1, -1), tfidf_matrix).flatten()
    recommendations_idx = cos_sim.argsort()[::-1][:10]

    # Display results
    recommendations = movies.iloc[recommendations_idx][['title']].copy()
    recommendations['similarity'] = cos_sim[recommendations_idx]

    print(f"\n Top 10 Recommended Movies for User {USER_ID}:")
    print(recommendations.to_string(index=False))

# -----------------------------------------------------------
# Summary Output
print("\n===== Summary =====")
print("CF (Bias-SVD) achieved RMSE ≈ {:.3f}, MAE ≈ {:.3f} (~{:.1f}% accuracy)".format(rmse, mae, accuracy_percent))
print("CBF (TF-IDF) generated Top-10 content-based personalized recommendations successfully.")
