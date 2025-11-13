import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 

# load dataset
movies = pd.read_csv('MovieLens_CBF_preprocessed.csv')
user_ratings = pd.read_csv('ratings.csv')

movies = movies[['movieId', 'title', 'content']]
user_ratings = user_ratings[['userId', 'movieId', 'rating']]

# use TF-IDF Vectorizer to convert contents into vectors
tfidf = TfidfVectorizer(stop_words='english')
movies['content'] = movies['content'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies['content'])


# craete user profile
# generate a weighted average user profile vector based on movies highly rated by the user
def createUserProfile(user_id, user_ratings, movies, tfidf_matrix):
    
    user_ratings = user_ratings[user_ratings['userId'] == user_id]
    high_rated_movies = user_ratings[user_ratings['rating'] >= 4.0]
    movie_indexes = movies[movies['movieId'].isin(high_rated_movies['movieId'])].index
    
    if len(movie_indexes) == 0:
        return None
    
    # create a weighted average user profile (using ratings as weights)
    # apply weights by merging the rating data with the TF-IDF vectors
    rated_movies_with_vector_info = movies.iloc[movie_indexes].copy()
    rated_movies_with_vector_info = pd.merge(rated_movies_with_vector_info, high_rated_movies, on='movieId')
    
    user_movie_vectors = tfidf_matrix[movie_indexes]
    
    # calculate user profile
    # the sum of (rating of each movie * its corresponding TF-IDF vector) / the sum of ratings.
    user_profile = (user_movie_vectors.T.dot(rated_movies_with_vector_info['rating']) / rated_movies_with_vector_info['rating'].sum()).T
    
    # the result is a sparse matrix or ndarray
    if hasattr(user_profile, 'toarray'):
        # if sparse matrix, convert ndarray (for error handling)
        return user_profile.toarray().flatten()
    else:
        return user_profile.flatten()


# recommend function
def contentBasedRecommend(user_id, user_profile, movies, tfidf_matrix, num_recommendations=10):
    
    # no high rated movie Error
    if user_profile is None:
        return pd.DataFrame([["N/A", f"No movies that were enjoyed"]], columns=['Title', 'Similarity Score'])
    
    # Reshape the user profile to a 1xN matrix
    user_profile_matrix = user_profile.reshape(1, -1)
    
    # calcurate Cosin Similarity
    cos_sim = cosine_similarity(user_profile_matrix, tfidf_matrix).flatten()
    
    # get the similarity
    # use lambda to sort by sim score (score is second column(= x[1]))
    sim_scores = list(enumerate(cos_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Retrieve the user list and exclude movies that have already been watched (이미 본 영화 제거)
    user_rated_movies = user_ratings[user_ratings['userId'] == user_id]['movieId'].unique()
    
    recommendations = []
    
    # get the top N movies
    for i, score in sim_scores:
        movie_id = movies.iloc[i]['movieId']
        if movie_id not in user_rated_movies:
            recommendations.append((movies.iloc[i]['title'], score))
        
        if len(recommendations) >= num_recommendations:
            break
            
    # convert result to df
    recommendations_df = pd.DataFrame(recommendations, columns=['Title', 'Similarity Score'])
    return recommendations_df



### example test code
USER_ID_TO_RECOMMEND = 600
NUM_RECOMMENDATIONS = 10

user_profile = createUserProfile(USER_ID_TO_RECOMMEND, user_ratings, movies, tfidf_matrix)

recommendations = contentBasedRecommend(USER_ID_TO_RECOMMEND, user_profile, movies, tfidf_matrix, NUM_RECOMMENDATIONS)

print(f'\nTop {NUM_RECOMMENDATIONS} recommended movies for user "{USER_ID_TO_RECOMMEND}":\n')
print(recommendations)



# ### tf-idf weighted (for debugging)
#
# if user_profile is not None:
#     feature_names = tfidf.get_feature_names_out()
#     top_features_indexes = np.argsort(user_profile)[::-1][:10]
#     top_features = [(feature_names[i], user_profile[i]) for i in top_features_indexes]
#     print(pd.DataFrame(top_features, columns=['Feature', 'Weight']))