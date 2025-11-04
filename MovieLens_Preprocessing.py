import pandas as pd
import matplotlib.pyplot as plt

# List all data
pd.set_option('display.max_seq_items', None)

# EDA of ratings.csv
movies_df = pd.read_csv('ml-latest-small/movies.csv')

print("=" * 50)
print("[movies_df.describe()]\n")
print(movies_df.describe())
print("=" * 50 + "\n")

print("=" * 50)
print("[movies_df.info()]\n")
print(movies_df.info())
print("=" * 50 + "\n")

# Number of null data per column
print("=" * 50)
print("[Before cleaning the dirty data]\n")
print(movies_df.isnull().sum())
print("=" * 50 + "\n")

genres_count = movies_df['genres'].str.split('|').apply(pd.Series).stack().value_counts()
genres_count.plot(kind='barh')
plt.title('Number of Movies by Genre')
plt.xlabel('Genre')
plt.ylabel('Number of Movies')
plt.tight_layout()
plt.show()

# EDA of ratings.csv
ratings_df = pd.read_csv('ml-latest-small/ratings.csv')

print("=" * 50)
print("[ratings_df.describe()]\n")
print(ratings_df.describe())
print("=" * 50 + "\n")

print("=" * 50)
print("[ratings_df.info()]\n")
print(ratings_df.info())
print("=" * 50 + "\n")

# Number of null data per column
print("=" * 50)
print("[Before cleaning the dirty data]\n")
print(ratings_df.isnull().sum())
print("=" * 50 + "\n")

ratings_df['rating'].hist()
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Number of Ratings')
plt.tight_layout()
plt.show()

# EDA of ratings.csv
tags_df = pd.read_csv('ml-latest-small/tags.csv')

print("=" * 50)
print("[tags_df.describe()]\n")
print(tags_df.describe())
print("=" * 50 + "\n")

print("=" * 50)
print("[tags_df.info()]\n")
print(tags_df.info())
print("=" * 50 + "\n")

# Number of null data per column
print("=" * 50)
print("[Before cleaning the dirty data]\n")
print(tags_df.isnull().sum())
print("=" * 50 + "\n")

tag_count = tags_df['tag'].value_counts().head(10)
tag_count.plot(kind='barh')
plt.title('Top 10 Tags')
plt.xlabel('Tag')
plt.ylabel('Number of Tags')
plt.tight_layout()
plt.show()

# Preprocessing for Collaborative Filtering (CF)
print("=" * 50)
print("Preprocessing for Collaborative Filtering")
print("=" * 50 + "\n")

# Create base DataFrame for pivoting
cf_base_df = ratings_df[['userId', 'movieId', 'rating']]

# Prepare data as a User-Item Matrix
user_item_matrix_df = cf_base_df.pivot(
    index='userId',
    columns='movieId',
    values='rating'
)

# NaN values represent movies not rated by a user, which is expected.
print("User-Item matrix created successfully")
print(user_item_matrix_df.head())

# Save the User-Item matrix
cf_matrix_filename = 'MovieLens_CF_preprocessed.csv'
user_item_matrix_df.to_csv(cf_matrix_filename)

print("=" * 50)
print("Collaborative Filtering preprocessing complete.")
print("=" * 50 + "\n")


# Preprocessing for Content-Based Filtering (CBF)
print("=" * 50)
print("Preprocessing for Content-Based Filtering")
print("=" * 50 + "\n")

# Aggregate tags for each movie
movie_tags = tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
movie_tags.rename(columns={'tag': 'tags'}, inplace=True)

# Merge tags with movies DataFrame
movies_content_df = pd.merge(movies_df, movie_tags, on='movieId', how='left')

# Handle missing tags and create combined content column
movies_content_df['tags'] = movies_content_df['tags'].fillna('')
movies_content_df['genres_cleaned'] = movies_content_df['genres'].str.replace('|', ' ', regex=False)
movies_content_df['content'] = movies_content_df['genres_cleaned'] + ' ' + movies_content_df['tags']

# Create and save the final preprocessed file for CBF
cb_preprocessed_df = movies_content_df[['movieId', 'title', 'content']]
print("Created 'content' column by combining genres and tags")
print(cb_preprocessed_df.head())

# Save the preprocessed content data
cb_output_filename = 'MovieLens_CBF_preprocessed.csv'
cb_preprocessed_df.to_csv(cb_output_filename, index=False)

print("=" * 50)
print("Content-Based preprocessing complete")
print("=" * 50 + "\n")