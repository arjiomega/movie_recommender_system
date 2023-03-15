import numpy as np
import pandas as pd
from pathlib import Path

# to convert string to the right data type (csv to pd dataframe)
from ast import literal_eval

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pickle
import requests
import json

from ast import literal_eval
from nltk.stem.snowball import SnowballStemmer

from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = Path(BASE_DIR,"data")


reader = Reader()
ratings = pd.read_csv(Path(DATA_DIR,'ratings.csv'),nrows=500000)
print(ratings)
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

svd = SVD()
cv = cross_validate(svd, data, measures=['RMSE', 'MAE'],cv=5)
print(cv)
trainset = data.build_full_trainset()
svd.fit(trainset)

ratings[ratings['userId'] == 1]

# user_id, tmdb_id
predict = svd.predict(1, 302, 3)

#print(predict)

user_id = 500

# Get a list of all movie IDs in the dataset
all_movie_ids = ratings['movieId'].unique()

print(all_movie_ids)

# Create a list of tuples, where each tuple contains a movie ID and the estimated rating for that movie by the user
movie_ratings = [(movie_id, svd.predict(user_id, movie_id).est) for movie_id in all_movie_ids]

# Sort the list of movie ratings by descending order of estimated rating
movie_ratings.sort(key=lambda x: x[1], reverse=True)

# Get the top 10 movie IDs with the highest estimated ratings
top_movie_ids = [movie_rating[0] for movie_rating in movie_ratings[:10]]

# Print the top 10 movie IDs
def CollabFilter():
    print("Top 10 movie IDs for user", user_id, ":", top_movie_ids)

CollabFilter()