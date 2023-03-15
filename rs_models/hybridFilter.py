import numpy as np
import pandas as pd

from pathlib import Path
import sys

BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = Path(BASE_DIR,"data")

sys.path.append(BASE_DIR)





def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan

load_movies_df =  pd.read_pickle('movies_data.pkl')

id_map = pd.read_csv(Path(DATA_DIR,'links_small.csv'))[['movieId', 'tmdbId']]
id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
id_map.columns = ['movieId', 'id']
id_map = id_map.merge(load_movies_df[['title', 'id']], on='id').set_index('title')

# smd = dataframe from soup

indices_map = id_map.set_index('id')

from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

DATA_DIR = Path(BASE_DIR,"data")


reader = Reader()
ratings = pd.read_csv(Path(DATA_DIR,'ratings.csv'))

# data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# svd = SVD()
# cv = cross_validate(svd, data, measures=['RMSE', 'MAE'],cv=5)
# print(cv)
# trainset = data.build_full_trainset()
# svd.fit(trainset)

# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
# count_matrix = count.fit_transform(load_movies_df['soup'])
# cosine_sim = cosine_similarity(count_matrix, count_matrix)

# indices = pd.Series(load_movies_df.index, index=load_movies_df['title'])

# def hybrid(userId, title):
#     idx = indices[title]
#     tmdbId = id_map.loc[title]['id']
#     movie_id = id_map.loc[title]['movieId']

#     sim_scores = list(enumerate(cosine_sim[int(idx)]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     sim_scores = sim_scores[1:101]#26
#     movie_indices = [i[0] for i in sim_scores]

#     movies = load_movies_df.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
#     movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
#     print(movies['est'])
#     movies = movies.sort_values('est', ascending=False)
#     print(movies)
#     return movies.head(10)

# print(hybrid(1, 'Avatar'))