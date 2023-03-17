import numpy as np
import pandas as pd
from pathlib import Path

# to convert string to the right data type (csv to pd dataframe)
from ast import literal_eval

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import sys
from .preprocess import weighted_rating
sys.path.append(Path(__file__).parent.parent.absolute())
from config import *

class run:
    def __init__(self,user_id,movie_list,user_df):
        self.user_id = user_id
        self.movie_list = movie_list
        self.user_df = user_df

    def predict(self):
        movie_list = self.movie_list
        user_df = self.user_df

        load_movies_df = pd.read_pickle('movies_data.pkl')
        load_movies_df = pd.concat([load_movies_df,user_df], axis=0)

        # drop title-id pairs
        load_movies_df = load_movies_df.drop_duplicates(subset=['title','id'],keep="first")
        # drop duplicate ids (duplicate title names is possible so it will not get dropped)
        load_movies_df = load_movies_df.drop_duplicates(subset=['id'],keep="first")
        load_movies_df = load_movies_df.reset_index(drop=True)

        # use id instead of title to prevent issues with movie duplicate titles
        indices = pd.Series(load_movies_df.index, index=load_movies_df['id'])

        # get the indices of the titles from input list
        movie_title = [load_movies_df.loc[load_movies_df['id']==index, 'title'].iloc[0] for index in movie_list]
        idx_list = [indices[tmdb_id] for tmdb_id in movie_list]

        count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
        count_matrix = count.fit_transform(load_movies_df['soup'])
        cosine_sim = cosine_similarity(count_matrix, count_matrix)

        sim_scores = {}
        for idx in idx_list:
            to_add = list(enumerate(cosine_sim[idx]))
            to_add = sorted(to_add, key=lambda x: x[1], reverse=True)
            to_add = to_add[1:25] # get top 10, exclude top 1 because it is usually the same bigram
            for i,score in to_add:
                if i in sim_scores and score > sim_scores[i]:
                    sim_scores[i] = score
                if i not in sim_scores and i not in idx_list:
                    sim_scores[i] = score

        movie_indices = [key for key in sim_scores]
        movies = load_movies_df.iloc[movie_indices][['title','id', 'vote_count', 'vote_average', 'year']]

        vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
        vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
        C = vote_averages.mean()
        m = vote_counts.quantile(0.60)

        qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())].copy()
        qualified['vote_count'] = qualified['vote_count'].astype('int')
        qualified['vote_average'] = qualified['vote_average'].astype('int')
        qualified['wr'] = qualified.apply(weighted_rating, axis=1, C=C, m=m)
        qualified = qualified.sort_values('wr', ascending=False).head(10)

        recommend = [{"title":title,"id":tmdb_id}  for title,tmdb_id in zip(qualified['title'].tolist(),qualified['id'].tolist())]

        return {'recommended_movies': recommend}