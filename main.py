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

BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = Path(BASE_DIR,"data")

#from config import tmdb_api_key

import os
tmdb_api_key = os.environ['API_KEY']

if not tmdb_api_key:
    raise ValueError("no tmdb_api_key")

base_url = 'https://api.themoviedb.org/3/'

class RecommendationSystem:
    def __init__(self,movie_list):
        self.load_movies_df = pd.read_pickle('movies_data.pkl')
        self.movie_list = movie_list

    def weighted_rating(self,x: pd.DataFrame,C:float,m:float) -> float:
        """Calculates the weighted rating for a movie

        Args:
            x: each row of pandas.Dataframe
            C: Mean of Average of Votes
            m: nth percentile vote counts

        Returns:
            calculated weighted rating
        """

        v = x['vote_count']
        R = x['vote_average']

        return (v/(v+m) * R) + (m/(m+v) * C)

    def filter_keywords(self,x,s):
        words = []
        for i in x:
            if i in s:
                words.append(i)
        return words

    def preprocess_soup(self,load_movies_df):

        s = load_movies_df.apply(lambda x: pd.Series(x['keywords'],dtype='object'),axis=1).stack().reset_index(level=1, drop=True)
        s.name = 'keyword'

        s = s.value_counts()
        s = s[s > 1]
        stemmer = SnowballStemmer('english')

        load_movies_df['keywords'] = load_movies_df['keywords'].apply(lambda x:self.filter_keywords(x,s))
        load_movies_df['keywords'] = load_movies_df['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
        load_movies_df['keywords'] = load_movies_df['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

        load_movies_df['cast'] = load_movies_df['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)
        load_movies_df['cast'] = load_movies_df['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
        load_movies_df['director'] = load_movies_df['director'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

        load_movies_df['soup'] = load_movies_df['keywords'] + load_movies_df['cast'] + load_movies_df['director'] + load_movies_df['genres']
        load_movies_df['soup'] = load_movies_df['soup'].apply(lambda x: ' '.join(x))

        return load_movies_df['soup']


    def add_soup(self,movie_list:list):

        user_df = pd.DataFrame(columns=['title','id','keywords','cast','director','genres'])

        # convert movie_ids to user_dataframe to get movie_title and preprocess user_soup
        for movie_id in movie_list:
            keywords_url = f'{base_url}/movie/{movie_id}/keywords?api_key={tmdb_api_key}'
            response = requests.get(keywords_url)
            json_ = json.loads(response.text)
            print(json_)
            keywords_list = [kw['name'] for kw in json_['keywords']]

            casts_crews_url = f'{base_url}/movie/{movie_id}/credits?api_key={tmdb_api_key}&language=en-US'
            response = requests.get(casts_crews_url)
            json_ = json.loads(response.text)
            casts_list = [cast['name'] for cast in json_['cast']]
            director_list = [crew['name'] for crew in json_['crew'] if crew['job'] == 'Director']

            genre_url = f'{base_url}/movie/{movie_id}?api_key={tmdb_api_key}&language=en-US'
            response = requests.get(genre_url)
            json_ = json.loads(response.text)
            genre_list = [genre['name'] for genre in json_['genres']]
            title = json_['title']

            user_df.loc[-1] = title,movie_id,keywords_list,casts_list,director_list,genre_list
            user_df.index = user_df.index + 1
            user_df = user_df.sort_index()

        user_df['soup'] = self.preprocess_soup(user_df)

        return user_df[['id','title','soup']]


    def recommend(self):
        load_movies_df = self.load_movies_df
        # Input movie list
        movie_list = self.movie_list

        #

        # Get User Soup (Add to Overall Soup)
        user_df = self.add_soup(movie_list)
        load_movies_df = pd.concat([load_movies_df,user_df], axis=0)
        load_movies_df = load_movies_df.drop_duplicates(subset='title',keep="first")
        load_movies_df = load_movies_df.reset_index(drop=True)

        indices = pd.Series(load_movies_df.index, index=load_movies_df['title'])

        # get the indices of the titles from input list
        movie_title = [load_movies_df.loc[load_movies_df['id']==index, 'title'].iloc[0] for index in movie_list]
        idx_list = [indices[title] for title in movie_title]

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
        qualified['wr'] = qualified.apply(self.weighted_rating, axis=1, C=C, m=m)
        qualified = qualified.sort_values('wr', ascending=False).head(10)

        return qualified


# test = RecommendationSystem()
# movie_list = [76600, 267805, 315162, 436270, 505642, 536554, 587092, 631842, 640146, 646389, 653851, 758009, 785084, 823999, 842544, 842942, 843794, 1058949]
# test = RecommendationSystem(movie_list)
# output_ = test.recommend()
# # #print(output_.to_dict())
# print(output_)
