import pandas as pd
from nltk.stem.snowball import SnowballStemmer
import json
import requests
from pathlib import Path
import sys
sys.path.append(Path(__file__).parent.parent.absolute())

from config import *

def weighted_rating(x: pd.DataFrame,C:float,m:float) -> float:
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

def filter_keywords(x,s):
        words = []
        for i in x:
            if i in s:
                words.append(i)
        return words

def preprocess_soup(load_movies_df):

    s = load_movies_df.apply(lambda x: pd.Series(x['keywords'],dtype='object'),axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'keyword'

    s = s.value_counts()
    s = s[s > 1]
    stemmer = SnowballStemmer('english')

    load_movies_df['keywords'] = load_movies_df['keywords'].apply(lambda x:filter_keywords(x,s))
    load_movies_df['keywords'] = load_movies_df['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
    load_movies_df['keywords'] = load_movies_df['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

    load_movies_df['cast'] = load_movies_df['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)
    load_movies_df['cast'] = load_movies_df['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
    load_movies_df['director'] = load_movies_df['director'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

    load_movies_df['soup'] = load_movies_df['keywords'] + load_movies_df['cast'] + load_movies_df['director'] + load_movies_df['genres']
    load_movies_df['soup'] = load_movies_df['soup'].apply(lambda x: ' '.join(x))

    return load_movies_df['soup']

def add_soup(movie_list:list):

    user_df = pd.DataFrame(columns=['title','id','keywords','cast','director','genres'])

    problem_movies = []

    # convert movie_ids to user_dataframe to get movie_title and preprocess user_soup
    for movie_id in movie_list:
        availability_test_url = f'{base_url}/movie/{movie_id}?api_key={tmdb_api_key}&language=en-US'
        response = requests.get(availability_test_url)
        json_ = json.loads(response.text)
        if 'status_message' in json_:
            if json_['status_message'] == "The resource you requested could not be found.":
                problem_movies.append(movie_id)
                continue


        keywords_url = f'{base_url}/movie/{movie_id}/keywords?api_key={tmdb_api_key}'
        response = requests.get(keywords_url)
        json_ = json.loads(response.text)
        keywords_list = [kw['name'] for kw in json_['keywords'] ]

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

    user_df['soup'] = preprocess_soup(user_df)

    if problem_movies:
        print("problematic movie ids",problem_movies)

    return user_df[['id','title','soup']]