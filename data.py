import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = Path(BASE_DIR,"data")

from ast import literal_eval
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_director(x):
        for i in x:
            if i['job'] == 'Director':
                return i['name']
        return np.nan

def filter_keywords(x,s):
        words = []
        for i in x:
            if i in s:
                words.append(i)
        return words

def elt():
    # LOAD
    movies_df = pd.read_csv(Path(DATA_DIR,"movies_metadata.csv"))
    links_df = pd.read_csv(Path(DATA_DIR,'links_small.csv'))
    credits = pd.read_csv(Path(DATA_DIR,'credits.csv'))
    keywords = pd.read_csv(Path(DATA_DIR,'keywords.csv'))

    # Transform
    movies_df['genres'] = movies_df['genres'].apply(literal_eval).apply(lambda genres: [genre['name'] for genre in genres] if isinstance(genres,list) else [] )
    movies_df['year'] = pd.to_datetime(movies_df['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
    movies_df = movies_df.drop([19730, 29503, 35587])

    links_df = links_df[links_df['tmdbId'].notnull()]['tmdbId'].astype('int')

    keywords['id'] = keywords['id'].astype('int')
    credits['id'] = credits['id'].astype('int')
    movies_df['id'] = movies_df['id'].astype('int')
    movies_df = movies_df.merge(credits, on='id')
    movies_df = movies_df.merge(keywords, on='id')
    load_movies_df = movies_df[movies_df['id'].isin(links_df)]
    load_movies_df['cast'] = load_movies_df['cast'].apply(literal_eval)
    load_movies_df['crew'] = load_movies_df['crew'].apply(literal_eval)
    load_movies_df['keywords'] = load_movies_df['keywords'].apply(literal_eval)
    load_movies_df['cast_size'] = load_movies_df['cast'].apply(lambda x: len(x))
    load_movies_df['crew_size'] = load_movies_df['crew'].apply(lambda x: len(x))


    load_movies_df['director'] = load_movies_df['crew'].apply(get_director)

    load_movies_df['cast'] = load_movies_df['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    load_movies_df['cast'] = load_movies_df['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)

    load_movies_df['keywords'] = load_movies_df['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    load_movies_df['cast'] = load_movies_df['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
    load_movies_df['director'] = load_movies_df['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
    load_movies_df['director'] = load_movies_df['director'].apply(lambda x: [x,x, x])

    s = load_movies_df.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'keyword'

    s = s.value_counts()
    s = s[s > 1]


    stemmer = SnowballStemmer('english')

    load_movies_df['keywords'] = load_movies_df['keywords'].apply(lambda x:filter_keywords(x,s))
    load_movies_df['keywords'] = load_movies_df['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
    load_movies_df['keywords'] = load_movies_df['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

    load_movies_df['soup'] = load_movies_df['keywords'] + load_movies_df['cast'] + load_movies_df['director'] + load_movies_df['genres']
    load_movies_df['soup'] = load_movies_df['soup'].apply(lambda x: ' '.join(x))

    load_movies_df = load_movies_df.reset_index()


    load_movies_df.to_pickle(Path(DATA_DIR,'movies_data.pkl'))



if __name__ == '__main__':
    elt()