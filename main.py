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
MODELS_DIR = Path(BASE_DIR,"models")

import os
tmdb_api_key = os.environ.get('API_KEY')

if not tmdb_api_key:
    from config import tmdb_api_key
#import rs_models
from rs_models import contentFilter,hybridFilter,collabFilter


base_url = 'https://api.themoviedb.org/3/'


model_classes = {
    'contentFilter': contentFilter,
    'hybridFilter': hybridFilter,
    'collabFilter': collabFilter
}

class RecommendationSystem:
    def __init__(self,MODEL_NAME='contentBasedFiltering'):
        #self.load_movies_df = pd.read_pickle('movies_data.pkl')
        #self.movie_list = movie_list
        self.MODEL_NAME = MODEL_NAME

    def get_model_name(self):
        return self.MODEL_NAME

    def load_input(self,movie_list):
        ...

    def load_model(self,model_name:str):
        if model_name in model_classes:
            model = model_classes[model_name]
        ...

    def predict(self):
        ...


# test = RecommendationSystem()
movie_list = [76600, 267805, 785084, 823999, 842544, 842942, 843794, 1058949]
test = RecommendationSystem()
test.load_model(model_name='contentFilter')
#output_ = test.recommend()
# #print(output_.to_dict())
#print(output_)
