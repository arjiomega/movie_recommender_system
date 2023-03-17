from pathlib import Path

BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = Path(BASE_DIR,"data")
MODELS_DIR = Path(BASE_DIR,"models")

import os
tmdb_api_key = os.environ.get('API_KEY')

from config import *
from rs_models import preprocess,contentFilter,collabFilter#, hybridFilter


base_url = 'https://api.themoviedb.org/3/'


model_classes = {
    'contentFilter': contentFilter,
    'hybridFilter': None, #hybridFilter,
    'collabFilter': collabFilter
}

class RecommendationSystem:
    def __init__(self,user_id):
        #self.load_movies_df = pd.read_pickle('movies_data.pkl')
        #self.movie_list = movie_list
        self.user_id = user_id
        ...

    def load_input(self,movie_list:list[int]):
        if hasattr(self,'model_name'):
            self.user_df = preprocess.add_soup(movie_list)
            self.movie_list = movie_list
        else:
            print('choose a model first using load_model method')

    def load_model(self,model_name:str):
        if model_name in model_classes:
            self.instance = model_classes[model_name]
            self.model_name = model_name
            return self.instance
        else:
            print(f"model {model_name} is not available. Choose one below.")
            print('\n'.join( list( model_classes.keys() ) ))

    def predict(self):
        if hasattr(self,'instance'):
            # for contentFilter
            predict_model = self.instance.run(user_id=self.user_id,movie_list=self.movie_list,user_df=self.user_df)

            ###############################
            #for collabFilter
            # import random
            # user_rating = [{'id':tmdb_id,'rating':rating} for tmdb_id,rating in zip(self.movie_list,[random.randint(0, 5) for _ in range(len(self.movie_list))])]
            #predict_model = self.instance.run(user_id=self.user_id,user_rating=user_rating,user_df=self.user_df)
            ###############################

            recommend = predict_model.predict()
            return recommend
        else:
            print('choose a model first using load_model method')


