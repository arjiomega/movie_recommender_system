import pandas as pd
from pathlib import Path
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

import sys
sys.path.append(Path(__file__).parent.parent.absolute())
from config import *

reader = Reader(rating_scale=(0.0, 5.0))

class run:
    def __init__(self,user_id:int,user_rating:list[dict[int,int]],user_df):
        self.user_id = user_id
        self.user_df = user_df
        self.user_df = self.user_df.merge(pd.DataFrame(user_rating),on='id')
        self.ratings = pd.read_csv(Path(DATA_DIR,'ratings_small.csv'),usecols=['userId','movieId','rating'])
        id_map = pd.read_csv(Path(DATA_DIR,'links_small.csv'))[['movieId', 'tmdbId']]

        # remove null
        id_map = id_map.dropna()

        id_map['tmdbId'] = id_map['tmdbId'].astype(int)
        id_map.columns = ['movieId', 'id']

        # inner join (exclude those that do not have matching movieId)
        self.ratings = self.ratings.merge(id_map, on='movieId')
        self.ratings = self.ratings.drop('movieId',axis=1)

        del id_map

    def fit(self):
        data = Dataset.load_from_df(self.ratings[['userId', 'id', 'rating']], reader)
        self.svd = SVD()
        before_fit = cross_validate(self.svd, data, measures=['RMSE', 'MAE'],cv=5)
        trainset = data.build_full_trainset()
        self.svd.fit(trainset)
        after_fit = cross_validate(self.svd, data, measures=['RMSE', 'MAE'],cv=5)

    def predict(self):
        self.fit()
        #self.user_df['est'] = self.user_df['id'].apply(lambda x: self.svd.predict(self.user_id,x).est)
        load = pd.read_pickle('movies_data.pkl')

        load['est'] = load['id'].apply(lambda x: self.svd.predict(self.user_id,x).est)

        load = load.sort_values(by='est',ascending=False)
        print('predict\n')
        print(load[['id','title','est']])

        print('\nactual\n')

        self.ratings = self.ratings.merge(load[['title','id']], on='id')

        print(self.ratings[self.ratings['userId']==1])

        #recommend = [{"title":title,"id":tmdb_id}  for title,tmdb_id in zip(self.user_df['title'].tolist(),self.user_df['id'].tolist())]

        #return {'recommended_movies': recommend}



# obj = run(user_id=500)
# obj.fit()


# # Get a list of all movie IDs in the dataset
# all_movie_ids = ratings['movieId'].unique()

# print(all_movie_ids)

# # Create a list of tuples, where each tuple contains a movie ID and the estimated rating for that movie by the user
# movie_ratings = [(movie_id, svd.predict(user_id, movie_id).est) for movie_id in all_movie_ids]

# # Sort the list of movie ratings by descending order of estimated rating
# movie_ratings.sort(key=lambda x: x[1], reverse=True)

# # Get the top 10 movie IDs with the highest estimated ratings
# top_movie_ids = [movie_rating[0] for movie_rating in movie_ratings[:10]]
# print(top_movie_ids)
# # Print the top 10 movie IDs
# #def CollabFilter():
#     #print("Top 10 movie IDs for user", user_id, ":", top_movie_ids)

# #CollabFilter()