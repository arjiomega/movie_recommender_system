import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# to convert string to the right data type (csv to pd dataframe)
from ast import literal_eval

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pickle

BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = Path(BASE_DIR,"data")


class RecommendationSystem:
    def __init__(self):
        self.load_movies_df = pd.read_pickle('movies_data.pkl')
        print(self.load_movies_df)
        self.movie_list = []
    def weighted_rating(self,x,C,m):
        v = x['vote_count']
        R = x['vote_average']
        return (v/(v+m) * R) + (m/(m+v) * C)

    def add_movie(self,movie_title:str):
        print(movie_title)
        if self.load_movies_df['title'].str.contains(movie_title).any():
            self.movie_list.append(movie_title)
        else:
            raise ValueError("Movie not in available data")

    def recommend(self):

        if not self.movie_list:
            raise ValueError("Movie list is empty!")

        movie_list = self.movie_list
        load_movies_df = self.load_movies_df

        indices = pd.Series(load_movies_df.index, index=load_movies_df['title'])

        # get the indices of the titles from input list
        idx_list = [indices[title] for title in movie_list]

        count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
        count_matrix = count.fit_transform(load_movies_df['soup'])

        cosine_sim = cosine_similarity(count_matrix, count_matrix)

        sim_scores = {}
        for idx in idx_list:
            to_add = list(enumerate(cosine_sim[idx]))
            to_add = sorted(to_add, key=lambda x: x[1], reverse=True)
            to_add = to_add[1:25] # get top 10, exclude top 1 because it is usually the same bigram
            for i,score in to_add:
                if i in idx_list:
                    print(f"i {i} is in idx_list") 
                if i in sim_scores and score > sim_scores[i]:
                    sim_scores[i] = score
                if i not in sim_scores and i not in idx_list:
                    sim_scores[i] = score

        movie_indices = [key for key in sim_scores]
        movies = load_movies_df.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
        vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
        vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
        C = vote_averages.mean()
        m = vote_counts.quantile(0.60)
        qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
        qualified['vote_count'] = qualified['vote_count'].astype('int')
        qualified['vote_average'] = qualified['vote_average'].astype('int')
        qualified['wr'] = qualified.apply(self.weighted_rating, axis=1, C=C, m=m)
        qualified = qualified.sort_values('wr', ascending=False).head(10)

        return qualified

# test = RecommendationSystem()
# test.add_movie("The Dark Knight")
# output_ = test.recommend()
# print(output_.to_dict())
# with open('predict.pkl','wb') as f:
#         pickle.dump(improved_recommendations,f)

# if __name__ == '__main__':

#     movie_list = ['The Dark Knight', 'Inception', 'Interstellar']

#     # with open('predict.pkl','wb') as f:
#     #     pickle.dump(improved_recommendations,f)

#     with open('predict.pkl','rb') as f:
#         test_ = pickle.load(f)
    
#     qualified = test_(movie_list)



#     print(qualified)