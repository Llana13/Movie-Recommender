import pandas as pd
import numpy as np
import pickle

movies_ = pd.DataFrame(pd.read_csv('/home/Llana13/movie_recommender/static/movies.csv'))
links = pd.DataFrame(pd.read_csv('/home/Llana13/movie_recommender/static/links.csv'))
ratings = pd.DataFrame(pd.read_csv('/home/Llana13/movie_recommender/static/ratings.csv'))
tags = pd.DataFrame(pd.read_csv('/home/Llana13/movie_recommender/static/tags.csv'))

movies_.drop_duplicates(subset='title',inplace=True)

# Created to get a list of all rated movies (titles and id)
ratings_movies = movies_.merge(ratings,how='left',on='movieId')

movies = list(ratings_movies['title'].unique())
moviesId = sorted(list(ratings_movies['movieId'].unique())) # id of every rated movie
users = range(1,611)

# Movie-Id dictionary
movie_dict = dict(zip(movies,moviesId))

# dictionary of moviesId and zeros
user_dict = dict(zip(moviesId,np.zeros(9719)))


# user_movie_ratings_matrix
R = ratings_movies.pivot(index='userId',columns='movieId',values='rating')

# Dropping movies that have not been rated
no_rated_movies = R.T[R.isna().all()].T.columns.tolist()
for id in no_rated_movies:
    R.drop(id,inplace=True,axis=1)

# Dropping the first row (NaN id)
R = R.iloc[1:]

binary = open('/home/Llana13/movie_recommender/model.bin', 'rb').read()
model = pickle.loads(binary)

# movie-genre matrix
Q = pd.DataFrame(model.components_, columns=R.columns.to_list(),index= range(model.n_components))
