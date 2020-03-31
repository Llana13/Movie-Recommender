import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.decomposition import NMF
from sklearn.impute import KNNImputer
import pickle

links = pd.DataFrame(pd.read_csv('Movie-Recommender/links.csv'))
movies_ = pd.DataFrame(pd.read_csv('Movie-Recommender/movies.csv'))
ratings = pd.DataFrame(pd.read_csv('Movie-Recommender/ratings.csv'))
tags = pd.DataFrame(pd.read_csv('Movie-Recommender/tags.csv'))

# Take columns from movies into links
links['title'] = movies_['title']
links['genres'] = movies_['genres']

# Set Indexes
links.set_index('movieId' ,inplace=True)
ratings.set_index(['movieId' ,'userId'], inplace=True)
tags.set_index(['movieId', 'userId'], inplace=True)

# Merge ratings and links into 'df'
links_ratings = pd.merge(left = links, right = ratings, left_index = True, right_index = True)

df = pd.merge(left = links_ratings, right = tags, how = 'left',left_on = ['movieId', 'userId'], right_on = ['movieId' ,'userId'])

df.rename(columns = {'timestamp_x': 'timestamp_rating','timestamp_y': 'timestamp_tag'}, inplace = True)

 ### NMF ###

ratings_ = pd.DataFrame(pd.read_csv('Movie-Recommender/ratings.csv'))

movies = df['title'].to_list()
moviesId = movies_['movieId'].to_list()
users = range(1,611)

# user_movie_ratings_matrix
R = ratings_.pivot(index = 'userId', columns = 'movieId', values = 'rating')

### K-Nearest-Neighbor (KNN) ###

imputer = KNNImputer()
Rtrans = imputer.fit_transform(R)

### Non-Negative Matrix Factorization (NMF) ###

model = NMF(n_components=100)
model.fit(Rtrans)

### Save the model ###

binary = pickle.dumps(model)
open('model.bin', 'wb').write(binary)
