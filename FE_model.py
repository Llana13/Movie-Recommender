'''
Feature Engineering and model training
'''
import pickle
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.impute import KNNImputer

links = pd.DataFrame(pd.read_csv('links.csv'))
movies_ = pd.DataFrame(pd.read_csv('movies.csv'))
ratings = pd.DataFrame(pd.read_csv('ratings.csv'))
tags = pd.DataFrame(pd.read_csv('tags.csv'))

# Take columns from movies into links
links['title'] = movies_['title']
links['genres'] = movies_['genres']

# Set Indexes
links.set_index('movieId', inplace=True)
ratings.set_index(['movieId', 'userId'], inplace=True)
tags.set_index(['movieId', 'userId'], inplace=True)

# Merge links, ratings and tags into "df"
links_ratings = pd.merge(left=links, right=ratings, left_index=True, right_index=True)
df = pd.merge(left=links_ratings, right=tags, how='left', left_on=['movieId', 'userId'], right_on=['movieId', 'userId'])
df.rename(columns={'timestamp_x': 'timestamp_rating', 'timestamp_y': 'timestamp_tag'}, inplace=True)

 ### Non-Negative Matrix Factorization ###
ratings_ = pd.DataFrame(pd.read_csv('ratings.csv'))
users = range(1, 611)

# user_movie_ratings_matrix
R = ratings_.pivot(index='userId', columns='movieId', values='rating')

### K-Nearest-Neighbor (KNN) ###
imputer = KNNImputer()
Rtrans = imputer.fit_transform(R)

### Non-Negative Matrix Factorization (NMF) ###
model = NMF(n_components=100)
model.fit(Rtrans)

# movie-genre matrix
Q = pd.DataFrame(model.components_, columns=R.columns.to_list(), index=range(model.n_components))

# user-genre matrix
P = pd.DataFrame(model.transform(Rtrans), columns=range(model.n_components), index=users)

# Reconstructed matrix
Rhat = pd.DataFrame(np.dot(P, Q))

# Reconstruction error
model.reconstruction_err_

### Save the model ###
binary = pickle.dumps(model)
open('nmf_model.bin', 'wb').write(binary)
