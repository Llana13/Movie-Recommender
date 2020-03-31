import random
import variables as V
import re
import pickle
import numpy as np
from heapq import nlargest
from sklearn.decomposition import NMF
'''
Update model binary file in this folder
'''
MOVIES = V.movies

binary = open('model.bin', 'rb').read()
model = pickle.loads(binary)

def match_movie(movie_dict, fav_movies, n):
    '''
    match the user input with a movie that is actually in the database
    '''
    return {x: y for x, y  in V.movie_dict.items() if re.match('.*%s.*' % fav_movies[n], str(x), re.IGNORECASE)}

def get_recommendation(fav_movies, n):

    # Match the film with one in the database
    matches = [match_movie(V.movie_dict,fav_movies,0),match_movie(V.movie_dict,fav_movies,1),match_movie(V.movie_dict,fav_movies,2)]
    # Get the id of the movies and create a vector with all films rated = 0 but each matched 5
    for i in range(3):
        film = matches[i]
        try:
            for x in range(20):
                Id = film.get(list(film.keys())[x])
                V.user_dict[Id] = 5
        except IndexError:
            V.user_dict[Id] = 5
    user_vector = list(V.user_dict.values())

    # weight for each feature for this user
    user_weights = model.transform([user_vector])

    # final recommendations (rating for each film)
    reco_vector = np.dot(user_weights,V.Q)

    # Make a dictionary with titles and recommendation value
    reco_dict = dict(zip(MOVIES,reco_vector[0].tolist()))

    # get the film with highest value
    best_matches = nlargest(n,reco_dict, key=reco_dict.get)

    n = min([len(MOVIES),n])
    return nlargest(n,reco_dict, key=reco_dict.get)
