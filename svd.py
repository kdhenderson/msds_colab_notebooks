# Nedelina Teneva
# Numerical Linear Algebra: https://cdn.bc-pf.org/resources/math/algebra/linear_algebra/Trefethen_Bau-Numerical_Linear_Algebra.pdf
# Sources: 
# https://alyssaq.github.io/2015/20150426-simple-movie-recommender-using-svd/
# https://towardsdatascience.com/beginners-guide-to-creating-an-svd-recommender-system-1fd7326d1f65

# Data:
# http://files.grouplens.org/datasets/movielens/ml-latest-small-README.html
# http://files.grouplens.org/datasets/movielens/ml-latest-small.zip

import numpy as np
import pandas as pd
np.seterr(divide='ignore', invalid='ignore')

from scipy.sparse.linalg import svds

import matplotlib.pyplot as plt

data = pd.io.parsers.read_csv('./ml-latest-small/ratings.csv')

movie_data = pd.io.parsers.read_csv('./ml-latest-small/movies.csv')
	
	
# matrix of number_users x number_movies 	
ratings_mat = np.ndarray(
    shape=(np.max(data.movieId.values), np.max(data.userId.values)),
    dtype=np.uint8)
ratings_mat[data.movieId.values-1, data.userId.values-1] = data.rating.values #assign values to tuples (mov_id, user_id)


normalised_mat = ratings_mat - np.asarray([(np.mean(ratings_mat, 1))]).T

A = normalised_mat.T / np.sqrt(ratings_mat.shape[0] - 1)
print(A.shape)

U, S, V = np.linalg.svd (A, full_matrices=False)

# V: idealized movie map. 
# U: idealized user map 
# [0 0 ] [1 1 ]^T = 0 
def top_cosine_similarity(data, movie_id, top_n=10):
    index = movie_id - 1 # Movie id starts from 1
    movie_row = data[index, :]
    
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    
    similarity = np.dot(movie_row, data.T) / (magnitude[index] * magnitude)
    
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]

# Helper function to print top N similar movies
def print_similar_movies(movie_data, movie_id, top_indexes):
    print('Recommendations for {0}: \n'.format(
    movie_data[movie_data.movieId == movie_id].title.values[0]))
    for id in (top_indexes + 1):
        print(movie_data[movie_data.movieId == id].title.values[0])
        
        
k = 30 # try 30, 50, 10 
movie_id = 1 # Grab an id from movies.dat
top_n = 10

sliced = V.T[:, :k] # representative data
indexes = top_cosine_similarity(sliced, movie_id, top_n)
print_similar_movies(movie_data, movie_id, indexes)

#let's set k based on the singular values
print ("top k: ", k)
plt.plot(S)
plt.show()

