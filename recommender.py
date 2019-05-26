import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as pyplot
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances 

#Read csv files for user and item related data
os.curdir
os.system("ls")
os.chdir(r"/home/trevor/Documents/Projects/recommender_system_data/ml-100k")

user_data = 'u.user'
rating_data = 'u.data'
rating_training = 'ua.base'
rating_test = 'ua.test'
item_data = 'u.item'

user_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv(user_data, sep='|', names=user_cols)

ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv(rating_data, sep='\t', names=ratings_cols,encoding='latin-1')
ratings_train = pd.read_csv(rating_training, sep='\t', names=ratings_cols, encoding='latin-1')
ratings_test = pd.read_csv(rating_test, sep='\t', names=ratings_cols, encoding='latin-1')

item_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv(item_data, sep='|', names=item_cols,encoding='latin-1')

#Get number of unique users and items
num_users = ratings.user_id.unique().shape[0]
num_items = ratings.movie_id.unique().shape[0]

matrix = np.zeros((num_users, num_items))
for line in ratings.itertuples():
    matrix[line[1]-1, line[2]-1] = line[3]

user_similarity = pairwise_distances(matrix, metric='cosine')
item_similarity = pairwise_distances(matrix.T, metric='cosine')

def predict(ratings, similarity, type='user'):
    if (type == 'user'):
        mean_user_rating = ratings.mean(axis=1)
        rating_difference = ratings - mean_user_rating[:, np.newaxis]
        prediction = mean_user_rating[:,np.newaxis] + similarity.dot(rating_difference) / np.array([np.abs(similarity).sum(axis=1)]).T

    elif (type == 'item'):
        prediction = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    
    return prediction

predictions = predict(matrix, item_similarity, 'item')

#Matrix factorization

class matrix_factorization():

    #setup data and hyperparameters 
    def __init__(self, R, n_features, learning_rate, beta, iterations):
        self.R = R
        self.n_users, self.n_items = R.shape
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.iterations = iterations

    def train(self):
        self.P = np.random.normal(scale=1/self.n_features, size=(self.num_items, self.n_features))

        self.Q = np.random.normal(scale=1/self.n_features, size= (self.num_items, self.n_features))

        self.b_u = np.zeroes(self.num_users)
        self.b_i = np.zeroes(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        self.samples = [
            (i,j, self.R[i,j]) for i in range(self.num_users) 
            for i in range(self.num_items) if self.Range[i,j >0 ]]
        
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i,mse))
            if (i+1) % 20 == 0:
                print("Iterations &d ; error = %.4f" % (i+1, mse))
        return training_process

    def mse(self):
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix
        error = 0
        for x, y, in zip(xs,ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def stoch_grad_descent(self):
        for i, j, r in self.samples:
            prediction = self.get_rating(i, j)
            e = (r - prediction)

        prediction = self.get_rating(i,j)
        e = (r - prediction)

        self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
        self.b_u[j] += self.alpha * (e - self.beta * self.b_i[j])

        self.P[i,:] += self.alpha * (e .self.Q[j, :] - self.beta * self.Q[j:])
        
        self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j, :])

    def get_rating(self, i, j):
        
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j,:].T)

        return prediction
    
    def full_matrix(self):
        return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)

#setup data
R = np.array(ratings.pivot(index = 'user_id', columns = 'movie_id', values = 'rating').fillna(0))

mf = matrix_factorization(R, k=20, alpha=0.001, beta=0.01, iterations = 250)
