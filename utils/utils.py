import os
from sklearn.preprocessing import LabelEncoder
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data_utils

def get_labels(items):
    le = LabelEncoder()
    le.fit(items)

    return (le.transform, le.inverse_transform)


def logger(path, text):
    """
    Function to write text logs to the given file

    Args:
        path (string): log file location
        text (string): log text to write to log file
    """

    with open(path, 'a') as f:
        f.write(text)



def filter_movies(ratings, n=-1):
    if n == -1:
        return ratings
    
    top_n_movies = ratings['movieId'].value_counts()[:n].index.tolist()
    ratings['top_n'] = ratings.movieId.apply(lambda id: id in top_n_movies)
    ratings = ratings[ratings['top_n'] == True].drop(['top_n'], axis=1)

    return ratings


def train_test_split(ratings, test_size=0.25):
    # Creating tensor dataset
    target = torch.tensor(ratings['rating'].values)[..., np.newaxis]
    features = torch.tensor(ratings[["userLabel", "movieLabel"]].values) 

    dataset = data_utils.TensorDataset(features, target)

    # Splitting dataset into train and test sets
    total_obs = len(dataset)
    test_obs = int(total_obs * test_size)
    train_obs = total_obs - test_obs
    split = [train_obs, test_obs]

    return data_utils.random_split(dataset, split)


def save_model(state_dict, n_users, n_movies, n_factors, userLabel_to_Id, movieLabel_to_Id, model_name, path):
    embedding_state = {"state_dict": state_dict,
                       "n_users": n_users,
                       "n_movies": n_movies,
                       "n_factors": n_factors,
                       "userLabel_to_Id": userLabel_to_Id,
                       "movieLabel_to_Id": movieLabel_to_Id}
    
    torch.save(embedding_state, os.path.join(path, model_name))


def get_closest_labels(label, movie_factors):
    distances = nn.CosineSimilarity(dim=1)(movie_factors, torch.unsqueeze(movie_factors[label], dim=0))

    closest_labels = distances.argsort(0).squeeze().tolist()
    closest_labels.remove(label)
    return closest_labels


def get_recommendations(label, movie_factors, movie_bias, weight=1, bias_weight=1, recommendations=dict()):
    if bias_weight>1: bias_weight=1
    if weight<1: weight=1

    closest_labels = get_closest_labels(label, movie_factors)
    n = len(closest_labels)

    for i, label in enumerate(closest_labels):
        new_rating = recommendations.get(label, 0) + weight * ( (n-i) * ((bias_weight * movie_bias[label].item()) + 1) )
        recommendations[label] = new_rating

    return recommendations