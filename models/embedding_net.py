import torch
import torch.nn as nn

class EmbeddingNet(nn.Module):

    def __init__(self, n_users, n_movies, n_factors = 10, y_range=(0.0,5.0)):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_factors = nn.Embedding(n_movies, n_factors)
        self.movie_bias = nn.Embedding(n_movies, 1)
        self.y_range = y_range

        self.double()


    def sigmoid_range(self, x, low, high):
        return torch.sigmoid(x) * (high - low) + low

      
    def forward(self, user_idx, movie_idx):
        users = self.user_factors(user_idx)
        movies = self.movie_factors(movie_idx)
        rating = (users * movies).sum(dim=1, keepdim=True)
        rating += self.user_bias(user_idx) + self.movie_bias(movie_idx)
        return self.sigmoid_range(rating, self.y_range[0], self.y_range[1])