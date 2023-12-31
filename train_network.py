import os
import numpy as np
import pandas as pd
import time

import copy
import json
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.embedding_net import EmbeddingNet
from utils import utils


# Setting resource paths
resource_path = os.path.join(os.path.dirname(__file__), 'data', 'ml-25m')
ratings_csv = os.path.join(resource_path, 'ratings.csv')
model_binaries_path = os.path.join(os.path.dirname(__file__), 'models', 'binaries')
log_file = os.path.join(model_binaries_path, "log.txt")


# Extracting data
ratings = pd.read_csv(ratings_csv)
ratings.drop(['timestamp'], axis=1, inplace=True)

utils.logger(log_file, f"Extracted data\n")
print("Extracted data")


# Pulling training config
config = dict() 
config = json.load(open('training_config.json'))


# Filtering top n movies
ratings = utils.filter_movies(ratings, config['n_movies'])

utils.logger(log_file, f"Filtered top {config['n_movies']} movies\n")
print(f"Filtered top {config['n_movies']} movies")


# Labelling the users and movies
userId_to_Label, userLabel_to_Id = utils.get_labels(ratings.userId)
movieId_to_Label, movieLabel_to_Id = utils.get_labels(ratings.movieId)

ratings['userLabel'] = userId_to_Label(ratings.userId)
ratings['movieLabel'] = movieId_to_Label(ratings.movieId)

utils.logger(log_file, f"Labeled Movies and Users\n")
print("Labeled Movies and Users")


# Getting number of users and movies
n_users = len(ratings["userLabel"].unique())
n_movies = len(ratings["movieLabel"].unique())


# Creating tensor train-test dataset
train_set, test_set = utils.train_test_split(ratings, config['test_size'])

utils.logger(log_file, f"Created and split Tensor dataset\n")
print("Created and split Tensor dataset")


# Creating Data Loader
train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=True)


# Initializing model
embedding_net = EmbeddingNet(n_users, n_movies, config['n_factors'])
criterion = nn.MSELoss()
optimizer = optim.Adam(embedding_net.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=10)

utils.logger(log_file, f"\nIntialized model\n{str(embedding_net)}\n\n")
print(f"Initialized model\n{str(embedding_net)}")

###########################################################################################################################

# Training
train_history = []
val_history = []
lr_history = []

# storing starting time
ts = time.time()

best_loss = np.inf
best_weights = None
for i in range(config['n_epochs']):
    train_loss = 0
    n_batches = 0

    # train phase
    for features, targets in tqdm(train_loader, desc=f'Epoch {i+1}'):
        optimizer.zero_grad()
        output = embedding_net(features[:,0], features[:,1])
        loss = criterion(output, targets)
 
        loss.backward()
        optimizer.step()
      
        train_loss += loss.detach()
        n_batches += 1

    scheduler.step()
      
    train_loss = train_loss/n_batches
    train_history.append(train_loss)
    lr_history.append(scheduler.get_last_lr())
  
    # validation
    n_batches = 0
    val_loss = 0
    with torch.no_grad():
        for features, targets in test_loader:
            output = embedding_net(features[:,0], features[:,1])
            loss = criterion(output, targets)
            val_loss += loss
            n_batches += 1  
    val_loss = val_loss/n_batches
    val_history.append(val_loss)
    if (val_loss < best_loss):
        best_weights = copy.deepcopy(embedding_net.state_dict())
        best_loss = val_loss

    text = f"Epoch {i+1} | Train_loss: {train_loss.numpy()} | Val_loss: {val_loss.numpy()} | Time_elapsed={(time.time()-ts)/60:.2f}min"
    print(text)
    utils.logger(log_file, text+"\n")

    model_name = f"Embedding_weights_Epoch{i+1}.pth"
    utils.save_model(embedding_net.state_dict(), n_users, n_movies, config['n_factors'], userLabel_to_Id, movieLabel_to_Id, model_name, model_binaries_path)

# Saving final weights
model_name = f"Embedding_weights_Final.pth"
utils.save_model(embedding_net.state_dict(), n_users, n_movies, config['n_factors'], userLabel_to_Id, movieLabel_to_Id, model_name, model_binaries_path)

utils.logger(log_file, f"Completed training in {(time.time()-ts)/60:.2f}mins")
print(f"Completed training in {(time.time()-ts)/60:.2f}mins")