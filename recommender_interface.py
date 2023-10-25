import os
import pandas as pd
import gradio as gr

import torch

from utils import utils

import warnings
warnings.filterwarnings("ignore")


model_path = os.path.join(".", 'models', 'binaries')
models = [model for model in os.listdir(model_path) if model.endswith(".pth")]
models.sort()

final_model = models[-1]
final_model_path = os.path.join(model_path, final_model)

del models


embedding_state = torch.load(final_model_path)

n_users = embedding_state['n_users']
n_movies = embedding_state['n_movies']
n_factors = embedding_state['n_factors']

movie_bias = embedding_state['state_dict']['movie_bias.weight']
movie_factors = embedding_state['state_dict']['movie_factors.weight']

userLabel_to_Id = embedding_state["userLabel_to_Id"]
movieLabel_to_Id = embedding_state["movieLabel_to_Id"]

del embedding_state, final_model_path


resource_path = os.path.join(".", 'data', 'ml-25m')
movies_csv = os.path.join(resource_path, 'movies.csv')
links_csv = os.path.join(resource_path, 'links.csv')

movies = pd.read_csv(movies_csv)
movies['genres'] = movies.genres.apply(lambda genres: genres.split('|'))

links = pd.read_csv(links_csv, dtype = {'movieId': int, 'imdbId': str, 'tmdbId': str})
movies = movies.merge(links)
del links


def get_movie_id(label):
    id = movieLabel_to_Id([label])
    return id.item(0)

movies_dict = {
    'label': [],
    'title': [],
    'genres':  [],
    'imdbId': []
}

for label in range(n_movies):
    movies_dict['label'].append(label)
    movies_dict['title'].append(movies[movies['movieId']==get_movie_id(label)]['title'].item())
    movies_dict['genres'].append(movies[movies['movieId']==get_movie_id(label)]['genres'].item())
    movies_dict['imdbId'].append(movies[movies['movieId']==get_movie_id(label)]['imdbId'].item())

movies = pd.DataFrame(movies_dict)
del movies_dict

user_movies = []


def get_movie_name(label):
    return movies[movies['label']==label]['title'].item()

def add_movie_to_list(movie_title, rating):
    label = movies[movies['title']==movie_title].label.item()
    
    user_movies.append((label, rating))

def disp_user_movies():
    if not len(user_movies):
        return gr.Markdown('No movies added to list yet')
    
    md_table = 'No. | Movie | Rating\n---|---|---'
    sorted_user_movies = sorted(user_movies, key=lambda x:x[1], reverse=True)
    for i, (label, rating) in enumerate(sorted_user_movies):
        md_table += f'\n{i+1} | {get_movie_name(label)} | {rating}'

    return gr.Markdown(md_table)

def disp_user_recommendations(bias_weight):
    if not len(user_movies):
        return gr.Markdown('No movies added to list yet')
    
    recommendations = dict()
    for label, rating in user_movies:
        recommendations = utils.get_recommendations(label, movie_factors, movie_bias, rating, bias_weight, recommendations)

    recommendations = sorted(recommendations.items(), key=lambda x:x[1], reverse=True)
    recommendations = [label for label, _ in recommendations][:20]

    md_table = 'No. | Movie\n---|---'
    for i, label in enumerate(recommendations):
        md_table += f'\n{i+1} | {get_movie_name(label)}'

    return gr.Markdown(md_table)

def set_bw(bias_wt):
    return disp_user_recommendations(bias_wt)

def add_movie(movie_title, rating, bias_wt):
    add_movie_to_list(movie_title, rating)

    return [disp_user_movies(), disp_user_recommendations(bias_wt)]

def clear_user_movies():
    user_movies.clear()

    return [disp_user_movies(), disp_user_recommendations(0.2)]



movie_list = sorted(movies.title.values.tolist())

with gr.Blocks() as app:
    gr.Markdown('<h1 style="text-align: center;">Movie Recommendation System</h1>')
    gr.Markdown('<p style="text-align: center;">Project by: Alex Prateek Shankar (2248302) and Sneh Shah (2248318)</p>')

    with gr.Blocks():
        movie_title = gr.Dropdown(
            choices=movie_list,
            label="Select Movie"
            )
        rating = gr.Slider(
            0.0, 5.0, 
            value=2.5, 
            step=0.5, 
            label="Rating", 
            info="Give your rating for the movie",
            interactive=True
            )
        add_btn = gr.Button(
            "Add Movie",
            label="Click here to add movies to your list"
            )
        
    gr.Markdown('<hr>')

    with gr.Blocks():
        gr.Markdown('<h2 style="text-align: center;">My Movies</h2>')

        my_movies = disp_user_movies()

        clear_movies = gr.Button(
            "Clear My Movies",
            label="Click here to clear your movies list"
        )

    gr.Markdown('<hr>')

    with gr.Blocks():
        gr.Markdown('<h2 style="text-align: center;">Bias Weight</h2>')

        bias_wt = gr.Slider(
            0.0, 1.0,
            value=0.2,
            step=0.1,
            label="Bias Weight",
            info="This will affect how the pre-existing movie ratings will affect your recommendations"
        )
        bw_btn = gr.Button(
            "Set Bias Weight",
            label="Click here to set the Bias Weight"
        )

    gr.Markdown('<hr>')

    with gr.Blocks():
        gr.Markdown('<h2 style="text-align: center;">My Top 20 Recommendations</h2>')

        my_recommendations = disp_user_recommendations(0.2)

    clear_movies.click(fn=clear_user_movies, outputs=[my_movies, my_recommendations])
    bw_btn.click(fn=set_bw, inputs=[bias_wt], outputs=[my_recommendations])
    add_btn.click(fn=add_movie, inputs=[movie_title, rating, bias_wt], outputs=[my_movies, my_recommendations])


app.title = "Neural Collaborative Filtering"
app.launch()