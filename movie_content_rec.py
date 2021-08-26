# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 19:15:18 2021

@author: Javier Garces

Content-based Movie Recommendation System
"""

import pandas as pd 
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

MOVIES_FILEPATH = './tmdb_5000_movies.csv'

#%% Data Acquisition

def read_files(): 
    df1 = pd.read_csv(MOVIES_FILEPATH)
    df2 = pd.read_csv(LINKS_FILEPATH)
    df3 = pd.read_csv(RATINGS_FILEPATH)
    return df1, df2, df3

df_movies, df_links, df_ratings = read_files()

#%% Data Preparation

def data_preparation(movies, links, ratings):
    # movies
    # rename id to tmdbId
    movies.rename(columns={'id':'tmdbId'}, inplace=True)
    
    # links
    # assign 0 to NULL tmdbIds
    notnull_tmdbIds = pd.to_numeric(links['tmdbId'], errors='coerce').notnull()
    links[~notnull_tmdbIds] = 0
    # turn tmdbId to int
    links['tmdbId'] = links['tmdbId'].astype(int)
    
    # ratings
    # only users with at least MIN_REVIEW_USERS reviews
    user_counts = ratings['userId'].value_counts()
    user_filter = user_counts[user_counts >= MIN_REVIEW_USERS].index
    ratings[ratings['userId'].isin(user_filter)]
    # only movies with at least MIN_REVIEWED_MOVIES reviews
    movie_counts = ratings['movieId'].value_counts()
    movie_filter = movie_counts[movie_counts >= MIN_REVIEWED_MOVIES].index
    ratings[ratings['movieId'].isin(movie_filter)]
    
    # new links (temporary)
    # remove entries with tmdbIds not in movies
    links = links.merge(movies)[links.columns]
    # remove entries with movieIds not in ratings
    movies_in_ratings = pd.DataFrame({'movieId': ratings['movieId'].unique()})
    links = links.merge(movies_in_ratings)[links.columns]
    
    # new movies
    # only movies with tmdbId present in links (merge with links)
    movies = movies.merge(links)
    
    # new ratings
    # only ratings with imdbId present in links
    ratings = ratings.merge(links)[ratings.columns]
    
    return movies, ratings

movies, ratings = data_preparation(df_movies, df_links, df_ratings)

#%% Approach 1: 
# Recommendations by Content Filtering: manually selected info.
# Create similarity matrix

def get_content_filter_matrix_1(movies, weights):
    # for each item in "weights" dict
    # we create a weighted list of elements
    def get_soup_terms(item, w):
        # create list of items to append to SOUP list
        names = []
        # if item is string, try to convert to list or dict
        if type(item) is str:
            try:
                item = json.loads(item)
            except json.JSONDecodeError:
                # item is string
                names.extend([item]*w) if w is int else names.extend([item]*w[0])
        if type(item) is list:
            if len(item) == 0:
                return []
            if type(item[0]) is dict:
                if type(w) is int:
                    for elem in item:
                        names.extend([str(elem.get('name', ''))] * w)
                if type(w) is list:
                    if len(item) < len(w):
                        w = w[:len(item)]
                    for i, elem in enumerate(item[:len(w)]):
                        names.extend([str(elem.get('name', ''))] * w[i])
            if type(item[0]) is list:
                if type(w) is int:
                    for elem in item:
                        names.extend([elem] * weight)
                if type(w) is list:
                    if type(w[0]) is int:
                        if len(item) < len(w):
                            w = w[:len(item)]
                        for i, elem in enumerate(item[:len(w)]):
                            names.extend([elem] * w[i])
        return names
    
    # we turn lists into string of wordswithnospaces
    def create_soup(x):
        clean_soup = [i.lower().replace(" ","") for i in x]
        return ' '.join(clean_soup)
    
    # from movies, we populate a "soup" Series
    # each element, a string of wordswithnospaces
    soup = pd.Series([[] for _ in range(len(movies))], index=movies.index)
    for item, weight in weights.items():
        soup = soup + movies[item].apply(get_soup_terms, args=(weight,))
    soup = soup.apply(create_soup)
    
    # TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(soup)
    
    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim
    
weights = {
    'genres': [4,3,2,1],
    'keywords': 1,
    'spoken_languages': [4],
    'production_companies': 3
}

sim_matrix_1 = get_content_filter_matrix_1(movies, weights)

#%% Approach 2: 
# Recommendations by Content Filtering: overview text similarity.
# Create similarity matrix

def get_content_filter_matrix_2(movies):
    # replace missing values for empty strings
    movies['overview'] = movies['overview'].fillna('')
    
    # TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['overview'])
    
    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim
    
    
    # Define a TF-IDF Vectorizer Object.
    # Remove all english stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(stop_words='english')

    #Replace NaN with an empty string

    tfidf_matrix = tfidf.fit_transform(movies['overview'])

sim_matrix_2 = get_content_filter_matrix_2(movies)

#%% Print

#Construct a reverse map of indices and movie titles
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim, rec_n = 10):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:rec_n+1]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return movies['title'].iloc[movie_indices]

print(tfidf_matrix.shape)

for i in range(PRINT_FROM, PRINT_FROM + PRINT_EXAMPLES):
    print("Movie: {}".format(movies.iloc[i]["title"]))
    print("Scores:", end= "")
    for n, j in enumerate(tfidf_matrix[i].indices):
        print("\t" + tfidf.get_feature_names()[j] + ": ", end="")
        print(tfidf_matrix[i].data[n])
    print("Similar:", end="")
    for j in get_recommendations(movies.iloc[i]["title"], rec_n = 5):
        print("\t" + j)
    print("")
    

        

