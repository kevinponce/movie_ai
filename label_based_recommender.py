# tutorial: https://www.datacamp.com/tutorial/recommender-systems-python
# data: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?resource=download

import pandas as pd
from ast import literal_eval
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

metadata = pd.read_csv('datasets/rounakbanik/the-movies-dataset/movies_metadata.csv', low_memory=False)

# Load keywords and credits
credits = pd.read_csv('datasets/rounakbanik/the-movies-dataset/credits.csv')
keywords = pd.read_csv('datasets/rounakbanik/the-movies-dataset/keywords.csv')

# Remove rows with bad IDs that break the code bellow...
metadata = metadata.drop([19730, 29503, 35587])

# Convert IDs to int. Required for merging
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
metadata['id'] = metadata['id'].astype('int')

title_lookup = {}
for index, row in metadata.iterrows():
    title_lookup[row['id']] = row['title']

# Merge keywords and credits into your main metadata dataframe
metadata = metadata.merge(credits, on='id')
metadata = metadata.merge(keywords, on='id')

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(literal_eval)

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []

# Define new director, cast, genres and keywords features that are in a suitable form.
metadata['director'] = metadata['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(get_list)

# print(metadata[['title', 'cast', 'director', 'keywords', 'genres']].head(3))

# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
# Apply clean_data function to your features.
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    metadata[feature] = metadata[feature].apply(clean_data)

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

metadata['soup'] = metadata.apply(create_soup, axis=1)
# print(metadata[['soup']].head(2))

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(metadata['soup'])
# print(count_matrix.shape)


cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Reset index of your main DataFrame and construct reverse mapping as before
metadata = metadata.reset_index()
indices = pd.Series(metadata.index, index=metadata['title'])

# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    try:
        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar movies
        sim_scores = sim_scores[1:11]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]

        # Return the top 10 most similar movies
        return metadata['id'].iloc[movie_indices]
    except:
        return []

print(get_recommendations('The Dark Knight Rises', cosine_sim))
print("??????")

#########################################
import redis
from redisgraph import Node, Edge, Graph, Path

r = redis.Redis(host='localhost', port=6379)
redis_graph = Graph('label_based_recommender', r)

hashmap = {}
for index, row in metadata.head(10).iterrows():
    id = row['id']
    print(id)
    print(row['title'])

    if hashmap.get(id) is None:
        hashmap[id] = Node(label='movie', properties={'title': row['title'], 'id': id})
        redis_graph.add_node(hashmap[id])

    print(get_recommendations(row['title']))
    for movieId in get_recommendations(row['title']):
        print(movieId)
        print(title_lookup[movieId])
        if hashmap.get(movieId) is None:
            hashmap[movieId] = Node(label='movie', properties={'title': title_lookup[movieId], 'id': movieId})
            redis_graph.add_node(hashmap[movieId])

        redis_graph.add_edge(Edge(hashmap[id], 'content_based', hashmap[movieId], properties={}))
        redis_graph.add_edge(Edge(hashmap[movieId], 'content_based', hashmap[id], properties={}))

redis_graph.commit()
print(hashmap)
