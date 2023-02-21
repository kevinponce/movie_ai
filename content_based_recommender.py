# tutorial: https://www.datacamp.com/tutorial/recommender-systems-python
# data: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?resource=download

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

metadata = pd.read_csv('datasets/rounakbanik/the-movies-dataset/movies_metadata.csv', low_memory=False)

title_lookup = {}
for index, row in metadata.iterrows():
    title_lookup[row['id']] = row['title']

# Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

# Replace NaN with an empty string
metadata['overview'] = metadata['overview'].fillna('')

# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(metadata['overview'])

# Output the shape of tfidf_matrix
# print(tfidf_matrix.shape)

# print(metadata['overview'].head())

# Array mapping from feature integer indices to feature name.
# print(tfidf.get_feature_names_out())

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# print(cosine_sim.shape)

#Construct a reverse map of indices and movie titles
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

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

# print(get_recommendations('The Dark Knight Rises'))
# print(get_recommendations('Heat'))

#########################################
import redis
from redisgraph import Node, Edge, Graph, Path

r = redis.Redis(host='localhost', port=6379)
redis_graph = Graph('content_based_recommender', r)


hashmap = {}
for index, row in metadata.head(10).iterrows():
    id = row['id']

    if hashmap.get(id) is None:
        hashmap[id] = Node(label='movie', properties={'title': row['title'], 'id': id})
        redis_graph.add_node(hashmap[id])

    for movieId in get_recommendations(row['title']):
        if hashmap.get(movieId) is None:
            hashmap[movieId] = Node(label='movie', properties={'title': title_lookup[movieId], 'id': movieId})
            redis_graph.add_node(hashmap[movieId])

        redis_graph.add_edge(Edge(hashmap[id], 'content_based', hashmap[movieId], properties={}))
        redis_graph.add_edge(Edge(hashmap[movieId], 'content_based', hashmap[id], properties={}))

redis_graph.commit()
print(hashmap)
