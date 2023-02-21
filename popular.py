# tutorial: https://www.datacamp.com/tutorial/recommender-systems-python
# data: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?resource=download

import pandas as pd

metadata = pd.read_csv('datasets/rounakbanik/the-movies-dataset/movies_metadata.csv', low_memory=False)

# Calculate mean of vote average column
C = metadata['vote_average'].mean()
# print(C)

# Calculate the minimum number of votes required to be in the chart, m
m = metadata['vote_count'].quantile(0.90)
# print(m)

# Filter out all qualified movies into a new DataFrame
q_movies = metadata.copy().loc[metadata['vote_count'] >= m]
# print(q_movies.shape)
# print(metadata.shape)

# Function that computes the weighted rating of each movie
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

#Sort movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending=False)

#Print the top 15 movies
q_movies[['title', 'vote_count', 'vote_average', 'score']].head(15)

print(q_movies)

print(q_movies.head(10))

for index, row in q_movies.head(10).iterrows():
    print(row['title'])
    print(row['id'])
    print("??????")
