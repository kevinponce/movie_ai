import numpy as np 
import pandas as pd
# import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process
import redis
from redisgraph import Node, Edge, Graph, Path

r = redis.Redis(host='localhost', port=6379)
label_based_recommender_graph = Graph('label_based_recommender', r)

query = """MATCH (n) RETURN n""" # to see all nodes
result = label_based_recommender_graph.query(query)
movieIds = []
for record in result.result_set:
  label = record[0].label
  id = record[0].properties['id']
  title = record[0].properties['title']

  movieIds.append(id)

rating = pd.read_csv('datasets/rounakbanik/the-movies-dataset/ratings.csv', low_memory=False)
rating = rating.drop(['timestamp'],axis=1)
rating['movieId'] = rating['movieId'].astype(int)
print(rating.head(2))

movie = pd.read_csv('datasets/rounakbanik/the-movies-dataset/movies_metadata.csv', low_memory=False)
movie = movie.drop([19730, 29503, 35587])
movie = movie.drop(['adult', 'belongs_to_collection', 'budget', 'genres', 'homepage', 'status', 'tagline', 'video', 'vote_average', 'vote_count', 'imdb_id', 'original_language', 'original_title', 'overview', 'popularity', 'production_countries', 'release_date', 'revenue', 'runtime', 'spoken_languages', 'poster_path', 'production_companies'],axis=1)
movie['id'] = movie['id'].astype(int)

title_lookup = {}
for index, row in movie.iterrows():
    title_lookup[row['id']] = row['title']

ratings_thres = 50
df_users_cnt = pd.DataFrame(rating.groupby('userId').size(), columns=['count'])
active_users = list(set(df_users_cnt.query('count >= @ratings_thres').index))

popularity_thres = 50
df_movies_cnt = pd.DataFrame(rating.groupby('movieId').size(), columns=['count'])
df_movies_cnt['count'].quantile(np.arange(1, 0.6, -0.05))
popular_movies = list(set(df_movies_cnt.query('count >= @popularity_thres').index))

df_ratings_drop_movies = rating[rating.movieId.isin(popular_movies)]
df_ratings_drop_users = df_ratings_drop_movies[df_ratings_drop_movies.userId.isin(active_users)]
print(movie.head(2))

#Applying the strip function to get rid of any ending whitespace characters that may have appeared
movie['title'] = movie['title'].str.strip()
movie.head()
print(movie.head(2))

print(df_ratings_drop_movies.head(2))
print(df_ratings_drop_users.head(2))


# df_ratings_drop_users = df_ratings_drop_users.head(5000)
df_ratings_drop_users = rating[rating.movieId.isin(movieIds)]

print(df_ratings_drop_users)
print("??????")
print(movieIds)


movie_user_mat = df_ratings_drop_users.pivot(index='movieId', columns='userId', values='rating').fillna(0)
matrix_movies_users = csr_matrix(movie_user_mat.values)

knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=7)
knn.fit(matrix_movies_users)

def recommender(movieId, data,model, n_recommendations ):
    my_list = []
    try:
        model.fit(data)
        idx=process.extractOne(movieId, df_ratings_drop_users['movieId'])[2]
        print(idx)
        print('Movie Selected:-',df_ratings_drop_users['movieId'][idx], 'Index: ',idx)
        print('Searching for recommendations.....')
        distances, indices=model.kneighbors(data[idx], n_neighbors=n_recommendations)
        for i in indices:
            dt=np.dtype('int,int')
            xarr = np.array(list(enumerate(df_ratings_drop_users['movieId'][i].where(i!=idx))),dtype=dt)

            for movieId in xarr['f1'] :
                my_list.append(movieId)

        return my_list
    except:
        return my_list

# print("???????????????")
# print(recommender("147", matrix_movies_users, knn,5))

#########################################
import redis
from redisgraph import Node, Edge, Graph, Path

r = redis.Redis(host='localhost', port=6379)
redis_graph = Graph('knn_based_recommender', r)
redis_graph.delete()

hashmap = {}
for index, row in df_ratings_drop_users.head(10).iterrows():
    id = int(row['movieId'])
    if title_lookup.get(id) is not None:
        print("?????")
        print(id)
        print(title_lookup[id])
        
        if hashmap.get(id) is None:
            hashmap[id] = Node(label='movie', properties={'title': title_lookup[id], 'id': id})
            redis_graph.add_node(hashmap[id])

        options = recommender(str(id), matrix_movies_users, knn,5)
        print('^^^^^^')
        print(options)

        for movieId in options:
            # movieId = int(movieId)
            print("@@@@@")
            print(movieId)
            print(title_lookup.get(movieId))

            if title_lookup.get(movieId) is not None:
                print(movieId)
                print(title_lookup[movieId])

                if hashmap.get(movieId) is None:
                    hashmap[movieId] = Node(label='movie', properties={'title': title_lookup[movieId], 'id': movieId})
                    redis_graph.add_node(hashmap[movieId])

                redis_graph.add_edge(Edge(hashmap[id], 'content_based', hashmap[movieId], properties={}))
                redis_graph.add_edge(Edge(hashmap[movieId], 'content_based', hashmap[id], properties={}))

redis_graph.commit()
print(hashmap)
