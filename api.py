from flask import Flask, render_template, make_response, jsonify, request
from flask_cors import CORS, cross_origin
import redis
from redisgraph import Node, Edge, Graph, Path

r = redis.Redis(host='localhost', port=6379)

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

graph_type = 'knn_based_recommender' # content_based_recommender label_based_recommender knn_based_recommender

@app.route('/movies')
@cross_origin()
def movies():
    movie = request.args.get('movie')
    redis_graph = Graph(graph_type, r)

    query = """MATCH (n) RETURN n.id, n.title"""
    result = redis_graph.query(query)

    movies = []
    for record in result.result_set:
        movies.append({ 'id': record[0], 'title': record[1]})

    return jsonify({ 'movies' : movies })


@app.route('/find')
@cross_origin()
def find():
    movie = request.args.get('movie')
    redis_graph = Graph(graph_type, r)
    query = f'MATCH (m:movie)-[s:content_based]->(c:movie) WHERE m.title = "{movie}" RETURN c.id, c.title'

    result = redis_graph.query(query)

    movies = []
    for record in result.result_set:
        movies.append({ 'id': record[0], 'title': record[1]})

    return jsonify({ 'movies' : movies })

if __name__ == "__main__":
    app.run(host="localhost", port=1234, debug=True)