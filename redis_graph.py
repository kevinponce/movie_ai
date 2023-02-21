import redis
from redisgraph import Node, Edge, Graph, Path

r = redis.Redis(host='localhost', port=6379)

# print(r.ping())

# r.sadd("primes", 2, 3, 5, 7)
# print(r.sismember("primes", 7))

redis_graph = Graph('social', r)

john = Node(label='person', properties={'name': 'John Doe', 'age': 33, 'gender': 'male', 'status': 'single'})
redis_graph.add_node(john)

japan = Node(label='country', properties={'name': 'Japan'})
redis_graph.add_node(japan)

edge = Edge(john, 'visited', japan, properties={'purpose': 'pleasure'})
redis_graph.add_edge(edge)

redis_graph.commit()

query = """MATCH (p:person)-[v:visited {purpose:"pleasure"}]->(c:country)
           RETURN p.name, p.age, v.purpose, c.name"""

result = redis_graph.query(query)

# Print resultset
result.pretty_print()

# Use parameters
params = {'purpose':"pleasure"}
query = """MATCH (p:person)-[v:visited {purpose:$purpose}]->(c:country)
           RETURN p.name, p.age, v.purpose, c.name"""

result = redis_graph.query(query, params)

# Print resultset
result.pretty_print()