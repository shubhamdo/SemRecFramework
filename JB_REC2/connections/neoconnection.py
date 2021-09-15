from py2neo import Graph


def connectToNeo4J():
    graph = Graph("bolt://localhost:7687", auth=('neo4j', 'root'))

    return graph
