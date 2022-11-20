import networkx as nx


def get_neigbors(g, node, depth=1):
    output = {}
    layers = dict(nx.bfs_successors(g, source=node))
    nodes = [node]
    for i in range(1,depth+1):
        output[i] = []
        for x in nodes:
            output[i].extend(layers.get(x,[]))
        nodes = output[i]
    return output

def kg_add(kg,node1,node2,t):
    kg.add_node(node1)
    kg.add_node(node2)
    kg.add_edge(node1,node2,time=t)