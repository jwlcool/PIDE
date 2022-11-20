import random
import networkx as nx

def dump_paths(Graph, user_id,item_id,user_list,item_list, maxLen, sample_size):
    '''
    dump the postive or negative paths

    Inputs:
        @Graph: the well-built knowledge graph
        @rating_pair: positive_rating or negative_rating
        @maxLen: path length
        @sample_size: size of sampled paths between user-movie nodes
    '''
    if Graph.has_node(user_id) and Graph.has_node(item_id):
        # print(user_id,movie_id)
        path_between_nodes=mine_paths_between_nodes(Graph, user_id, item_id,user_list,item_list, maxLen, sample_size)
        return path_between_nodes


def mine_paths_between_nodes(Graph, user_node, location_node,user_list,item_list, maxLen, sample_size):
    # connected_path = []
    # for path in nx.all_simple_paths(Graph, source=user_node, target=movie_node, cutoff=maxLen):
    #     count = 0
    #     if len(path) == maxLen + 1:
    #         for i,value in enumerate(path):
    #             if i%2!=0 and value in item_list:count+=1
    #             if i%2==0 and value in user_list:count+=1
    #     if count==4:
    #         connected_path.append(path)
    # path_size = len(connected_path)
    # if path_size > sample_size:
    #     random.shuffle(connected_path)
    #     connected_path = connected_path[:sample_size]
    # return connected_path
    path_list=[]
    if Graph.has_node(user_node) and Graph.has_node(location_node):
        for path in nx.all_simple_paths(Graph, source=user_node, target=location_node, cutoff=maxLen):
            path_list.append(path)
        return path_list