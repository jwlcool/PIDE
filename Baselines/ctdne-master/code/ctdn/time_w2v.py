'''
__author__ : 'Shubhranshu Shekhar'
This file generates the time honoring random walks given a graph
'''
import gensim
import numpy as np
import pickle
from cmg import cmd_args,cmd_opt

def learn_node_representation(time_walks, size=cmd_args.embedding_size):
    documents = []
    for walk in time_walks:
        documents.append([str(w) for w in walk])

    # w2v model
    model = gensim.models.Word2Vec(documents, size=size, window=5, min_count=1, workers=4, negative=5)
    model.train(documents, total_examples=len(documents), epochs=10)

    # save the vectors into a dict and then pickle it
    node_representation = {}
    for key, val in model.wv.vocab.items():
        node_representation[int(key)] = model.wv[key]

    return node_representation


def main():
    path = cmd_args.save_dir+'ia-contact.time.walks'
    w2v_embedding_path = cmd_args.save_dir+'ia-contact.w2v.pkl'

    with open(path, 'rb') as f:
        time_walks = pickle.load(f)
    node_representation = learn_node_representation(time_walks)
    f.close()

    with open(w2v_embedding_path, 'wb') as f:
        pickle.dump(node_representation, f)
        f.close()

    with open(w2v_embedding_path, 'rb') as f:
        temp = pickle.load(f)
        f.close()


if __name__ == '__main__':
    main()
