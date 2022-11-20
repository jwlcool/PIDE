import numpy as np


def get_implict_matrix(rec_items, test_set):
    rel_matrix = [[0] * rec_items.shape[1] for _ in range(rec_items.shape[0])]
    for user in range(len(test_set)):
        for index, item in enumerate(rec_items[user]):
            if item in test_set[user]:
                rel_matrix[user][index] = 1
    return np.array(rel_matrix)


def DCG(items):
    return np.sum(items / np.log(np.arange(2, len(items) + 2)))


def nDCG(rec_items, test_set):
    assert rec_items.shape[0] == len(test_set)
    # 获得隐式反馈的rel分数矩阵
    rel_matrix = get_implict_matrix(rec_items, test_set)
    ndcgs = []
    ndcg=0
    dcgs=0
    dcg=0
    idcgs=0
    idcg = 0
    for user in range(len(test_set)):
        rels = rel_matrix[user]
        dcg = DCG(rels)
        idcg = DCG(sorted(rels, reverse=True))

        ndcg = dcg / idcg if idcg != 0 else 0
        ndcgs.append(ndcg)

    #     dcgs = dcgs + dcg
    #     idcgs=idcgs+idcg
    # ndcg=dcgs / idcgs
    # return ndcg
    return np.array(ndcgs).mean()
