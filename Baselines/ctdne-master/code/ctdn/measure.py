import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer
from sklearn.utils import check_X_y
import sys



def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim




def dcg_score(y_true, y_score, k=5):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gain = 2 ** y_true - 1
    # print(gain)
    discounts = np.log2(np.arange(len(y_true)) + 2)
    # print(discounts)
    return np.sum(gain / discounts)


def ndcg_score(y_true, y_score, k=5):
    y_score, y_true = check_X_y(y_score, y_true)
    # Make sure we use all the labels (max between the length and the higher
    # number in the array)
    lb = LabelBinarizer()
    lb.fit(np.arange(max(np.max(y_true) + 1, len(y_true))))
    binarized_y_true = lb.transform(y_true)
    print(binarized_y_true)
    if binarized_y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score have different value ranges")
    scores = []
    # Iterate over each y_value_true and compute the DCG score
    for y_value_true, y_value_score in zip(binarized_y_true, y_score):
        actual = dcg_score(y_value_true, y_value_score, k)
        best = dcg_score(y_value_true, y_value_true, k)
        # print(best)
        scores.append(actual / best)
    return np.mean(scores)


