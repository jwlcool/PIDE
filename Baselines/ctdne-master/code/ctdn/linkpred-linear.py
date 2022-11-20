import pickle
import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from torch import optim

from Linear import MLP
from cmg import cmd_args,cmd_opt
import torch.nn.functional as F

from measure import cos_sim


def load_node_embeddings(path_to_w2v):
    '''
    load the saved word2vec representation
    :return:
    '''
    with open(path_to_w2v, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings


def load_edges(save_path):
    with open(save_path + 'pos_edges', 'rb') as f:
        pos_edges = pickle.load(f)
    with open(save_path + 'neg_edges', 'rb') as f:
        neg_edges = pickle.load(f)

    return pos_edges, neg_edges


def operator(u, v, op):
    if op=='mean':
        return (u + v)/2.0
    elif op=='l1':
        return np.abs(u - v)
    elif op == 'l2':
        return np.abs(u - v)**2
    elif op == 'hadamard':
        return np.multiply(u, v)
    else:
        return None


def get_dataset_from_embedding(embeddings, pos_edges, neg_edges, op='mean'):
    '''
    op can take values from 'mean', 'l1', 'l2', 'hadamard'
    '''
    y = []
    X = []

    # process positive links
    for u, v, prop in pos_edges:
        # get node representation and average them
        u_enc = embeddings.get(u)
        v_enc = embeddings.get(v)

        if (u_enc is None) or (v_enc is None):
            continue

        datapoint = operator(u_enc, v_enc, op=op)  # (u_enc + v_enc)/2.0

        X.append(datapoint)
        y.append(0.0)

    # process negative links
    for u, v, prop in neg_edges:
        # get node representation and average them
        u_enc = embeddings.get(u)
        v_enc = embeddings.get(v)

        if (u_enc is None) and (v_enc is not None):
            u_enc = v_enc
        if (v_enc is None) and (u_enc is not None):
            v_enc = u_enc
        if (u_enc is None) and (v_enc is None):
            continue

        datapoint = operator(u_enc, v_enc, op=op)  # (u_enc + v_enc) / 2.0

        X.append(datapoint)
        y.append(1.0)

    dataset = np.array(X), np.array(y)
    return dataset



def compute_HR(k,embeddings,user_list,item_list,user_item,item):
    user_rec={}
    for u in user_list:
        score_list=[]
        user_rank={}
        user_rec[u]=[]
        u_enc = embeddings.get(u)
        if (u_enc is None):
            continue
        for v in item_list:
            v_enc = embeddings.get(v)
            if (v_enc is None):
                continue
            score=cos_sim(u_enc,v_enc)
            score_list.append(score)
        score_list = np.array(score_list)
        ranks = rankdata(score_list)
        for id,rank in enumerate(ranks):
            user_rank[rank]=id
        user_rank=sorted(user_rank.items(),key=lambda x:x[0])
        for j in user_rank:
            if j[0]>k:
                break
            user_rec[u].append(j[1])
    Hits=[]
    max=0
    hr=0
    for i in user_rec:
        if i in user_item:
            NumOfHits = 0
            for j in user_rec[i]:
                if j in user_item[i]:
                    NumOfHits+=1
            if len(user_rec[i]) !=0:
                hr=NumOfHits/len(user_rec[i])
            if hr>=max:
                max=hr

    return max


def load_test_useritem(path):
    user_item={}
    item=[]
    with open(path,'r') as f:
        for line in f:
            tokens = line.strip().split()
            u = int(tokens[0])
            v = int(tokens[1])
            if u not in user_item:
                user_item[u]=[]
            user_item[u].append(v)
            if v not in item:
                item.append(v)
    return user_item,item



def compute_MAR(embeddings, pos_edges, neg_edges,user_list,item_list):
    MAR=0
    mar=0
    for u in user_list:
        MAR += mar
        score_list=[]
        u_enc = embeddings.get(u)
        if (u_enc is None):
            continue
        for v in item_list:
            v_enc = embeddings.get(v)
            if (v_enc is None):
                continue
            # score=base_compatibility(u_enc,v_enc,op='exp').view(-1).cpu().data.item()
            score=cos_sim(u_enc, v_enc)
            score_list.append(score)
        score_list = np.array(score_list)
        ranks = rankdata(score_list)
        mar=sum(ranks)/len(item_list)
    MAR=MAR/len(user_list)
    return MAR



def load_list(train_path,test_path):
    '''
    Returns a networkx graph.
    Edge property is called 'time'
    :param path: path the to dataset with header u, v, weight(u, v), time(u, v)
    :return: G
    '''
    user_list=[]
    item_list=[]
    with open(train_path,'r') as f:
        for line in f:
            tokens = line.strip().split()
            u = int(tokens[0])
            v = int(tokens[1])
            if u not in user_list:
                user_list.append(u)
            if v not in item_list:
                item_list.append(v)
    with open(test_path,'r') as f:
        for line in f:
            tokens = line.strip().split()
            u = int(tokens[0])
            v = int(tokens[1])
            if u not in user_list:
                user_list.append(u)
            if v not in item_list:
                item_list.append(v)
    return user_list,item_list

def base_compatibility(user, item,op):
    embed_user=torch.Tensor(user)
    embed_item=torch.Tensor(item)
    inn = torch.sum(embed_item * embed_user,dim=-1)
    if op == 'softplus':
        return F.softplus(inn)
    else:
        assert op == 'exp'
        return torch.exp(inn)



def main(user_list, item_list):
    embeddings_path = '../full_data/reddit_1000_random_0.7/save_dir/ia-contact.w2v.pkl'
    embeddings = load_node_embeddings(embeddings_path)

    edges_save_basepath = '../full_data/reddit_1000_random_0.7/save_dir/'
    pos_edges, neg_edges = load_edges(edges_save_basepath)

    X, y = get_dataset_from_embedding(embeddings, pos_edges, neg_edges)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = MLP(embeddings,cmd_args.embedding_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    output=model(pos_edges, neg_edges)

    logReg = LogisticRegression(solver='lbfgs')
    logReg.fit(X_train, y_train)

    y_pred = logReg.predict(X_test)
    y_score = logReg.predict_proba(X_test)
    print(y_score.shape)
    MAR=compute_MAR(embeddings, pos_edges, neg_edges,user_list,item_list)
    user_item,item=load_test_useritem(path=cmd_args.test_file)#测试集中每个用户实际交互的item
    user_rec=compute_HR(5,embeddings,user_list,item_list,user_item,item)

    print('Link prediction accuracy:', metrics.accuracy_score(y_true=y_test, y_pred=y_pred))
    print('Link prediction roc:', metrics.roc_auc_score(y_true=y_test, y_score=y_score[:, 1]))
    print('Link prediction MAR:', MAR)



if __name__ == '__main__':
    train_path = cmd_args.train_file
    test_path=cmd_args.test_file
    user_list, item_list=load_list(train_path,test_path)
    main(user_list, item_list)

#
# 1. x_train:包括所有自变量，这些变量将用于训练模型，同样，我们已经指定测试_size=0.4，这意味着来自完整数据的60%的观察值将用于训练/拟合模型，其余40%将用于测试模型。
# 2. y_train-这是因变量，需要此模型进行预测，其中包括针对自变量的类别标签，我们需要在训练/拟合模型时指定我们的因变量
# 3. x_test:这是数据中剩余的40%的自变量部分，这些自变量将不会在训练阶段使用，并将用于进行预测，以测试模型的准确性。
# 4. y_test-此数据具有测试数据的类别标签，这些标签将用于测试实际类别和预测类别之间的准确性。