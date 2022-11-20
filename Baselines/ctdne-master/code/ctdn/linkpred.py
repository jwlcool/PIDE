import pickle
import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from cmg import cmd_args,cmd_opt
import torch.nn.functional as F
from sklearn.model_selection import ShuffleSplit
from measure import *


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


def get_dataset_from_embedding(embeddings, pos_edges, neg_edges, op):
    '''
    op can take values from 'mean', 'l1', 'l2', 'hadamard'
    '''
    y = []
    X = []
    pair= {}
    count=0
    # process positive links
    for u, v, prop in pos_edges:
        # get node representation and average them
        # X[(u,v)]=[]
        pair[count]=[]
        u_enc = embeddings.get(u)
        v_enc = embeddings.get(v)

        if (u_enc is None) or (v_enc is None):
            continue

        datapoint = operator(u_enc, v_enc, op=op)  # (u_enc + v_enc)/2.0

        # X[(u,v)].append(datapoint)
        X.append(datapoint)
        y.append(0.0)
        pair[count].append([u,v])
        count+=1

    # process negative links
    for u, v, prop in neg_edges:
        # get node representation and average them
        u_enc = embeddings.get(u)
        v_enc = embeddings.get(v)
        # X[(u, v)] = []
        pair[count] = []
        if (u_enc is None) and (v_enc is not None):
            u_enc = v_enc
        if (v_enc is None) and (u_enc is not None):
            v_enc = u_enc
        if (u_enc is None) and (v_enc is None):
            continue

        datapoint = operator(u_enc, v_enc, op=op)  # (u_enc + v_enc) / 2.0

        # X[(u,v)].append(datapoint)
        X.append(datapoint)
        y.append(1.0)
        pair[count].append([u,v])
        count+=1
    dataset = np.array(X), np.array(y),pair
    return dataset



def compute_HR(k,X,y,y_pred,y_score,train_index,test_index,logReg,pair,user_list, item_list):

    hits=0
    count=0
    score={}
    user_test={}
    user_rec={}
    for i in range(len(y_pred)):
        score[i] = []
        index=test_index[i]
        pr=pair[index]
        score[i].append(pr)
        score[i].append(y_score[i][0])  # 预测标签概率,>0.5为正样本
        score[i].append(y_pred[i])#预测标签， 0为正样本，1为负样本
        score[i].append(y[index])#实际标签， 0为正样本，1为负样本

    score_rank=sorted(score.items(),key=lambda x:x[1][1],reverse=True)
    for j in test_index:
        user=pair[j][0][0]
        item=pair[j][0][1]
        if user not in user_test:
            user_test[user]=[]
        if item not in user_test[user]:
            user_test[user].append(item)


    for i in score_rank:
        user=i[1][0][0][0]
        item=i[1][0][0][1]
        if user not in user_rec:
            user_rec[user] = []
        if i[1][2]==0:#预测为正样本，即进行推荐
            if len(user_rec[user])>=k:
                user_rec[user].clear()
            if len(user_rec[user])<k:
                if item not in user_rec[user]:
                    user_rec[user].append(item)#推荐的前k个item
    max=0
    hr=0
    Hit=0
    Gt=0
    users=[]
    # for j in test_index:
    #     user=pair[j][0][0]
    #     if user in users:
    #         continue
    #     else: users.append(user)
    #     if len(user_test[user]) == 0:
    #         hit=0
    #         gt = 0
    #         Hit += hit
    #         Gt += gt
    #     else:
    #         if len(user_rec[user]) !=0:
    #             a = [x for x in user_rec[user] if x in user_test[user]]
    #             hit=len(a)
    #             gt=len(user_test[user])
    #             Hit+=hit
    #             Gt+=gt
    #         if len(user_rec[user]) ==0:
    #             hit=0
    #             gt=len(user_test[user])
    #             Hit+=hit
    #             Gt+=gt
    # hr=Hit/Gt

    for u in user_list:
        if u in users:
            continue
        else:
            users.append(u)
        if u not in user_test and u not in user_rec:
            continue
        if u in user_test and u not in user_rec:
            hit=0
            gt = len(user_test[u])
            Hit += hit
            Gt += gt
        if u not in user_test and u in user_rec:
            hit=0
            gt=0
            Hit += hit
            Gt += gt
        if u in user_test and u in user_rec:
            a = [x for x in user_rec[u] if x in user_test[u]]
            hit = len(a)
            gt = len(user_test[u])
            Hit += hit
            Gt += gt

    hr=Hit/(Gt+Hit)
    return hr




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



def load_list(train_path):
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
    return user_list,item_list

def base_compatibility(user, item,op):
    embed_user=torch.Tensor(user)
    embed_item=torch.Tensor(item)
    inn = torch.sum(embed_item * embed_user, dim=-1)
    if op == 'softplus':
        return F.softplus(inn)
    if op == 'exp':
        return torch.exp(inn)
    if op == 'sigmoid':
        return torch.sigmoid(inn)



def main(user_list, item_list):
    sss = ShuffleSplit(1,test_size=cmd_args.test_size,random_state=10)
    embeddings_path = cmd_args.save_dir+'ia-contact.w2v.pkl'
    embeddings = load_node_embeddings(embeddings_path)

    edges_save_basepath = cmd_args.save_dir
    pos_edges, neg_edges = load_edges(edges_save_basepath)

    pos_edges = sorted(pos_edges, key=lambda x: float(x[2]['time']), reverse=False)
    X, y, pair= get_dataset_from_embedding(embeddings, pos_edges, neg_edges,op='l2')

    sss.get_n_splits(X, y)
    train_index, test_index = next(sss.split(X, y))

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cmd_args.test_size,shuffle=False)



    logReg = LogisticRegression(solver='lbfgs')
    logReg.fit(X_train, y_train)
    #y_test实际的标签
    y_pred = logReg.predict(X_test)#预测的标签
    y_score = logReg.predict_proba(X_test)#预测的标签概率
    print(y_score.shape)
    MAR=compute_MAR(embeddings, pos_edges, neg_edges,user_list,item_list)
    # user_item,item=load_test_useritem(path=cmd_args.test_file)#测试集中每个用户实际交互的item
    HR_5=compute_HR(5,X,y,y_pred,y_score,train_index,test_index,logReg,pair,user_list, item_list)
    HR_10 = compute_HR(10, X, y, y_pred, y_score, train_index, test_index, logReg, pair,user_list, item_list)
    HR_20 = compute_HR(20, X, y, y_pred, y_score, train_index, test_index, logReg, pair,user_list, item_list)

    # NDGC_5=ndcg_score(y_test, y_score, k=5)

    print('Link prediction accuracy:', metrics.accuracy_score(y_true=y_test, y_pred=y_pred))
    # print('Link prediction roc:', metrics.roc_auc_score(y_true=y_test, y_score=y_score[:, 1]))
    print('Link prediction MAR:', MAR)
    print('Link prediction HR@5:',HR_5)
    print('Link prediction HR@10:', HR_10)
    print('Link prediction HR@20:', HR_20)


if __name__ == '__main__':
    file_path = cmd_args.train_file
    user_list, item_list=load_list(file_path)
    main(user_list, item_list)

#
# 1. x_train:包括所有自变量，这些变量将用于训练模型，同样，我们已经指定测试_size=0.4，这意味着来自完整数据的60%的观察值将用于训练/拟合模型，其余40%将用于测试模型。
# 2. y_train-这是因变量，需要此模型进行预测，其中包括针对自变量的类别标签，我们需要在训练/拟合模型时指定我们的因变量
# 3. x_test:这是数据中剩余的40%的自变量部分，这些自变量将不会在训练阶段使用，并将用于进行预测，以测试模型的准确性。
# 4. y_test-此数据具有测试数据的类别标签，这些标签将用于测试实际类别和预测类别之间的准确性。