from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import copy
import math
from math import ceil

import numpy as np
import os
from scipy.stats import rankdata
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from coevolve.common.recorder import cur_time, dur_dist
from coevolve.common.consts import DEVICE, args
from coevolve.common.cmd_args import cmd_args
from coevolve.model.rayleigh_proc import ReyleighProc
from coevolve.common.neg_sampler import rand_sampler

from coevolve.common.pytorch_utils import SpEmbedding, weights_init
import networkx as nx
import random
from path import *
from coevolve.common.dataset import *


class DeepCoevolve(nn.Module):
    def __init__(self, train_data, test_data, num_users, num_items, embed_size, score_func, dt_type, max_norm):
        super(DeepCoevolve, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.max_norm = max_norm
        self.embed_size = embed_size
        self.score_func = score_func
        self.dt_type = dt_type
        self.user_embedding = SpEmbedding(num_users, embed_size)
        self.item_embedding = SpEmbedding(num_items, embed_size)
        self.user_lookup_embed = {}
        self.item_lookup_embed = {}
        self.delta_t = np.zeros((self.num_items,), dtype=np.float32)
        self.user_cell = nn.GRUCell(embed_size, embed_size)
        self.item_cell = nn.GRUCell(embed_size, embed_size)
        self.train_data = train_data
        self.test_data = test_data
        self.inter_linear = nn.Linear(embed_size, embed_size)
        self.neibor_linear = nn.Linear(embed_size, embed_size)
        self.train_user_list = train_data.user_list
        self.train_item_list = train_data.item_list
        self.test_user_list = test_data.user_list
        self.test_item_list = test_data.item_list
        self.user_list = merge_list(self.train_user_list, self.test_user_list)
        self.item_list = merge_list(self.train_item_list, self.test_item_list)
        self.kg = nx.Graph()
        self.user_ground={}
        self.node_count={}
        self.all_variables={}
        self.paths_between_pairs={}
        self.positive_label={}
        self.all_user={}
        self.all_movie={}
        self.u_linear = nn.Linear(cmd_args.embed_dim * 2, cmd_args.embed_dim, bias=True)
        self.i_linear = nn.Linear(cmd_args.embed_dim * 2, cmd_args.embed_dim, bias=True)
        self.lstm = nn.LSTM(cmd_args.embed_dim, cmd_args.embed_dim, 1, dropout=0.2)
        self.time_threshold=0
        self.neighbor_ration=0.5


    def normalize(self):
        if self.max_norm is None:
            return
        self.user_embedding.normalize(self.max_norm)
        self.item_embedding.normalize(self.max_norm)

    def _get_embedding(self, side, idx, lookup):
        if not idx in lookup:
            if side == 'user':
                lookup[idx] = self.user_embedding([idx])
            else:
                lookup[idx] = self.item_embedding([idx])
        return lookup[idx]

    def get_cur_user_embed(self, user):
        return self._get_embedding('user', user, self.user_lookup_embed)

    def get_cur_item_embed(self, item):
        return self._get_embedding('item', item, self.item_lookup_embed)

    def get_pred_score(self, comp, delta_t):
        if self.score_func == 'log_ll':
            d_t = np.clip(delta_t, a_min=1e-10, a_max=None)
            return np.log(d_t) + np.log(comp) - 0.5 * comp * (d_t ** 2)
        elif self.score_func == 'comp':
            return comp
        elif self.score_func == 'intensity':
            return comp * delta_t
        else:
            raise NotImplementedError

    def path_extract(self,paths_between_one_pair):
        sum_hidden = Variable(torch.Tensor(), requires_grad=True)
        paths_between_one_pair_id = paths_between_one_pair
        paths_size = len(paths_between_one_pair_id)
        for i in range(paths_size):
            path_embedding = []
            path = paths_between_one_pair_id[i]
            path_size = len(path)
            for id,j in enumerate(path):
                if id%2==0:
                    path_embedding.append(self.user_embedding([int(j[1:])]))
                if id%2!=0:
                    path_embedding.append(self.item_embedding([int(j[1:])]))
            path_emb = torch.cat((path_embedding[0],path_embedding[1]),0)
            path_emb = torch.cat((path_emb, path_embedding[2]), 0)
            path_emb = torch.cat((path_emb, path_embedding[3]), 0)
            path_embedding = path_emb.view(path_size, 1, cmd_args.embed_dim)
            if torch.cuda.is_available() and args.gpu >= 0:
                path_embedding = path_embedding.cuda()

            path_out, h = self.lstm(path_embedding)
            if i == 0:
                sum_hidden = h[0]
            else:
                sum_hidden = torch.cat((sum_hidden, h[0]), 0)

        pool = nn.MaxPool2d((1, 128))
        att = pool(sum_hidden)
        att = F.softmax(att, dim=0)
        path_extract = torch.mul(sum_hidden, att)  # 2,1,16
        path_emb = torch.sum(path_extract, 0, True)  # 1.1.16
        path_extract = path_emb.view(1, -1)
        return path_extract



    def get_output(self, cur_event, phase, kg, depth,user_rec_5,user_rec_10,user_rec_20):
        cur_user_embed = self.get_cur_user_embed(cur_event.user)
        cur_item_embed = self.get_cur_item_embed(cur_event.item)
        if cur_event.user not in self.user_ground:
            self.user_ground[cur_event.user]=[]
        if cur_event.item not in self.user_ground[cur_event.user]:
            self.user_ground[cur_event.user].append(cur_event.item)

        ur='u'+str(cur_event.user)
        im='i'+str(cur_event.item)
        neibor = get_neigbors(kg, ur, depth)[1]
        for i in neibor:
            if i != ur:
                self._get_embedding('item', int(i[1:]), self.item_lookup_embed)

        neibor = get_neigbors(kg, im, depth)[1]
        for j in neibor:
            if j != im:
                self._get_embedding('user', int(j[1:]), self.user_lookup_embed)


        t_end = cur_event.t
        base_comp = ReyleighProc.base_compatibility(cur_user_embed, cur_item_embed)#计算intensity function的过程

        dur = cur_event.t - cur_time.get_cur_time(cur_event.user, cur_event.item)
        dur_dist.add_time(dur)
        time_pred = ReyleighProc.time_mean(base_comp)

        mae = torch.abs(time_pred - dur)
        mse = (time_pred - dur) ** 2

        if phase == 'test':
            comp = ReyleighProc.base_compatibility(cur_user_embed, self.updated_item_embed).view(-1).cpu().data.numpy()
            for i in range(self.num_items):
                prev = cur_time.get_last_interact_time(cur_event.user,
                                                       i) if self.dt_type == 'last' else cur_time.get_cur_time(
                    cur_event.user, i)
                self.delta_t[i] = cur_event.t - prev
            scores = self.get_pred_score(comp, self.delta_t)
            ranks = rankdata(-scores)
            mar = ranks[cur_event.item]
            user_rec_5 = user_topK(5, user_rec_5, cur_event, ranks)
            user_rec_10 = user_topK(10, user_rec_10, cur_event, ranks)
            user_rec_20 = user_topK(20, user_rec_20, cur_event, ranks)
            return mar, mae, mse, user_rec_5, user_rec_10, user_rec_20

        neg_users = rand_sampler.sample_neg_users(cur_event.user, cur_event.item)
        neg_items = rand_sampler.sample_neg_items(cur_event.user, cur_event.item)

        neg_users_embeddings = self.user_embedding(neg_users)
        neg_items_embeddings = self.item_embedding(neg_items)
        for i, u in enumerate(neg_users):
            if u in self.user_lookup_embed:
                neg_users_embeddings[i] = self.user_lookup_embed[u]
        for j, i in enumerate(neg_items):
            if i in self.item_lookup_embed:
                neg_items_embeddings[j] = self.item_lookup_embed[i]
        #计算survive function值
        survival = ReyleighProc.survival(cur_event.user, cur_user_embed,
                                         cur_event.item, cur_item_embed,
                                         neg_users_embeddings, neg_users,
                                         neg_items_embeddings, neg_items,
                                         cur_event.t)
        loss = -torch.log(base_comp) + survival
        return loss, mae, mse


    def forward(self, T_begin, events, phase):
        user_rec_5 = {}
        user_rec_10 = {}
        user_rec_20 = {}
        self.kg.clear()
        self.time_matrix=dict()
        kg = create_kg(self.kg, self.time_matrix,T_begin, self.train_data.user_event_lists, self.train_data.item_event_lists)

        if phase == 'train':
            cur_time.reset(T_begin)
        self.user_lookup_embed = {}
        self.item_lookup_embed = {}

        with torch.set_grad_enabled(phase == 'train'):
            if phase == 'test':
                self.updated_user_embed = self.user_embedding.weight.clone()
                self.updated_item_embed = self.item_embedding.weight.clone()

            loss = 0.0
            mae = 0.0
            mse = 0.0
            pbar = enumerate(events)
            events_len=len(events)
            e_idx=0

            interaction_set = []
            while e_idx<events_len:
                num_interaction=1
                t_interaction=[]
                t_interaction.append(events[e_idx])

                # while events[e_idx].t==events[e_idx+num_interaction].t:
                #     t_interaction.append(events[e_idx+num_interaction])
                #     num_interaction=num_interaction+1
                #     if e_idx+num_interaction>=events_len:break

                e_idx=e_idx+num_interaction


                cur_event = t_interaction[0]
                kg_reduce = kg

                # kg_reduce = threshold(kg, self.time_threshold * cur_event.t)
                # kg = kg_reduce

                for i in t_interaction:
                    ur = 'u' + str(cur_event.user)
                    im = 'i' + str(cur_event.item)
                    interaction_set.extend([ur, im])
                    kg_add(kg,self.time_matrix, ur, im, i.t)
                kg = kg_reduce
                cur_event=t_interaction[0]
                assert cur_event.t >= T_begin
                if cur_event.user not in user_rec_5:
                    user_rec_5[cur_event.user]=[]
                if cur_event.user not in user_rec_10:
                    user_rec_10[cur_event.user]=[]
                if cur_event.user not in user_rec_20:
                    user_rec_20[cur_event.user]=[]



                kg_add(kg,self.time_matrix, ur, im, i.t)
                list_path = dump_paths(kg, ur, im, self.user_list,self.item_list,maxLen=3, sample_size=5)
                pos_list=[]
                for i in list_path:
                    if len(i)==4:
                        pos_list.append(i)

                if phase == 'test':
                    cur_loss, cur_mae, cur_mse,user_rec_5,user_rec_10,user_rec_20 = self.get_output(cur_event, phase,kg, 1,user_rec_5,user_rec_10,user_rec_20)
                if phase == 'train':
                    cur_loss, cur_mae, cur_mse = self.get_output(cur_event, phase,kg, 1, user_rec_5,user_rec_10,user_rec_20)
                # cur_loss, cur_mae, cur_mse = self.get_output(cur_event, phase, kg, 1)
                loss += cur_loss
                mae += cur_mae
                mse += cur_mse
                if e_idx + 1 == len(events):
                    break
                # coevolve的部分，更新交互节点
                # cur_user_embed = self.user_lookup_embed[cur_event.user]
                # cur_item_embed = self.item_lookup_embed[cur_event.item]
                # self.user_lookup_embed[cur_event.user] = self.user_cell(cur_item_embed, cur_user_embed)
                # self.item_lookup_embed[cur_event.item] = self.item_cell(cur_user_embed, cur_item_embed)

                cur_user_embed = self.user_lookup_embed[cur_event.user]
                cur_item_embed = self.item_lookup_embed[cur_event.item]
                if len(pos_list)!=0 :
                    path_extract = self.path_extract(pos_list)
                    cur_user_embed = self.u_linear(torch.cat((cur_user_embed, path_extract), -1) )
                    cur_item_embed = self.i_linear(torch.cat((cur_item_embed, path_extract), -1) )
                self.user_lookup_embed[cur_event.user] = self.user_cell(cur_item_embed, cur_user_embed)
                self.item_lookup_embed[cur_event.item] = self.item_cell(cur_user_embed, cur_item_embed)




                if phase == 'test':  # update embeddings into the embed mat
                    self.updated_user_embed[cur_event.user] = self.user_lookup_embed[cur_event.user]
                    self.updated_item_embed[cur_event.item] = self.item_lookup_embed[cur_event.item]
                cur_time.update_event(cur_event.user, cur_event.item, cur_event.t)

            rmse = torch.sqrt(mse / len(events)).item()
            mae = mae.item() / len(events)
            torch.set_grad_enabled(True)
            if phase == 'train':
                return loss, mae, rmse
            else:
                gt = self.user_ground
                HR_5 = computeHR(user_rec_5, gt)
                HR_10 = computeHR(user_rec_10, gt)
                HR_20 = computeHR(user_rec_20, gt)
                return loss / len(events), HR_5, HR_10, HR_20, mae, rmse

def del_list_element(list,element):
    diff_list=[]
    for item in list:
        if item not in element:
            diff_list.append(item)
    return diff_list


def user_topK(k, user_rec, cur_event, ranks):
    user = cur_event.user
    if len(user_rec[user]) >= k:
        user_rec[user].clear()
    for item_id, rank in enumerate(ranks):
        if rank < (k + 1):
            user_rec[user].append(item_id)
    return user_rec


def computeHR(user_rec, gt):
    hit = 0
    GT = 0
    for user in user_rec:
        GT += len(gt[user])
        for item in user_rec[user]:
            if item in gt[user]:
                hit += 1
    HR = hit / GT
    return HR

def computeMaxHR(user_rec, gt):
    GT = 0
    hit = 0
    HR=0
    for user in user_rec:
        hit = 0
        GT = len(gt[user])
        for item in user_rec[user]:
            if item in gt[user]:
                hit += 1
        hr = hit / GT
        if hr > HR:HR=hr
    return HR