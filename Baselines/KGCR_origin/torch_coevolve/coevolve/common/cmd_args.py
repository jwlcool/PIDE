from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import argparse
import os

# dataset='music'
# dataset='Taobao'
dataset='movie'

###LastFm
if dataset=='music':
    cmd_opt = argparse.ArgumentParser(description='Argparser for coevolve')
    cmd_opt.add_argument('-dataset', default='LastFm', help='Reddit')
    cmd_opt.add_argument('-save_dir', default='.', help='result output root')
    cmd_opt.add_argument('-dropbox', default=None, help='dropbox folder')
    cmd_opt.add_argument('-init_model_dump', default=None, help='model dump')
    cmd_opt.add_argument('-data_name', default=None, help='dataset name')
    cmd_opt.add_argument('-phase', default=None, help='phase')
    cmd_opt.add_argument('-dt_type', default='last', help='last/cur')
    cmd_opt.add_argument('-int_act', default='exp', help='activation function for intensity', choices=['exp', 'softplus'])
    cmd_opt.add_argument('-score_func', default='log_ll', help='log_ll/comp/intensity')
    cmd_opt.add_argument('-is_training', default=True, type=eval, help='is training')
    cmd_opt.add_argument('-meta_file', default='../data/music/meta.txt', help='meta_file')
    cmd_opt.add_argument('-train_file', default='../data/music/train0.6.txt', help='train_file')
    cmd_opt.add_argument('-test_file', default='../data/music/test0.6.txt', help='test_file')
    cmd_opt.add_argument('-embed_dim', default=128, type=int, help='embedding dim of gnn')
    cmd_opt.add_argument('-bptt', default=100, type=int, help='bptt size')
    cmd_opt.add_argument('-num_items', default=0, type=int, help='num items')
    cmd_opt.add_argument('-num_users', default=0, type=int, help='num users')
    cmd_opt.add_argument('-neg_items', default=100, type=int, help='neg items')
    cmd_opt.add_argument('-neg_users', default=100, type=int, help='neg users')
    cmd_opt.add_argument('-max_norm', default=0.1, type=float, help='max embed norm')
    cmd_opt.add_argument('-time_scale', default=0.001, type=float, help='time scale')
    cmd_opt.add_argument('-time_lb', default=0.01, type=float, help='min time dur')
    cmd_opt.add_argument('-time_ub', default=0.1, type=float, help='max time dur')
    cmd_opt.add_argument('-seed', default=19260817, type=int, help='seed')
    cmd_opt.add_argument('-learning_rate', default=0.001, type=float, help='learning rate')
    cmd_opt.add_argument('-grad_clip', default=5, type=float, help='clip gradient')
    cmd_opt.add_argument('-num_epochs', default=1000, type=int, help='number of training epochs')
    cmd_opt.add_argument('-iters_per_val', default=100, type=int, help='number of iterations per evaluation')
    cmd_opt.add_argument('-batch_size', default=64, type=int, help='batch size for training')
    cmd_args, _ = cmd_opt.parse_known_args()


###Movie-len
if dataset=='movie':
    cmd_opt = argparse.ArgumentParser(description='Argparser for coevolve')
    cmd_opt.add_argument('-dataset', default='Movie', help='Reddit')
    cmd_opt.add_argument('-save_dir', default='.', help='result output root')
    cmd_opt.add_argument('-dropbox', default=None, help='dropbox folder')
    cmd_opt.add_argument('-init_model_dump', default=None, help='model dump')
    cmd_opt.add_argument('-data_name', default=None, help='dataset name')
    cmd_opt.add_argument('-phase', default=None, help='phase')
    cmd_opt.add_argument('-dt_type', default='last', help='last/cur')
    cmd_opt.add_argument('-int_act', default='exp', help='activation function for intensity', choices=['exp', 'softplus'])
    cmd_opt.add_argument('-score_func', default='log_ll', help='log_ll/comp/intensity')
    cmd_opt.add_argument('-is_training', default=True, type=eval, help='is training')
    cmd_opt.add_argument('-meta_file', default='../data/movie-1m/meta.txt', help='meta_file')
    cmd_opt.add_argument('-train_file', default='../data/movie-1m/train0.1.txt', help='train_file')
    cmd_opt.add_argument('-test_file', default='../data/movie-1m/test0.1.txt', help='test_file')
    cmd_opt.add_argument('-embed_dim', default=128, type=int, help='embedding dim of gnn')
    cmd_opt.add_argument('-bptt', default=100, type=int, help='bptt size')
    cmd_opt.add_argument('-num_items', default=0, type=int, help='num items')
    cmd_opt.add_argument('-num_users', default=0, type=int, help='num users')
    cmd_opt.add_argument('-neg_items', default=100, type=int, help='neg items')
    cmd_opt.add_argument('-neg_users', default=100, type=int, help='neg users')
    cmd_opt.add_argument('-max_norm', default=0.1, type=float, help='max embed norm')
    cmd_opt.add_argument('-time_scale', default=0.001, type=float, help='time scale')
    cmd_opt.add_argument('-time_lb', default=0.01, type=float, help='min time dur')
    cmd_opt.add_argument('-time_ub', default=0.1, type=float, help='max time dur')
    cmd_opt.add_argument('-seed', default=19260817, type=int, help='seed')
    cmd_opt.add_argument('-learning_rate', default=0.001, type=float, help='learning rate')
    cmd_opt.add_argument('-grad_clip', default=5, type=float, help='clip gradient')
    cmd_opt.add_argument('-num_epochs', default=1000, type=int, help='number of training epochs')
    cmd_opt.add_argument('-iters_per_val', default=100, type=int, help='number of iterations per evaluation')
    cmd_opt.add_argument('-batch_size', default=64, type=int, help='batch size for training')
    cmd_args, _ = cmd_opt.parse_known_args()

##TaoBao
if dataset=='Taobao':
    cmd_opt = argparse.ArgumentParser(description='Argparser for coevolve')
    cmd_opt.add_argument('-dataset', default='Taobao', help='Reddit')
    cmd_opt.add_argument('-save_dir', default='.', help='result output root')
    cmd_opt.add_argument('-dropbox', default=None, help='dropbox folder')
    cmd_opt.add_argument('-init_model_dump', default=None, help='model dump')
    cmd_opt.add_argument('-data_name', default=None, help='dataset name')
    cmd_opt.add_argument('-phase', default=None, help='phase')
    cmd_opt.add_argument('-dt_type', default='last', help='last/cur')
    cmd_opt.add_argument('-int_act', default='exp', help='activation function for intensity', choices=['exp', 'softplus'])
    cmd_opt.add_argument('-score_func', default='log_ll', help='log_ll/comp/intensity')
    cmd_opt.add_argument('-is_training', default=True, type=eval, help='is training')
    cmd_opt.add_argument('-meta_file', default='../data/Taobao/meta.txt', help='meta_file')
    cmd_opt.add_argument('-train_file', default='../data/Taobao/train0.7.txt', help='train_file')
    cmd_opt.add_argument('-test_file', default='../data/Taobao/test0.7.txt', help='test_file')
    cmd_opt.add_argument('-embed_dim', default=128, type=int, help='embedding dim of gnn')
    cmd_opt.add_argument('-bptt', default=100, type=int, help='bptt size')
    cmd_opt.add_argument('-num_items', default=0, type=int, help='num items')
    cmd_opt.add_argument('-num_users', default=0, type=int, help='num users')
    cmd_opt.add_argument('-neg_items', default=100, type=int, help='neg items')
    cmd_opt.add_argument('-neg_users', default=100, type=int, help='neg users')
    cmd_opt.add_argument('-max_norm', default=0.1, type=float, help='max embed norm')
    cmd_opt.add_argument('-time_scale', default=0.001, type=float, help='time scale')
    cmd_opt.add_argument('-time_lb', default=0.01, type=float, help='min time dur')
    cmd_opt.add_argument('-time_ub', default=0.1, type=float, help='max time dur')
    cmd_opt.add_argument('-seed', default=19260817, type=int, help='seed')
    cmd_opt.add_argument('-learning_rate', default=0.001, type=float, help='learning rate')
    cmd_opt.add_argument('-grad_clip', default=5, type=float, help='clip gradient')
    cmd_opt.add_argument('-num_epochs', default=1000, type=int, help='number of training epochs')
    cmd_opt.add_argument('-iters_per_val', default=100, type=int, help='number of iterations per evaluation')
    cmd_opt.add_argument('-batch_size', default=64, type=int, help='batch size for training')
    cmd_args, _ = cmd_opt.parse_known_args()

# #reddit
# cmd_opt = argparse.ArgumentParser(description='Argparser for coevolve')
# cmd_opt.add_argument('-dataset', default='Reddit', help='Reddit')
# cmd_opt.add_argument('-save_dir', default='.', help='result output root')
# cmd_opt.add_argument('-dropbox', default=None, help='dropbox folder')
# cmd_opt.add_argument('-init_model_dump', default=None, help='model dump')
# cmd_opt.add_argument('-data_name', default=None, help='dataset name')
# cmd_opt.add_argument('-phase', default=None, help='phase')
# cmd_opt.add_argument('-dt_type', default='last', help='last/cur')
# cmd_opt.add_argument('-int_act', default='exp', help='activation function for intensity', choices=['exp', 'softplus'])
# cmd_opt.add_argument('-score_func', default='log_ll', help='log_ll/comp/intensity')
# cmd_opt.add_argument('-is_training', default=True, type=eval, help='is training')
# cmd_opt.add_argument('-meta_file', default='full_data/reddit_1000_random_0.7/meta.txt', help='meta_file')
# cmd_opt.add_argument('-train_file', default='full_data/reddit_1000_random_0.7/train.txt', help='train_file')
# cmd_opt.add_argument('-test_file', default='full_data/reddit_1000_random_0.7/test.txt', help='test_file')
# cmd_opt.add_argument('-embed_dim', default=128, type=int, help='embedding dim of gnn')
# cmd_opt.add_argument('-bptt', default=100, type=int, help='bptt size')
# cmd_opt.add_argument('-num_items', default=0, type=int, help='num items')
# cmd_opt.add_argument('-num_users', default=0, type=int, help='num users')
# cmd_opt.add_argument('-neg_items', default=100, type=int, help='neg items')
# cmd_opt.add_argument('-neg_users', default=100, type=int, help='neg users')
# cmd_opt.add_argument('-max_norm', default=0.1, type=float, help='max embed norm')
# cmd_opt.add_argument('-time_scale', default=0.001, type=float, help='time scale')
# cmd_opt.add_argument('-time_lb', default=0.01, type=float, help='min time dur')
# cmd_opt.add_argument('-time_ub', default=0.1, type=float, help='max time dur')
# cmd_opt.add_argument('-seed', default=19260817, type=int, help='seed')
# cmd_opt.add_argument('-learning_rate', default=0.001, type=float, help='learning rate')
# cmd_opt.add_argument('-grad_clip', default=5, type=float, help='clip gradient')
# cmd_opt.add_argument('-num_epochs', default=1000, type=int, help='number of training epochs')
# cmd_opt.add_argument('-iters_per_val', default=100, type=int, help='number of iterations per evaluation')
# cmd_opt.add_argument('-batch_size', default=64, type=int, help='batch size for training')
# cmd_args, _ = cmd_opt.parse_known_args()

#IPTV
# cmd_opt = argparse.ArgumentParser(description='Argparser for coevolve')
# cmd_opt.add_argument('-dataset', default='IPTV', help='IPTV')
# cmd_opt.add_argument('-save_dir', default='.', help='result output root')
# cmd_opt.add_argument('-dropbox', default=None, help='dropbox folder')
# cmd_opt.add_argument('-init_model_dump', default=None, help='model dump')
# cmd_opt.add_argument('-data_name', default=None, help='dataset name')
# cmd_opt.add_argument('-phase', default=None, help='phase')
# cmd_opt.add_argument('-dt_type', default='last', help='last/cur')
# cmd_opt.add_argument('-int_act', default='exp', help='activation function for intensity', choices=['exp', 'softplus'])
# cmd_opt.add_argument('-score_func', default='log_ll', help='log_ll/comp/intensity')
# cmd_opt.add_argument('-is_training', default=True, type=eval, help='is training')
# cmd_opt.add_argument('-meta_file', default='full_data/IPTV_0.7/meta.txt', help='meta_file')
# cmd_opt.add_argument('-train_file', default='full_data/IPTV_0.7/train.txt', help='train_file')
# cmd_opt.add_argument('-test_file', default='full_data/IPTV_0.7/test.txt', help='test_file')
# cmd_opt.add_argument('-embed_dim', default=128, type=int, help='embedding dim of gnn')
# cmd_opt.add_argument('-bptt', default=100, type=int, help='bptt size')
# cmd_opt.add_argument('-num_items', default=0, type=int, help='num items')
# cmd_opt.add_argument('-num_users', default=0, type=int, help='num users')
# cmd_opt.add_argument('-neg_items', default=100, type=int, help='neg items')
# cmd_opt.add_argument('-neg_users', default=100, type=int, help='neg users')
# cmd_opt.add_argument('-max_norm', default=0.1, type=float, help='max embed norm')
# cmd_opt.add_argument('-time_scale', default=0.001, type=float, help='time scale')
# cmd_opt.add_argument('-time_lb', default=0.01, type=float, help='min time dur')
# cmd_opt.add_argument('-time_ub', default=0.1, type=float, help='max time dur')
# cmd_opt.add_argument('-seed', default=19260817, type=int, help='seed')
# cmd_opt.add_argument('-learning_rate', default=0.001, type=float, help='learning rate')
# cmd_opt.add_argument('-grad_clip', default=5, type=float, help='clip gradient')
# cmd_opt.add_argument('-num_epochs', default=10000, type=int, help='number of training epochs')
# cmd_opt.add_argument('-iters_per_val', default=100, type=int, help='number of iterations per evaluation')
# cmd_opt.add_argument('-batch_size', default=64, type=int, help='batch size for training')
# cmd_args, _ = cmd_opt.parse_known_args()

#yelp
# cmd_opt = argparse.ArgumentParser(description='Argparser for coevolve')
# cmd_opt.add_argument('-dataset', default='Yelp', help='Yelp')
# cmd_opt.add_argument('-save_dir', default='.', help='result output root')
# cmd_opt.add_argument('-dropbox', default=None, help='dropbox folder')
# cmd_opt.add_argument('-init_model_dump', default=None, help='model dump')
# cmd_opt.add_argument('-data_name', default=None, help='dataset name')
# cmd_opt.add_argument('-phase', default=None, help='phase')
# cmd_opt.add_argument('-dt_type', default='cur', help='last/cur')
# cmd_opt.add_argument('-int_act', default='softplus', help='activation function for intensity', choices=['exp', 'softplus'])
# cmd_opt.add_argument('-score_func', default='comp', help='log_ll/comp/intensity')
# cmd_opt.add_argument('-is_training', default=True, type=eval, help='is training')
# cmd_opt.add_argument('-meta_file', default='full_data/yelp_0.7/meta.txt', help='meta_file')
# cmd_opt.add_argument('-train_file', default='full_data/yelp_0.7/train.txt', help='train_file')
# cmd_opt.add_argument('-test_file', default='full_data/yelp_0.7/test.txt', help='test_file')
# cmd_opt.add_argument('-embed_dim', default=128, type=int, help='embedding dim of gnn')
# cmd_opt.add_argument('-bptt', default=100, type=int, help='bptt size')
# cmd_opt.add_argument('-num_items', default=0, type=int, help='num items')
# cmd_opt.add_argument('-num_users', default=0, type=int, help='num users')
# cmd_opt.add_argument('-neg_items', default=100, type=int, help='neg items')
# cmd_opt.add_argument('-neg_users', default=100, type=int, help='neg users')
# cmd_opt.add_argument('-max_norm', default=-1, type=float, help='max embed norm')
# cmd_opt.add_argument('-time_scale', default=0.001, type=float, help='time scale')
# cmd_opt.add_argument('-time_lb', default=0.1, type=float, help='min time dur')
# cmd_opt.add_argument('-time_ub', default=0.1, type=float, help='max time dur')
# cmd_opt.add_argument('-seed', default=19260817, type=int, help='seed')
# cmd_opt.add_argument('-learning_rate', default=0.001, type=float, help='learning rate')
# cmd_opt.add_argument('-grad_clip', default=5, type=float, help='clip gradient')
# cmd_opt.add_argument('-num_epochs', default=10000, type=int, help='number of training epochs')
# cmd_opt.add_argument('-iters_per_val', default=500, type=int, help='number of iterations per evaluation')
# cmd_opt.add_argument('-batch_size', default=64, type=int, help='batch size for training')
# cmd_args, _ = cmd_opt.parse_known_args()


if cmd_args.save_dir is not None:
    if not os.path.isdir(cmd_args.save_dir):
        os.makedirs(cmd_args.save_dir)

assert cmd_args.meta_file is not None
with open(cmd_args.meta_file, 'r') as f:
    row = f.readline()
    row = [int(t) for t in row.split()[:2]]
    cmd_args.num_users, cmd_args.num_items = row

print(cmd_args)

