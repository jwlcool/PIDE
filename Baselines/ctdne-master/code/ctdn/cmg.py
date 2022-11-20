import argparse
import os

# #reddit
# cmd_opt = argparse.ArgumentParser(description='Argparser for coevolve')
# cmd_opt.add_argument('-meta_file', default='../full_data/reddit_1000_random_0.7/meta.txt', help='meta_file')
# cmd_opt.add_argument('-train_file', default='../full_data/reddit_1000_random_0.7/train.txt', help='train_file')
# cmd_opt.add_argument('-test_file', default='../full_data/reddit_1000_random_0.7/test.txt', help='test_file')
# cmd_opt.add_argument('-train_edge',default=0.75, help='train_edge')
# cmd_opt.add_argument('-test_size',default=0.25, help='test_size')
# cmd_opt.add_argument('-embedding_size',default=128, help='embedding_size')
# cmd_opt.add_argument('-data_dir', default='../full_data/reddit_1000_random_0.7/', help='data_dir')
# cmd_opt.add_argument('-save_dir', default='../full_data/reddit_1000_random_0.7/save_dir/', help='data_dir')
# cmd_opt.add_argument('-walk_path', default='../full_data/reddit_1000_random_0.7/save_dir/ia-contact.time.walks', help='data_dir')
# cmd_args, _ = cmd_opt.parse_known_args()

#IPTV
# cmd_opt = argparse.ArgumentParser(description='Argparser for coevolve')
# cmd_opt.add_argument('-meta_file', default='../full_data/IPTV_0.7/back/meta.txt', help='meta_file')
# cmd_opt.add_argument('-train_file', default='../full_data/IPTV_0.7/back/data0.3.txt', help='train_file')
# cmd_opt.add_argument('-test_file', default='../full_data/IPTV_0.7/back/test0.3.txt', help='test_file')
# cmd_opt.add_argument('-train_edge',default=0.75, help='train_edge')
# cmd_opt.add_argument('-test_size',default=0.6666666, help='test_size')
# cmd_opt.add_argument('-embedding_size',default=64, help='embedding_size')
# cmd_opt.add_argument('-data_dir', default='../full_data/IPTV_0.7/', help='data_dir')
# cmd_opt.add_argument('-save_dir', default='../full_data/IPTV_0.7/save_dir/', help='data_dir')
# cmd_opt.add_argument('-walk_path', default='../full_data/IPTV_0.7/save_dir/ia-contact.time.walks', help='data_dir')
# cmd_args, _ = cmd_opt.parse_known_args()

#Yelp
cmd_opt = argparse.ArgumentParser(description='Argparser for coevolve')
# cmd_opt.add_argument('-meta_file', default='../full_data/yelp_0.7/meta.txt', help='meta_file')
cmd_opt.add_argument('-train_file', default='../full_data/yelp_0.7/train.txt', help='train_file')
# cmd_opt.add_argument('-test_file', default='../full_data/yelp_0.7/yelp.txt', help='test_file')
cmd_opt.add_argument('-train_edge',default=0.7, help='train_edge')
cmd_opt.add_argument('-test_size',default=0.4, help='test_size')
cmd_opt.add_argument('-embedding_size',default=128, help='embedding_size')
cmd_opt.add_argument('-data_dir', default='../full_data/yelp_0.7/', help='data_dir')
cmd_opt.add_argument('-save_dir', default='../full_data/yelp_0.7/save_dir/', help='data_dir')
cmd_opt.add_argument('-walk_path', default='../full_data/yelp_0.7/save_dir/ia-contact.time.walks', help='data_dir')
cmd_opt.add_argument('-int_act', default='exp', help='activation function for intensity', choices=['exp', 'softplus'])
cmd_args, _ = cmd_opt.parse_known_args()



# #reddit
# cmd_opt = argparse.ArgumentParser(description='Argparser for coevolve')
# cmd_opt.add_argument('-meta_file', default='../full_data/reddit_1000_random_0.7/meta.txt', help='meta_file')
# cmd_opt.add_argument('-train_file', default='../full_data/reddit_1000_random_0.7/train.txt', help='train_file')
# cmd_opt.add_argument('-test_file', default='../full_data/reddit_1000_random_0.7/test.txt', help='test_file')
# cmd_opt.add_argument('-train_edge',default=0.75, help='train_edge')
# cmd_opt.add_argument('-test_size',default=0.25, help='test_size')
# cmd_opt.add_argument('-embedding_size',default=128, help='embedding_size')
# cmd_opt.add_argument('-data_dir', default='../full_data/reddit_1000_random_0.7/', help='data_dir')
# cmd_opt.add_argument('-save_dir', default='../full_data/reddit_1000_random_0.7/save_dir/', help='data_dir')
# cmd_opt.add_argument('-walk_path', default='../full_data/reddit_1000_random_0.7/save_dir/ia-contact.time.walks', help='data_dir')
# cmd_args, _ = cmd_opt.parse_known_args()