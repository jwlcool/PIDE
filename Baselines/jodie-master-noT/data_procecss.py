
path="data/reddit_s.csv"
new_path="data/reddit_x.csv"
new=open(new_path,"a")
with open(path,"r") as f:
    f.readline()
    for cnt, l in enumerate(f):
        # FORMAT: user, item, timestamp, state label, feature list
        ls = l.strip().split(" ")
        user=ls[0]
        item=ls[1]
        timestamp=ls[2]
        str=user+","+item+","+timestamp+","+'0'+","+'0'+","+'0'
        new.write('\n')
        new.write(str)
    f.close()

# path=open("data/reddit_1000_random_0.7/reddit_s.csv")
# new_path= "data/reddit_s.csv"
# result=[]
# iter_f=iter(path)
# for line in iter_f:
#     result.append(line)
#
# path.close()
# result.sort(key=lambda x:float(x.split(' ')[2]),reverse=False)
# f=open(new_path,'w')
# f.writelines(result)
# f.close()

# path="data/IPTV.csv"
# event_list={}
# with open(path,"r") as f:
#     f.readline()
#     for cnt, l in enumerate(f):
#         # FORMAT: user, item, timestamp, state label, feature list
#         ls = l.strip().split(",")
#         user=ls[0]
#         item=ls[1]
#         timestamp=ls[2]
#         if user not in event_list:
#             event_list[user]=[]
#         if item not in event_list[user]:
#             event_list[user].append((item,timestamp))
#
#
#     f.close()