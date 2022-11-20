

train_path="full_data/yelp/train.txt"
test_path="full_data/yelp/train.txt"
mata_path="full_data/yelp/meta.txt"
user_list=[]
item_list=[]
with open(train_path,"r") as f:
    f.readline()
    for cnt, l in enumerate(f):
        # FORMAT: user, item, timestamp, state label, feature list
        ls = l.strip().split(" ")
        user=ls[0]
        item=ls[1]
        timestamp=ls[2]
        if user not in user_list:user_list.append(user)
        if item not in item_list: item_list.append(item)

    f.close()