'''
This code trains the JODIE model for the given dataset. 
The task is: interaction prediction.

How to run: 
$
python jodie.py --network reddit --model jodie --epochs 50
python jodie.py --network lastfm --model jodie --epochs 50

Paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019. 
'''

import time

import networkx as nx
from tqdm.notebook import trange
from library_data import *
import library_models as lib
from library_models import *

# INITIALIZE PARAMETERS
from propagate import get_neigbors, kg_add

parser = argparse.ArgumentParser()
parser.add_argument('--network', required=True, help='Name of the network/dataset')
parser.add_argument('--model', default="jodie", help='Model name to save output in file')
parser.add_argument('--gpu', default=3, type=int, help='ID of the gpu to run on. If set to -1 (default), the GPU with most free memory will be chosen.')
parser.add_argument('--epochs', default=50, type=int, help='Number of epochs to train the model')
parser.add_argument('--embedding_dim', default=128, type=int, help='Number of dimensions of the dynamic embedding')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Fraction of interactions (from the beginning) that are used for training.The next 10% are used for validation and the next 10% for testing')
parser.add_argument('--state_change', default=True, type=bool, help='True if training with state change of users along with interaction prediction. False otherwise. By default, set to True.')
parser.add_argument('--threshold', default=0.1, type=float, help='threshold')
args = parser.parse_args()

args.datapath = "data/%s.csv" % args.network
# args.datapath = "data/reddit_1000_random_0.7/train.txt"




if args.train_proportion > 0.8:
    sys.exit('Training sequence proportion cannot be greater than 0.8.')

# SET GPU
# if args.gpu == -1:
#     args.gpu = select_free_gpu()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# LOAD DATA
[user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
 item2id, item_sequence_id, item_timediffs_sequence, 
 timestamp_sequence, feature_sequence, y_true] = load_network(args)
num_interactions = len(user_sequence_id)
num_users = len(user2id) 
num_items = len(item2id) + 1 # one extra item for "none-of-these"
num_features = len(feature_sequence[0])
true_labels_ratio = len(y_true)/(1.0+sum(y_true)) # +1 in denominator in case there are no state change labels, which will throw an error. 
print("*** Network statistics:\n  %d users\n  %d items\n  %d interactions\n  %d/%d true labels ***\n\n" % (num_users, num_items, num_interactions, sum(y_true), len(y_true)))

# SET TRAINING, VALIDATION, TESTING, and TBATCH BOUNDARIES
train_end_idx = validation_start_idx = int(num_interactions * args.train_proportion) 
test_start_idx = int(num_interactions * (args.train_proportion+0.1))
test_end_idx = int(num_interactions * (args.train_proportion+0.2))

# SET BATCHING TIMESPAN
'''
Timespan is the frequency at which the batches are created and the JODIE model is trained. 
As the data arrives in a temporal order, the interactions within a timespan are added into batches (using the T-batch algorithm). 
The batches are then used to train JODIE. 
Longer timespans mean more interactions are processed and the training time is reduced, however it requires more GPU memory.
Longer timespan leads to less frequent model updates. 
'''
timespan = timestamp_sequence[-1] - timestamp_sequence[0]
tbatch_timespan = timespan /500

kg=nx.Graph()
for iter in range(num_interactions):
    userid = user_sequence_id[iter]
    itemid = item_sequence_id[iter]
    user='u'+str(userid)
    item='i'+str(itemid)
    kg.add_node(user)
    kg.add_node(item)
    kg.add_edge(user,item)
    kg.add_edge(item,user)
    kg.add_edge(user,item, time=timestamp_sequence[iter])



# tbatch_timespan = 0.01
# INITIALIZE MODEL AND PARAMETERS
model = JODIE(args, num_features, num_users, num_items).cuda()
weight = torch.Tensor([1,true_labels_ratio]).cuda()
crossEntropyLoss = nn.CrossEntropyLoss(weight=weight)
MSELoss = nn.MSELoss()

# INITIALIZE EMBEDDING
initial_user_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0)) # the initial user and item embeddings are learned during training as well
initial_item_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0))
model.initial_user_embedding = initial_user_embedding
model.initial_item_embedding = initial_item_embedding

user_embeddings = initial_user_embedding.repeat(num_users, 1) # initialize all users to the same embedding 
item_embeddings = initial_item_embedding.repeat(num_items, 1) # initialize all items to the same embedding
item_embedding_static = Variable(torch.eye(num_items).cuda()) # one-hot vectors for static embeddings
user_embedding_static = Variable(torch.eye(num_users).cuda()) # one-hot vectors for static embeddings 

# INITIALIZE MODEL
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# RUN THE JODIE MODEL
'''
THE MODEL IS TRAINED FOR SEVERAL EPOCHS. IN EACH EPOCH, JODIES USES THE TRAINING SET OF INTERACTIONS TO UPDATE ITS PARAMETERS.
'''
print("*** Training the JODIE model for %d epochs ***" % args.epochs)

# variables to help using tbatch cache between epochs
is_first_epoch = True
cached_tbatches_user = {}
cached_tbatches_item = {}
cached_tbatches_interactionids = {}
cached_tbatches_feature = {}
cached_tbatches_user_timediffs = {}
cached_tbatches_item_timediffs = {}
cached_tbatches_previous_item = {}

with trange(args.epochs) as progress_bar1:
    for ep in progress_bar1:
        progress_bar1.set_description('Epoch %d of %d' % (ep, args.epochs))

        epoch_start_time = time.time()
        # INITIALIZE EMBEDDING TRAJECTORY STORAGE
        user_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).cuda())
        item_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).cuda())

        optimizer.zero_grad()
        reinitialize_tbatches()
        total_loss, loss, total_interaction_count = 0, 0, 0

        tbatch_start_time = None
        tbatch_to_insert = -1
        tbatch_full = False

        # TRAIN TILL THE END OF TRAINING INTERACTION IDX
        with trange(train_end_idx) as progress_bar:
            for j in progress_bar:
                progress_bar.set_description('%dth interaction for training' % j)

                # LOAD INTERACTION J
                userid = user_sequence_id[j]
                itemid = item_sequence_id[j]
                feature = feature_sequence[j]
                user_timediff = user_timediffs_sequence[j]
                item_timediff = item_timediffs_sequence[j]
                timestamp = timestamp_sequence[j]
                if not tbatch_start_time:
                    tbatch_start_time = timestamp
                itemid_previous = user_previous_itemid_sequence[j]


                user_neighbors = {}
                item_neighbors = {}
                user_set=[]
                item_set=[]
                depth=1
                user='u'+str(userid)
                item='i'+str(itemid)
                user_set.append(userid)
                item_set.append(itemid)


                kg_add(kg, user, item, timestamp)
                kg_reduce = threshold(kg, args.threshold * timestamp)
                kg=kg_reduce



                user_neighbors[user] = get_neigbors(kg, user, depth=1)[1]
                item_neighbors[item] = get_neigbors(kg, item, depth=1)[1]
                for i in user_neighbors[user]:
                    if int(i[1:]) not in item_set:
                        item_set.append(int(i[1:]))
                for i in item_neighbors[item]:
                    if int(i[1:]) not in user_set:
                        user_set.append(int(i[1:]))


                # LOAD USER AND ITEM EMBEDDING
                user_embedding_input = user_embeddings[torch.cuda.LongTensor([userid])]
                user_embedding_static_input = user_embedding_static[torch.cuda.LongTensor([userid])]
                item_embedding_input = item_embeddings[torch.cuda.LongTensor([itemid])]
                item_embedding_static_input = item_embedding_static[torch.cuda.LongTensor([itemid])]

                u_embedding_input=user_embeddings[torch.cuda.LongTensor(user_set)]
                i_embedding_input=item_embeddings[torch.cuda.LongTensor(item_set)]

                feature_tensor = Variable(torch.Tensor(feature).cuda()).unsqueeze(0)
                user_timediffs_tensor = Variable(torch.Tensor([user_timediff]).cuda()).unsqueeze(0)
                item_timediffs_tensor = Variable(torch.Tensor([item_timediff]).cuda()).unsqueeze(0)
                item_embedding_previous = item_embeddings[torch.cuda.LongTensor([itemid_previous])]

                # PROJECT USER EMBEDDING
                user_projected_embedding = model.forward(u_embedding_input,i_embedding_input,user_set,item_set,
                                                         user_embedding_input, item_embedding_previous,
                                                         timediffs=user_timediffs_tensor, features=feature_tensor,
                                                         select='project')
                user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous,
                                                 item_embedding_static[torch.cuda.LongTensor([itemid_previous])],
                                                 user_embedding_static_input], dim=1)

                # PREDICT ITEM EMBEDDING,当前用户预测embedding
                predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

                # CALCULATE PREDICTION LOSS
                loss += MSELoss(predicted_item_embedding,
                                torch.cat([item_embedding_input, item_embedding_static_input], dim=1).detach())

                # UPDATE USER AND ITEM EMBEDDING
                user_embedding_output,i_embedding_output = model.forward(u_embedding_input,i_embedding_input,user_set,item_set,
                                                      user_embedding_input, item_embedding_input,
                                                      timediffs=user_timediffs_tensor, features=feature_tensor,
                                                      select='user_update')
                item_embedding_output,u_embedding_output = model.forward(u_embedding_input,i_embedding_input,user_set,item_set,
                                                      user_embedding_input, item_embedding_input,
                                                      timediffs=item_timediffs_tensor, features=feature_tensor,
                                                      select='item_update')

                # SAVE EMBEDDINGS
                item_embeddings[itemid, :] = item_embedding_output.squeeze(0)
                user_embeddings[userid, :] = user_embedding_output.squeeze(0)
                item_embeddings[item_set] = i_embedding_output.squeeze(0)
                user_embeddings[user_set] = u_embedding_output.squeeze(0)

                user_embeddings_timeseries[j, :] = user_embedding_output.squeeze(0)
                item_embeddings_timeseries[j, :] = item_embedding_output.squeeze(0)

                # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
                loss += MSELoss(item_embedding_output, item_embedding_input.detach())
                loss += MSELoss(user_embedding_output, user_embedding_input.detach())

                # CALCULATE STATE CHANGE LOSS
                if args.state_change:
                    loss += calculate_state_prediction_loss(model, [j], user_embeddings_timeseries, y_true,
                                                            crossEntropyLoss)

                    # UPDATE THE MODEL IN REAL-TIME USING ERRORS MADE IN THE PAST PREDICTION
                if timestamp - tbatch_start_time > tbatch_timespan:
                    tbatch_start_time = timestamp
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # RESET LOSS FOR NEXT T-BATCH
                    loss = 0
                    item_embeddings.detach_()
                    user_embeddings.detach_()
                    item_embeddings_timeseries.detach_()
                    user_embeddings_timeseries.detach_()

        print("Last epoch took {} minutes".format((time.time()-epoch_start_time)/60))
        # END OF ONE EPOCH 
        print("\n\nTotal loss in this epoch = %f" % (total_loss))
        item_embeddings_dystat = torch.cat([item_embeddings, item_embedding_static], dim=1)
        user_embeddings_dystat = torch.cat([user_embeddings, user_embedding_static], dim=1)
        # SAVE CURRENT MODEL TO DISK TO BE USED IN EVALUATION.
        save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx, user_embeddings_timeseries, item_embeddings_timeseries)

        user_embeddings = initial_user_embedding.repeat(num_users, 1)
        item_embeddings = initial_item_embedding.repeat(num_items, 1)

# END OF ALL EPOCHS. SAVE FINAL MODEL DISK TO BE USED IN EVALUATION.
print("\n\n*** Training complete. Saving final model. ***\n\n")
save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx, user_embeddings_timeseries, item_embeddings_timeseries)

