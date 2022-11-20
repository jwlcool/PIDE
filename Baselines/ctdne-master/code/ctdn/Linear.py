import torch
import torch.utils.data
from torch import optim, nn
from torchvision import datasets
from torchvision.transforms import transforms
import torch.nn.functional as F
from cmg import cmd_opt,cmd_args


class MLP(nn.Module):
    def __init__(self,embeddings,embedding_size,):
        super(MLP, self).__init__()
        self.embedding_size=embedding_size
        self.embeddings=embeddings
        self.mlp = nn.Sequential(
            nn.Linear(2*self.embedding_size, self.embedding_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.embedding_size, 10),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10,1),
            nn.Sigmoid()
        )

    def forward(self, pos_edges, neg_edges):
        for u, v, prop in pos_edges:
            # get node representation and average them
            u_enc = self.embeddings.get(u)
            v_enc = self.embeddings.get(v)
            if (u_enc is None) or (v_enc is None):
                continue
            user=torch.Tensor(u_enc)
            item=torch.Tensor(u_enc)

            x=torch.cat((user,item),-1)
            x = self.mlp(x)

        return x
