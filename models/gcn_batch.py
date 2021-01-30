import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
#from layers import GraphConv, Relation, Relationv2, Discriminator, Generator

# GCN first
class GCN_Batch(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, device):
        super(GCN_Batch, self).__init__()

        self.W1 = nn.Linear(nfeat, nhid, bias=False)
        self.W2 = nn.Linear(nhid, nclass, bias=False)
        

    def gcn(self, x, adj, mask, W):
        
        adj = adj[:,1:]
        mask = mask[:,1:]
        
        x = W(x)

        '''
        embedding = nn.Embedding.from_pretrained(x, norm_type=None)
        embedding.weight.requires_grad = True
        embed = embedding(adj)
        '''
        #print('check 0')

        '''
        #lookup
        embed = x[adj[0]].unsqueeze(0)
        for i in range(1, adj.shape[0]):
            feat = x[adj[i]].unsqueeze(0)
            embed = torch.cat((embed,feat))
        '''
        embed = x[adj]
        
        #print('check 1')

        embed = embed * mask[:,:,None]
        embed = torch.sum(embed, axis=1) 
        
        #print('check 2')

        return embed

        #embed = F.dropout(embed, 0.5)
        #return F.elu(embed)


    def forward(self, x, adj1, adj1_mask, adj2, adj2_mask):

        # 1st layer
        h1 = self.gcn(x, adj1, adj1_mask, self.W1)
        h1 = F.dropout(h1, 0.5, training=self.training)
        h1 = F.elu(h1)

        # 2nd layer
        h2 = self.gcn(h1, adj2, adj2_mask, self.W2)
        #h2 = F.dropout(h2, 0.5)
        #h2 = F.elu(h2)

        return h2


   