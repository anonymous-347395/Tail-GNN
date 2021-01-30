import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
from layers import GraphConv, Relation, Relationv2, Generator


# LRGCN 
class LRGCN_Batch(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, device, g_sigma, ver=1):
        super(LRGCN_Batch, self).__init__()

        self.W1 = nn.Linear(nfeat, nhid, bias=False)
        self.W2 = nn.Linear(nhid, nclass, bias=False)
        
        if ver == 1:
            self.r1 = Relation(nfeat)
            self.r2 = Relation(nhid)
        else:
            self.r1 = Relationv2(nfeat,nhid)
            self.r2 = Relationv2(nhid,nclass)
        
        self.g1 = Generator(nfeat, g_sigma)
        self.g2 = Generator(nhid, g_sigma)


    def lrgcn(self, x, adj, w_gcn, w_r, tail):

        adj_0 = adj[0][:,1:].long()
        adj_1 = adj[1][:,1:]
        adj_2 = adj[2][:,1:]

        #get lookup
        x = x[adj_0]

        mean_x = x * adj_1[:,:,None]
        mean_x = torch.mean(mean_x, axis=1)

        #relation
        m_info = w_r(x, mean_x) 
        h_s = self.g(m_info)

        if tail:
            x = w_gcn(x)
            embed = embed * adj_2[:,:,None]
            embed = torch.sum(embed, axis=1) 
            norm = np.sum(adj_1, axis=1) + 1
            embed = (embed + h_s) / norm
        else:
            x = w_gcn(x)
            embed = embed * adj_2[:,:,None]
            embed = torch.mean(embed, axis=1) 

        return embed, m_info

        #embed = F.dropout(embed, 0.5)
        #return F.elu(embed)


    def forward(self, x, adj1, adj2, tail=False):

        # 1st layer
        h1, m1 = self.lrgcn(x, adj1, self.W1, self.r1, tail)

        h1 = F.dropout(h1, 0.5, training=self.training)
        h1 = F.elu(h1)

        # 2nd layer
        h2, m2 = self.lrgcn(h1, adj2, self.W2, self.r2, tail)
        #h2 = F.dropout(h2, 0.5)
        #h2 = F.elu(h2)


        return h2, m


   