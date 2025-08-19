import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

class Rs_GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(Rs_GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        
        conv_nd = nn.Conv1d
        self.in_channels = nfeat
        self.inter_channels = nhid
        self.g = conv_nd(in_channels=nfeat, out_channels=nhid,
                             kernel_size=1, stride=1, padding=0)
        self.dropout = dropout
    

    def forward(self, v):
        batch_size = v.size(0)
        n_r = v.size(2)
        g_v = self.g(v).view(batch_size, self.inter_channels, -1)
        g_v = g_v.permute(0, 2, 1)

        theta_v = v.permute(0, 2, 1)
        phi_v = v
        A = torch.matmul(theta_v, phi_v)
        
        # KNN graph
        K = 10
        indices = torch.topk(A, k=K, dim=-1, sorted=True).indices
        mask = torch.ones(A.shape, dtype=torch.bool)
        batch_idx = torch.arange(batch_size).view(batch_size, 1, 1)      
        node_idx = torch.arange(n_r).view(1, n_r, 1)       
        mask[batch_idx, node_idx, indices] = False
        mask = mask.cuda()
        A = A.cuda()
        A = A.masked_fill(mask, 0)
        I = torch.eye(n_r)
        I = I.unsqueeze(0).repeat_interleave(repeats=batch_size, dim=0).cuda()
        A = A + I
        degrees = torch.sum(A, dim=-1)
        D = torch.diag_embed(degrees)
        D = torch.pow(D, -0.5)
        D[torch.isinf(D)] = 0.0
        A_hat = D @ A @ D
        
        x = F.relu(self.gc1(g_v, A_hat))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, A_hat)
        x = x.permute(0, 2, 1)
        return x + v
        
    
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        conv_nd = nn.Conv1d
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.bmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
               
               