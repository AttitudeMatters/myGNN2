import torch
from torch import nn
import torch.nn.functional as F
from layers import GraphConvolution


class GNNr(nn.Module):
    def __init__(self, num_feature, num_hidden, num_edge_type, dropout, adj):
        super(GNNr, self).__init__()
        self.adj = adj

        self.m1 = GraphConvolution(num_feature, num_hidden, adj)
        self.m2 = GraphConvolution(num_hidden, num_hidden, adj)
        self.fc = nn.Linear(2 * num_hidden, num_edge_type) # predict edge type
        self.dropout = dropout

    #     cuda待写

    def reset(self):
        self.m1.reset_parameters()
        self.m2.reset_parameters()
        self.fc.reset_parameters()

    def forward(self, x, edges):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.m1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.m2(x)
        x = F.relu(x)

        # get node representation for each edge
        edge_repr = torch.cat([x[edges[0]], x[edges[1]]], dim=-1)  # concatenate representations

        # predict edge type
        out = self.fc(edge_repr)

        return out


class GNNp(nn.Module):
    def __init__(self, num_edge_type, num_hidden, num_class, dropout, adj):
        super(GNNp, self).__init__()
        self.adj = adj

        self.m1 = GraphConvolution(num_edge_type, num_hidden, adj)
        self.m2 = GraphConvolution(num_hidden, num_class, adj)
        self.dropout = dropout

    def reset(self):
        self.m1.reset_parameters()
        self.m2.reset_parameters()

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.m1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.m2(x)
        return x
