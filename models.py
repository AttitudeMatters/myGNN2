import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from layers import GraphConvolution


class GNNr(nn.Module):
    def __init__(self, num_feature, num_hidden, edge_type, dropout, adj):
        super(GNNr, self).__init__()
        self.adj = adj

        self.m1 = GraphConvolution(num_feature, num_hidden, adj)
        self.m2 = GraphConvolution(num_hidden, edge_type, adj)
        self.dropout = dropout

    #     cuda待写

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


class GNNp(nn.Module):
    def __init__(self, edge_type, num_hidden, num_class, dropout, adj):
        super(GNNp, self).__init__()
        self.adj = adj

        self.m1 = GraphConvolution(edge_type, num_hidden, adj)
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
