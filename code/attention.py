import torch
from torch import nn
from torch.nn import Parameter

class NodeAttention(nn.Module):
    # Node-level Attention
    def __init__(self, in_dim, out_dim=128):
        super(NodeAttention, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.a = Parameter(torch.Tensor(out_dim, 1))

    def forward(self, h):
        z = torch.tanh(self.linear(h))
        e = torch.mm(z, self.a)
        return torch.softmax(e, dim=0)

class SemanticAttention(nn.Module):
    # Semantic-level Attention
    def __init__(self, in_dim, out_dim=128):
        super(SemanticAttention, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.a = Parameter(torch.Tensor(out_dim, 1))

    def forward(self, H):
        z = torch.tanh(self.linear(H))
        e = torch.mm(z, self.a)
        return torch.softmax(e, dim=0)