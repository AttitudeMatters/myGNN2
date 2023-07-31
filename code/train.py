import argparse
import copy
import random

import numpy as np
import torch

from trainer import Trainer
from utils import *
from models import *

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

adj, features, labels, idx_train, idx_val, idx_test = load_data()

num_classes = labels.max().item() + 1
num_edge_type = get_num_edge_type(num_classes, 2)
num_nodes = labels.shape[0]
num_edges = adj.shape[1]
num_features = features.shape[1]

idx_train = torch.arange(num_nodes)[idx_train]
idx_dev = torch.arange(num_nodes)[idx_val]
idx_test = torch.arange(num_nodes)[idx_test]
idx_all = torch.arange(num_nodes)


input_r = torch.zeros(num_edges, num_features)
target_r = torch.zeros(num_edges, num_edge_type)
input_p = torch.zeros(num_nodes, num_edge_type)
target_p = torch.zeros(num_nodes, num_classes)


gnnr = GNNr(num_feature=num_features,
            num_hidden=args.hidden,
            num_edge_type=num_edge_type,
            dropout=args.dropout,
            adj=adj)

gnnp = GNNp(num_edge_type=num_edge_type,
            num_hidden=args.hidden,
            num_class=num_classes,
            dropout=args.dropout,
            adj=adj)

trainer_r = Trainer(args, gnnr)
trainer_p = Trainer(args, gnnp)

def init_r_data(edge_index, node_labels):
    # Assuming edge_index is a 2xn tensor, where n is the number of edges
    # and node_labels is a tensor of node class labels.

    # Define a mapping from a pair of classes to an edge type.
    # This can be a dictionary, where the key is a tuple of class labels
    # and the value is the corresponding edge type.
    edge_type_dict = class_pairs_to_edge_type(num_classes)

    # Iterate over all edges
    for i in range(num_edges):
        node1, node2 = edge_index[:, i]
        class1, class2 = node_labels[node1], node_labels[node2]

        # Ensure the pair of classes is in a consistent order
        class_pair = tuple(sorted((class1, class2)))

        # Get the edge type
        edge_type = edge_type_dict[class_pair]

        target_r[i] = edge_type


def pre_train(epoches):
    best = 0.0
    init_r_data(adj, labels)
    results = []
    for epoch in range(epoches):
        loss = trainer_r.update_soft(input_r, target_r, idx_train)
        _, preds, accuracy_dev = trainer_r.evaluate(input_r, labels, idx_dev)
        _, preds, accuracy_test = trainer_r.evaluate(input_r, labels, idx_test)
        results += [(accuracy_dev, accuracy_test)]
        if accuracy_dev > best:
            best = accuracy_dev
            state = dict([('model', copy.deepcopy(trainer_r.model.state_dict())), ('optim', copy.deepcopy(trainer_r.optimizer.state_dict()))])
    trainer_r.model.load_state_dict(state['model'])
    trainer_r.optimizer.load_state_dict(state['optim'])
    return results


def update_p_data():
    pass


def train_p(epoches):
    update_p_data()
    results = []
    for epoch in range(epoches):
        loss = trainer_p.update_soft(input_p, target_p, idx_all)
        _, preds, accuracy_dev = trainer_p.evaluate(input_p, labels, idx_dev)
        _, preds, accuracy_test = trainer_p.evaluate(input_p, labels, idx_test)
        results += [(accuracy_dev, accuracy_test)]
    return results


def update_r_data():
    pass


def train_r(epoches):
    update_r_data()
    results = []
    for epoch in range(epoches):
        loss = trainer_r.update_soft(input_r, target_r, idx_all)
        _, preds, accuracy_dev = trainer_r.evaluate(input_r, edge_type, idx_dev)
        _, preds, accuracy_test = trainer_r.evaluate(input_r, edge_type, idx_test)
        results += [(accuracy_dev, accuracy_test)]
    return results