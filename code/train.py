import argparse
import copy
import random

import numpy as np
import torch

from trainer import Trainer
from utils import *
from models import *
from attention import *

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--outdim', type=int, default=128, help='Outdim for the node and semantic attention.')
parser.add_argument('--pre_epoch', type=int, default=200, help='Number of pre-training epochs.')
parser.add_argument('--iter', type=int, default=10, help='Number of training iterations.')
parser.add_argument('--epoch', type=int, default=200, help='Number of training epochs per iteration.')
parser.add_argument('--save', type=str, default='/')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
out_dim = args.outdim

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

edge_types = generate_edge_type(adj, labels, num_classes, num_edges)

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

    # Iterate over edges to be trained
    for i in range(idx_train):
        node1, node2 = edge_index[:, i]
        class1, class2 = node_labels[node1], node_labels[node2]

        # Ensure the pair of classes is in a consistent order
        class_pair = tuple(sorted((class1.item(), class2.item())))

        # Get the edge type
        edge_type = edge_type_dict[class_pair]

        # Initialize the target_r for this edge with a one-hot encoding of the edge type
        target_r[i] = 0
        target_r[i, edge_type] = 1


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
    # Get the edge type predictions from GNNr model
    edge_type_preds = trainer_r.predict(input_r)

    # Initialize a list to store the node representations for each type of subgraph
    subgraph_node_representations = []

    # Iterate over all edge types
    for edge_type in edge_types:
        # Create a subgraph based on the current edge type
        subgraph = create_subgraph_based_on_edge_type(adj, edge_type_preds, edge_type)

        # Apply Node-level Attention Mechanism on each subgraph
        node_attention = NodeAttention(num_features, out_dim)
        node_repr = node_attention(subgraph)

        subgraph_node_representations.append(node_repr)

    # Compute semantic-level attention scores
    semantic_attention = SemanticAttention(out_dim, 1)
    semantic_scores = semantic_attention(torch.stack(subgraph_node_representations))

    # Combine the node representations of each subgraph according to semantic-level attention scores
    combined_node_repr = torch.zeros(num_nodes, out_dim)
    for semantic_score, node_repr in zip(semantic_scores, subgraph_node_representations):
        combined_node_repr += semantic_score * node_repr

    # Update input_p
    input_p = combined_node_repr


def train_p(epoches):
    update_p_data()
    results = []
    for epoch in range(epoches):
        loss = trainer_p.update_soft(input_p, target_p, idx_all)
        _, preds, accuracy_dev = trainer_p.evaluate(input_p, labels, idx_dev)
        _, preds, accuracy_test = trainer_p.evaluate(input_p, labels, idx_test)
        results += [(accuracy_dev, accuracy_test)]
    return results


def calculate_confidence_threshold(epoch, start_threshold=0.9, end_threshold=0.5, total_epochs=100):
    """
    Calculate a linearly decreasing confidence threshold based on the current epoch.
    """
    threshold_decrease = (start_threshold - end_threshold) / total_epochs
    return start_threshold - threshold_decrease * epoch



def update_confident_labels(node_label_preds, epoch):
    # Dynamically calculate the confidence threshold based on the current epoch
    confidence_threshold = calculate_confidence_threshold(epoch)

    # Get the maximum predicted probability for each node
    max_probs, max_labels = torch.max(node_label_preds, dim=1)

    # Only update the labels that are confident
    node_labels_confident = torch.where(max_probs > confidence_threshold, max_labels, -1)

    return node_labels_confident


def update_r_data(node_representations, validation_performance):
    # Get the node label predictions from GNNp model
    node_label_preds = trainer_p.predict(input_p)

    # Update the node labels that are confident
    node_labels_confident = update_confident_labels(node_label_preds, validation_performance)

    # Calculate the prototypes for each class
    prototypes = calculate_prototypes(node_labels_confident, node_representations)

    # Initialize the adjacency matrix for GNNr
    adj_r = torch.zeros(num_nodes, num_nodes)

    # Iterate over all pairs of nodes
    for node1 in range(num_nodes):
        for node2 in range(node1 + 1, num_nodes):
            # Calculate the new edge type with attention mechanism
            edge_type = calculate_edge_type_with_attention(node1, node2, prototypes, node_representations)

            # Update the adjacency matrix
            adj_r[node1, node2] = edge_type
            adj_r[node2, node1] = edge_type

    # Update input_r
    input_r = adj_r

    return input_r



def train_r(epoches):
    results = []
    for epoch in range(epoches):
        # Train GNNr model
        node_representations = update_p_data()
        input_r = update_r_data(node_representations, epoch)
        loss = trainer_r.update_soft(input_r, target_r, idx_all)
        _, preds, accuracy_dev = trainer_r.evaluate(input_r, edge_types, idx_dev)
        _, preds, accuracy_test = trainer_r.evaluate(input_r, edge_types, idx_test)
        results += [(accuracy_dev, accuracy_test)]
    return results


base_results, r_results, p_results = [], [], []
base_results += pre_train(args.pre_epoch)
for k in range(args.iter):
    p_results += train_p(args.epoch)
    r_results += train_r(args.epoch)

def get_accuracy(results):
    best_dev, acc_test = 0.0, 0.0
    for d, t in results:
        if d > best_dev:
            best_dev, acc_test = d, t
    return acc_test

acc_test = get_accuracy(r_results)

print('{:.3f}'.format(acc_test * 100))

if args.save != '/':
    trainer_r.save(args.save + '/gnnr.pt')
    trainer_p.save(args.save + '/gnnp.pt')