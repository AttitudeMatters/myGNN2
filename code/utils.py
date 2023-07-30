import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import math


def load_data():
    dataset = Planetoid(root='../data', name='Cora', transform=NormalizeFeatures())

    # Adjacency matrix
    edge_index = dataset[0].edge_index

    # Node features
    features = dataset[0].x

    # Labels
    labels = dataset[0].y

    # For Cora, the first 140 nodes are used for training, the next 500 for validation, and the rest for testing
    train_mask = torch.zeros(labels.shape, dtype=bool)
    val_mask = torch.zeros(labels.shape, dtype=bool)
    test_mask = torch.zeros(labels.shape, dtype=bool)

    train_mask[:140] = 1
    val_mask[140:140 + 500] = 1
    test_mask[140 + 500:] = 1

    return edge_index, features, labels, train_mask, val_mask, test_mask


# combination
def get_num_edge_type(n, r):
    if n < r:
        raise ValueError("n must be greater than or equal to r")
    return math.comb(n, r)

def class_pairs_to_edge_type(num_classes):
    edge_type_dict = {}
    counter = 0
    for i in range(num_classes):
        for j in range(i, num_classes):  # we only traverse the upper triangle to ensure consistent types of undirected edges
            edge_type_dict[(i, j)] = counter
            edge_type_dict[(j, i)] = counter  # For undirected edges, we hope that (i, j) and (j, i) have the same type
            counter += 1
    return edge_type_dict