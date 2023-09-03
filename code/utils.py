import torch
from torch import nn
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


def generate_edge_type(edge_index, node_labels, num_classes, num_edges):
    edge_type_dict = class_pairs_to_edge_type(num_classes)
    edge_types = torch.zeros(num_edges, dtype=torch.long)  # assuming edge types are represented as integers
    for i in range(num_edges):
        node1, node2 = edge_index[:, i]
        class1, class2 = node_labels[node1], node_labels[node2]
        class_pair = tuple(sorted((class1.item(), class2.item())))
        edge_type = edge_type_dict[class_pair]
        edge_types[i] = edge_type
    return edge_types


def create_subgraph_based_on_edge_type(adj, edge_type_preds, edge_type):
    # Filter the edges based on the predicted edge type
    edge_mask = (edge_type_preds == edge_type)
    subgraph_adj = adj.clone()
    subgraph_adj[~edge_mask] = 0
    return subgraph_adj


def calculate_prototypes(node_labels_confident, node_representations):
    # Calculate the prototype (mean representation) for each class
    prototypes = {}
    for class_label in torch.unique(node_labels_confident):
        mask = (node_labels_confident == class_label)
        class_representations = node_representations[mask]
        prototypes[class_label.item()] = class_representations.mean(dim=0)
    return prototypes


def calculate_edge_type_with_attention(node1, node2, prototypes, node_representations):
    # Compute the similarity between the nodes and their class prototypes
    prototype1 = prototypes[node1.item()]
    prototype2 = prototypes[node2.item()]
    node_repr1 = node_representations[node1]
    node_repr2 = node_representations[node2]

    # Use cosine similarity as the attention score
    cos_sim = nn.CosineSimilarity(dim=0)
    attention_score1 = cos_sim(node_repr1, prototype1)
    attention_score2 = cos_sim(node_repr2, prototype2)

    # Edge type is the class with the higher attention score
    edge_type = node1 if attention_score1 > attention_score2 else node2

    return edge_type
