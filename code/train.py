import argparse
import random

import numpy as np
import torch

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

gnnr = GNNr(num_feature=features.shape[1],
            num_hidden=args.hidden,
            num_edge_type=num_edge_type,
            dropout=args.dropout,
            adj=adj)