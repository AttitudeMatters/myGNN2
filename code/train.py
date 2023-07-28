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

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

adj, features, labels, idx_train, idx_val, idx_test = load_data()
num_edge_type = get_num_edge_type(labels, 2)

gnnr = GNNr(num_feature=features.shape[1],
            num_hidden=args.hidden,
            num_edge_type=num_edge_type,
            dropout=args.dropout,
            adj=adj)