Certainly! Here
's the modified code with comments added to explain each function used and provide additional context for a Python beginner:

```python
import argparse
import numpy as np
import torch
import trainer
import utils
import models
import attention

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Graph Neural Network Script')
parser.add_argument('--hidden_units', type=int, default=64,
                    help='Number of hidden units in the GNN model')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate for the GNN model')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate for training the GNN model')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs')
parser.add_argument('--save_dir', type=str, default=None,
                    help='Directory to save the trained models')
args = parser.parse_args()

# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = utils.load_data()

# Initialize GNN models and trainers
gnn_r = models.GNNr(args.hidden_units, args.dropout)
gnn_p = models.GNNp(args.hidden_units, args.dropout)
gnn_r_trainer = trainer.Trainer(gnn_r, args.lr)
gnn_p_trainer = trainer.Trainer(gnn_p, args.lr)

# Pre-training GNNr
gnn_r_target_labels = utils.get_target_labels(labels, adj)
gnn_r_trainer.initialize(gnn_r_target_labels)
gnn_r_best_model = gnn_r_trainer.train(adj, features, idx_train, idx_val, idx_test, args.epochs)

# Update data for GNNp
input_p = attention.update_input(gnn_r_best_model, adj, features)

# Training GNNp
gnn_p_trainer.initialize(labels)
gnn_p_best_model = gnn_p_trainer.train(adj, input_p, idx_train, idx_val, idx_test, args.epochs)

# Updating data and training GNNr iteratively
for epoch in range(args.epochs):
    # Update input data for GNNr
    input_r = attention.update_input(gnn_p_best_model, adj, features)

    # Update node labels and adjacency matrix for GNNr
    confident_labels = utils.update_labels(gnn_r_best_model, input_r, labels, epoch)
    prototypes = utils.compute_prototypes(input_r, confident_labels)
    adj = attention.update_adjacency(adj, prototypes)

    # Train GNNr
    gnn_r_trainer.initialize(confident_labels)
    gnn_r_best_model = gnn_r_trainer.train(adj, input_r, idx_train, idx_val, idx_test, args.epochs)

# Calculate the best test accuracy from GNNr training results
best_test_acc = max(gnn_r_trainer.test_acc_list)

# Print the best test accuracy
print(f"Best test accuracy: {best_test_acc:.2%}")

# Save the trained models if a save directory is specified
if args.save_dir:
    utils.save_model(gnn_r_best_model, args.save_dir, "gnn_r")
    utils.save_model(gnn_p_best_model, args.save_dir, "gnn_p")
```

Now, let
's go through the functions used in the code, one by one:

1.
`argparse.ArgumentParser()`
creates
an
argument
parser
object
to
handle
command - line
arguments and options.

1.
`parser.add_argument()`
adds
arguments
to
the
argument
parser.The
function
takes
several
arguments, such as the
argument
name(`--hidden_units`), the
argument
data
type(`int`, `float`, or `str`), default
values, and help
messages.

1.
`args = parser.parse_args()
` parses
the
command - line
arguments and stores
them in the
`args`
object.

1.
`np.random.seed(0)`
sets
the
random
seed
for NumPy to ensure reproducibility.

1.
`torch.manual_seed(0)`
sets
the
random
seed
for PyTorch to ensure reproducibility.

1.
`utils.load_data()` is a
custom
function
that
loads
the
graph
data, including
the
adjacency
matrix, node
features, node
labels, and index
splits
for training, validation, and testing.

1.
`models.GNNr()`
creates
an
instance
of
the
GNNr
model, which is a
graph
neural
network
for node - level classification.

1.
`models.GNNp()`
creates
an
instance
of
the
GNNp
model, which is a
graph
neural
network
for semantic - level classification.

1.
`trainer.Trainer()`
creates
an
instance
of
the
Trainer


class , which is responsible for training the GNN models.


1.
`utils.get_target_labels()` is a
custom
function
that
initializes
the
target
labels
for the GNNr model using a one-hot encoding of the edge types based on the node labels and adjacency matrix.

1.
`gnn_r_trainer.initialize()`
initializes
the
GNNr
trainer
with the target labels obtained from `utils.get_target_labels()`.

1.
`gnn_r_trainer.train()`
trains
the
GNNr
model
using
the
`update_soft`
method
of
the
trainer.It
takes
the
adjacency
matrix, node
features, and index
splits as inputs and performs
training
for the specified number of epochs.

1.
`attention.update_input()` is a
custom
function
that
updates
the
input
data
for the GNNp model based on the node representations obtained from the GNNr model.

1.
`gnn_p_trainer.initialize()`
initializes
the
GNNp
trainer
with the original node labels.

1.
`gnn_p_trainer.train()`
trains
the
GNNp
model
using
the
`update_soft`
method
of
the
trainer.It
takes
the
updated
adjacency
matrix(`adj`), updated
node
features(`input_p`), and index
splits as inputs and performs
training
for the specified number of epochs.

1.
`utils.update_labels()` is a
custom
function
that
updates
the
node
labels
for the GNNr model based on the node representations obtained from the GNNp model and the current epoch.It also returns the confident node labels.

1.
`utils.compute_prototypes()` is a
custom
function
that
calculates
prototypes
for each class based on the confident node labels and node representations.

1.
`attention.update_adjacency()` is a
custom
function
that
updates
the
adjacency
matrix
for the GNNr model based on the prototypes and attention mechanism.

1.
`gnn_r_trainer.test_acc_list`
stores
the
test
accuracies
obtained
during
the
training
of
the
GNNr
model.

1.
`print()`
function
prints
the
best
test
accuracy in percentage
format.

1.
`utils.save_model()` is a
custom
function
that
saves
the
trained
GNN
models(`gnn_r_best_model` and `gnn_p_best_model`) in the
specified
save
directory.

Note
that
some
functions(`utils.load_data()`, `utils.get_target_labels()`, `attention.update_input()`, `utils.update_labels()`,
          `utils.compute_prototypes()`, `attention.update_adjacency()`, and `utils.save_model()`) are
custom
functions and their
implementations
are
not provided in the
code
snippet.You
would
need
to
define
these
functions
separately
based
on
the
specific
requirements
of
your
application.

I
hope
these
explanations
help
you
understand
the
code
better! Let
me
know if you
have
any
further
questions.