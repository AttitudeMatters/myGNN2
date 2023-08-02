# myGNN2

A heterophily GNN based on the EM algorithm. We assume edge connecting nodes from different classes has a type. One GNN classify the edge type and the other GNN classify the node class. We iteratively train two GNNs.

## New Benchmarks for Learning on Non-Homophilous Graphs (temp instruction)

Original Paper:
[[PDF link](https://graph-learning-benchmarks.github.io/assets/papers/Non_Homophilous_Camera_Ready.pdf)]

### Organization
`xxx_NH.py` suffix `_NH` stands for scripts related with Non-Homophilous Datasets

`main.py` contains the main experimental scripts.

`dataset.py` loads our datasets.

`homophily.py` contains functions for computing homophily measures, including the one that we introduce in `our_measure`.

### Datasets

![Alt text](https://user-images.githubusercontent.com/58995473/113487776-f3665d80-9487-11eb-8035-fbf5126f85df.png "Our Datasets")

<span style="color.red">Choose arxiv-year dataset for relatively small size and high non-homophily.</span> The arxiv-year and ogbn-proteins datasets are downloaded using OGB downloaders. `load_nc_dataset` returns an NCDataset, the documentation for which is also provided in `dataset.py`. It is functionally equivalent to OGB's Library-Agnostic Loader for Node Property Prediction, except for the fact that it returns torch tensors. See the [OGB website](https://ogb.stanford.edu/docs/nodeprop/) for more specific documentation. Just like the OGB function, `dataset.get_idx_split()` returns fixed dataset split for training, validation, and testing. 

![Alt text](https://user-images.githubusercontent.com/58995473/113487966-18a79b80-9489-11eb-91cf-0d5c73ebdef3.png "Dataset Compatibility Matrices")

### Installation instructions

1. Create and activate a new conda environment using python=3.8 (i.e. `conda create --name non-hom python=3.8`) 
2. Activate your conda environment
3. Check CUDA version using `nvidia-smi` 
4. In the root directory of this repository, run `bash install.sh cu110`, replacing cu110 with your CUDA version (i.e. CUDA 11 -> cu110, CUDA 10.2 -> cu102, CUDA 10.1 -> cu101). We tested on Ubuntu 18.04, CUDA 11.0.

## Cite
If you use this code or our results in your research, please cite:
```
@article{lim2021new,
  title={New Benchmarks for Learning on Non-Homophilous Graphs},
  author={Lim, Derek and Li, Xiuyu and Hohne, Felix and Lim, Ser-Nam},
  journal={arXiv preprint arXiv:2104.01404},
  year={2021}
}
```