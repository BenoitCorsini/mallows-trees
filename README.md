# Mallows trees

This project computes and represents Mallows trees.
It is meant to represent Mallows trees as defined in [The height of Mallows trees](https://arxiv.org/abs/2007.13728)

## Getting started

### Organization of the code

`model.py` contains the two class: `MallowsPermutation` and `MallowsTree`. The class `MallowsTree` builds a binary search tree from an instance of the `MallowsPermutation` class.
`plots.py` contains the class `TreePlots` which is used to represent instances of the `MallowsTree` class. It has three main functions:
* `tree` which creates an image representing the tree.
* `construction` which creates a gif representing the construction of the Mallows tree from its permutation.
* `evolution` which creates a gif representing the evolution of a Mallows tree according to its parameter q.
`main.py` used to run the algorithms.

### Running the code

The library requirements for this code can be found in `requirements.txt`. In order to install them, run the following command in the terminal:
```sh
pip install -r requirements.txt
```
Once this is done, run the following command in the terminal:
```sh
python main.py
```

The parameters in `main.py` can easily be changed by adding extra arguments at the end of the command line. All parameters can be found in `main.py` but an example where the code would run on trees of size 20 instead of 10, with a seed of 42 for reprodicibility would be
```sh
python main.py --n 20 --seed 42
```

## Results

This code produces three types of results.

### Tree representation

Mallows trees can be represented. The position of the node corresponds to its value and its depth. Typical figures look like the following.

<p align="center"><img style="width:50%;border:solid;" src="figures/mallows-tree.png"/></p>

### Tree construction

The construction of Mallows trees can be represented. This construction corresponds to the typical construction of binary search trees from a permutation. Typical figures look like the following.

![construction](/figures/construction.gif)

### Tree evolution

The evolution of Mallows trees according to their parameter q can be represented. A specific coupling is used here to obtain correlated trees (see the definition of `MallowsPermutation` in `model.py`). Typical figures look like the following.

![evolution](/figures/evolution.gif)