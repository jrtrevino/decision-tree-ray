# decision-tree-ray

Random Forest is an ensemble classifier that makes predictions about datapoints using numerous decision trees. Often, students beginning their journey into the world of machine learning are requested to create their own classifier. Often, this is the C4.5 algorithm for a decision tree. 

Unfortunately, this classifier is rather outdated and great for teaching purposes, but can be extremely inefficient. This is especially true in cases where vectorized approaches are not considered.

## Problem

Many C4.5 implementations take a long time to run and often times these programs are ran overnight with the hopes of it producing results in the morning. If a decision tree generator takes long to produce a viable tree, then extending out this algorithm to a random forest classifier would only lengthen the amount of time needed. We attempt to speed up these solutions by using Ray, a powerful Python library that distributes a workload across worker-nodes.

## Solution

In this repository contains a file named `ray-c45.ipynb.` This Jupyter notebook contains the process and steps required to parallelize a C4.5 algorithm using the Ray library, located in `induceC45.py`. In the notebook, we parallelize the decision tree generation as well as the classification process.

Because decision trees are essentially a JSONStree, we need an algorithm to traverse it to make a prediction regarding a data point. This algorithm is as simple as a breadth-first-search, located in `Classifier.py`

### Results

We see mixed results in our implementation. One, our generation has a noticable speed increase. Two, our classification process does not see any improvement. This is probably due to tree traversal being relatively quick in the first place, and thus Ray's overhead mitigates any benefits from parallelization.

## Going further

I would like to attempt to distribute the workload across worker-nodes in a virtual machine cluster, instead of only on a local machine. Currently using Ray on a local machine produces results similar to a implementation using a multiprocessing library in Python. Overall, exciting!

