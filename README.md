# DCCA: Deep Canonical Correlation Analysis

I'm currently adapting the code from https://github.com/VahidooX/DeepCCA.

### Objectives
* ~~Solving compatibility-issues with anterior version of Keras~~
* ~~Make it compatible with Tensorflow backend (remove Theano dependency)~~
* Make it compatible with scikit-learn framework
* Currently, the code is designed to solve a classification problem - I'm interested in solving regression problems

## Description

This is an implementation of Deep Canonical Correlation Analysis (DCCA or Deep CCA) in Python.

DCCA is a non-linear version of CCA which uses neural networks as the mapping functions instead of linear transformers. DCCA is originally proposed in the following paper:

Galen Andrew, Raman Arora, Jeff Bilmes, Karen Livescu, "[Deep Canonical Correlation Analysis.](http://www.jmlr.org/proceedings/papers/v28/andrew13.pdf)", ICML, 2013.

It uses the Keras library with the Tensorflow backend.

### Differences with the original paper
The following are the differences between this implementation and the original paper:

 * The non-saturating version of sigmoid is substituted by another non-saturating activation function (ReLU).
 * Pre-training is not done in this implementation. 

