# Target propagation

![image](https://cloud.githubusercontent.com/assets/7425776/22083370/db10a78a-dd99-11e6-9b98-5d04b201bd64.png)

This repository contains

* a Tensorflow implementation of the difference target propagation (DTP) algorithm for training deep networks, from [Lee, Zhang, Fischer, Bengio 2014](https://arxiv.org/abs/1412.7525)
* python / numpy implementations of new variants of target propagation

Target propagation is an alternative to backpropagation that propagates targets (instead of errors) via inverses (instead of the chain rule).

The layer inverses can be defined explicitly or learned.

Files:
- `targprop.py` contains the numpy implementation of difference target propagation and regularized target propagation. The main function, `run_tprop()` trains a simple feedforward MNIST classifier using three different target/error propagation methods (`err_algs`) and three different weight update methods (`training_algs`). At the moment, it is still not the most user-friendly code...
- `targproptflow.py` contains the tensorflow implementation of difference target propagation and some exploratory versions of regularized target propagation. Here, tensorflow is basically only used to apply gradient descent on the layer-local cost functions.

-----------

### Links

[Original DTP implementations (Theano)](https://github.com/donghyunlee/dtp)