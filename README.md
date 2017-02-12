# Target propagation

![image](https://cloud.githubusercontent.com/assets/7425776/22083370/db10a78a-dd99-11e6-9b98-5d04b201bd64.png)

This repository contains

* a Tensorflow implementation of the difference target propagation (DTP) algorithm for training deep networks, from [Lee, Zhang, Fischer, Bengio 2014](https://arxiv.org/abs/1412.7525)
* python / numpy implementations of new variants of target propagation

Target propagation is an alternative to backpropagation that propagates targets (instead of errors) via inverses (instead of the chain rule).

The layer inverses can be defined explicitly or learned.

Files:
- `targprop.tprop_train` contains the main implementation of target propagation. The main function, `train_net()` trains a network on `dataset=MNIST` or `dataset=cifar`, as a classifier (`mode=classification`) or autoencoder (`mode=autoencoder`), using one of a few different target propagation methods (`err_alg=0`, `1`, `2`, or `3`). This function relies primarily on numpy to do forward/backward propagation. The parameters of the model are updated based on layer-local cost functions, which can be minimized using gradient descent (`update_implementation=numpy`) or using tensorflow's Adam optimizer (`update_implementation=tf`).

- `targprop.tproptflow_train` is similar to `tprop_train.py`, but the entire graph is built in tensorflow. The disadvantage of doing everything with a tensorflow graph is that it is difficult to implement new target propagation methods. 

- `targprop.operations` contain an `Op` class for implementing standard operations. Whereas tensorflow operations require both the function and its derivative, the `Op` class requires the function, its derivative,  its least-squares inverse `f_inv` and regularized least-squares inverse `f_rinv`, which are used in some target propagation methods that we test.

- `targprop.datasets` contain a self-explanatory `DataSet` class.

![image](https://cloud.githubusercontent.com/assets/7425776/22864905/ce9808f8-f127-11e6-9fc2-e666b6744154.png)

-----------

### Links

[Original DTP implementations (Theano)](https://github.com/donghyunlee/dtp)