"""
  Toy datasets for deep learning.
  # TODO: implement uniform-distance sampling (trivial to do with circles)
  # TODO: implement poisson-disk sampling
  # TODO: add DataSet.train.inputs, DataSet.train.outputs, DataSet.test... etc. 
"""
import numpy as np

### DATASET CLASS
class DataSet(object):
  def __init__(self,
               inputs,
               outputs=None):
    """
      Construct a DataSet object.
      Adapted from mnist.py from the TensorFlow code base.
      Inputs: a shape (N, d_1) numpy array; N samples of a d_1-dimensional vector.
      Outputs: a shape (N, d_2) numpy array; N samples of a d_2-dimensional vector.
    """
    self._inputs = inputs
    self._outputs = outputs
    self._num_examples = inputs.shape[0]
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def inputs(self):
    return self._inputs
  
  @property
  def outputs(self):
    return self._outputs

  @inputs.setter
  def inputs(self, value):
    self._inputs = value

  @outputs.setter
  def outputs(self, value):
    self._outputs = value

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed
  
  def rand_batch(self, batch_size):
    """
      Return a random batch of input / outputs.
      x_batch, y_batch = DataSet.rand_batch(batch_size) if outputs is not None
      x_batch = DataSet.rand_batch(batch_size) if outputs is None
    """
    inds = np.random.choice(self._num_examples, batch_size)
    if self._outputs is None:
      return self._inputs[inds]
    else:
      return self._inputs[inds], self._outputs[inds]

  def next_batch(self, batch_size):
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      np.random.seed(self._epochs_completed) # ensure the same shuffling behavior for each experiment.
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._inputs = self._inputs[perm]
      self._outputs = self._outputs[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    if self._outputs is None:
      return self._inputs[start:end]
    else:
      return self._inputs[start:end], self._outputs[start:end]


### TOY DATA
def sphere(n=100, d=3, r=1., c=0):
  """
    Uniformly sample from a sphere.
    n data points, d dimensions, radius r
  """
  # Assert that c.shape = (d,)
  x = np.random.randn(n, d)
  return DataSet(r*x/np.linalg.norm(x, axis=1, keepdims=True) + c)

def ball(n=100, d=3, r=1, c=0):
  """
    Uniformly sample from the interior of a sphere
  """
  x = sphere(n, d, r).inputs
  u = r*np.random.rand(n)**(1./d)
  return DataSet(u[:, np.newaxis]*x + c)

def annulus(n=100, d=3, r1=0.5, r2=2, c=0):
  """
    Sample from an annulus / shell, where r1 is the inner radius, r2 is the outer radius
  """
  x = sphere(n, d, 1).inputs
  u = (r2-r1)*np.random.rand(n)**(1./d) + r1
  #u = u**(1./d)
  return DataSet(u[:, np.newaxis]*x + c)

def torus(n=100, d=2, r=1, c=0):
  """
    Sample from a (high-dimensional) torus - the product of d circles embedded in 2d-dimensional space
  """
  x = []
  for _ in range(d):
    x.append(sphere(n, 2, r, c).inputs)
  return DataSet(np.concatenate(x, axis=1))

def torus_annulus(n=100, d=3, r1=0.5, r2=2, c=0):
  """
    Sample from a torus annulus / shell.
  """
  x = []
  for _ in range(d):
    x.append(annulus(n, 2, r1, r2, c).inputs)
  return DataSet(np.concatenate(x, axis=1))

def xor_data():
  inputs = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]).astype('float32')
  outputs = np.array([[1], [-1], [1], [-1]]).astype('float32')
  return DataSet(inputs, outputs)

### REAL DATA
def mnist_data():
  from tensorflow.examples.tutorials.mnist import input_data
  mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
  return DataSet(mnist.train.images, mnist.train.labels)

def mnist_data_test():
  # TODO: make dataset class have both train and test...
  from tensorflow.examples.tutorials.mnist import input_data
  mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
  return DataSet(mnist.test.images, mnist.test.labels)

def simple_mnist_data(digits=3):
  from tensorflow.examples.tutorials.mnist import input_data
  mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
  inds = mnist.train.labels[:, :digits].sum(axis=1).astype('bool')
  return DataSet(mnist.train.images[inds], mnist.train.labels[inds])

### HELPFUL FUNCTIONS
# TODO: make part of DataSet class.
def combine_data(data):
  """
    Turns a list of DataSet objects into one DataSet object, where .inputs and .outputs are concatenated.
    Input: a list of DataSet objects.
  """
  inputs = np.concatenate([data[i].inputs for i in range(len(data))])
  if data[0].outputs is not None:
    outputs = np.concatenate([data[i].outputs for i in range(len(data))])
    return DataSet(inputs, outputs)
  else:
    return DataSet(inputs)

def one_hotify(vector):
  """
    Converts an (N, 1) vector into a one-hot representation of the array of shape (N, d), where d is the number of unique entries in the vector
  """
  cats = np.unique(vector) # categories
  hot = np.zeros((vector.shape[0], cats.size)) # the one-hot representation
  for i, c in enumerate(cats):
    hot[vector == c, i] = 1.
  return hot

def un_hotify(vector):
  """
    The inverse of one_hotify.
    E.g.:
    [[0, 1, 0, 0],
     [0, 0, 0, 1]] 
     gets converted into [1, 3]
  """
  return np.nonzero(vector)[1].astype('float64')

