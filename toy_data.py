"""
  Toy datasets for deep learning.
  # TODO: implement uniform-distance sampling (trivial to do with circles)
  # TODO: implement poisson-disk sampling
  # TODO: add DataSet.train.inputs, DataSet.train.outputs, DataSet.test... etc. 
"""
import numpy as np
import gzip
import os
import sys
import tarfile
from six.moves import urllib

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

# def spiral(n=100, d=2, r=1, ):



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

def cifar10_data():
  maybe_download_and_extract('http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
  batch1 = unpickle('./cifar10_data/cifar-10-batches-py/data_batch_1')
  batch2 = unpickle('./cifar10_data/cifar-10-batches-py/data_batch_2')
  batch3 = unpickle('./cifar10_data/cifar-10-batches-py/data_batch_3')
  batch4 = unpickle('./cifar10_data/cifar-10-batches-py/data_batch_4')
  batch5 = unpickle('./cifar10_data/cifar-10-batches-py/data_batch_5')
  inputs = np.concatenate((batch1['data'], batch2['data'], batch3['data'], batch4['data'], batch5['data'])).astype('float32')
  outputs = np.concatenate((batch1['labels'], batch2['labels'], batch3['labels'], batch4['labels'], batch5['labels'])).astype('float32')
  outputs = one_hotify(outputs)
  return DataSet(inputs, outputs)

def cifar10_data_test():
  maybe_download_and_extract('http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
  test   = unpickle('./cifar10_data/cifar-10-batches-py/test_batch')
  inputs = np.array(test['data']).astype('float32')
  outputs = np.array(test['labels']).astype('float32')
  outputs = one_hotify(outputs)
  return DataSet(inputs, outputs)

### HELPER FUNCTIONS
# TODO: make part of DataSet class.
def unpickle(file):
  import cPickle
  fo = open(file, 'rb')
  dict = cPickle.load(fo)
  fo.close()
  return dict

def maybe_download_and_extract(data_url):
  """ From tensorflow cifar10 tutorial """
  dest_directory = './cifar10_data'
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = data_url.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

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

