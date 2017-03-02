"""
  Toy datasets for deep learning.
  # TODO: implement uniform-distance sampling (trivial to do with circles)
  # TODO: implement poisson-disk sampling
  # TODO: add DataSet.train.inputs, DataSet.train.outputs, DataSet.test... etc. 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gzip
import os
import tempfile

import sys
import tarfile
import tensorflow as tf
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
#from tensorflow.python.platform import gfile
#from tensorflow.python.platform.default import _gfile as gfile


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
  x = sphere(n, d, 1.).inputs
  u = r*np.random.rand(n)**(1./d)
  return DataSet(u[:, np.newaxis]*x + c)

def annulus(n=100, d=3, r1=0.5, r2=2, c=0):
  """
    Uniformly sample from an annulus / shell, where r1 is the inner radius, r2 is the outer radius
  """
  x = sphere(n, d, 1).inputs
  u = (r2-r1)/r2*np.random.rand(n) + r1/r2
  u = r2*u**(1./d)
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

def xor_full(n=100):
  ''' like xor, but full quadrants intsead of single points '''
  split = np.array_split(np.arange(n), 4)
  inputs = np.random.rand(n, 2)
  inputs[split[0], :] += np.array([[ 0,  0]])
  inputs[split[1], :] += np.array([[-1, -1]])
  inputs[split[2], :] += np.array([[-1,  0]])
  inputs[split[3], :] += np.array([[ 0, -1]])
  outputs = np.zeros_like(inputs)
  outputs[split[0],0] = 1
  outputs[split[1],0] = 1
  outputs[split[2],1] = 1
  outputs[split[3],1] = 1
  return DataSet(inputs, outputs)

def xor_data():
  inputs = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]).astype('float32')
  outputs = np.array([[1], [-1], [1], [-1]]).astype('float32')
  return DataSet(inputs, outputs)

def spiral(n=100, r=2., s=3.):
  ''' spiral dataset '''
  split = np.array_split(np.arange(n), 2)
  # u = (r2-r1)/r2*np.random.rand(n) + r1/r2 
  u = np.random.rand(n)
  u = s*2*np.pi*u**(1./2.)
  r = r*u/s/2/np.pi
  i1 = r[split[0], np.newaxis]*np.vstack((np.cos(u[split[0]]), np.sin(u[split[0]]))).T
  i2 = r[split[1], np.newaxis]*np.vstack((-np.cos(u[split[1]]), -np.sin(u[split[1]]))).T
  inputs = np.concatenate((i1, i2), axis=0)
  outputs = np.zeros_like(inputs)
  outputs[split[0], 0] = 1
  outputs[split[1], 1] = 1
  return DataSet(inputs, outputs)



### REAL DATA
def mnist_data():
  from tensorflow.examples.tutorials.mnist import input_data
  mnist = input_data.read_data_sets('MNIST_data', one_hot=True, validation_size=10000)
  return DataSet(mnist.train.images, mnist.train.labels)

def mnist_data_val():
  from tensorflow.examples.tutorials.mnist import input_data
  mnist = input_data.read_data_sets('MNIST_data', one_hot=True, validation_size=10000)
  return DataSet(mnist.validation.images, mnist.validation.labels)

def mnist_data_test():
  from tensorflow.examples.tutorials.mnist import input_data
  mnist = input_data.read_data_sets('MNIST_data', one_hot=True, validation_size=10000)
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
def unpickle(file):
  """because nobody likes pickles"""
  import cPickle
  fo = open(file, 'rb')
  dict = cPickle.load(fo)
  fo.close()
  return dict

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

def combine_and_label(datalist):
  """
    same as comebine_data but assigns one-hot output labels based on which dataset the data came from
  """
  return None

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




# def maybe_download(filename, work_directory, source_url):
#   """From tensorflow mnist tutorial """
#   """Download the data from source url, unless it's already here."""
#   if not tf.gfile.Exists(work_directory):
#     tf.gfile.MakeDirs(work_directory)
#   filepath = os.path.join(work_directory, filename)
#   if not tf.gfile.Exists(filepath):
#     with tempfile.NamedTemporaryFile() as tmpfile:
#       temp_file_name = tmpfile.name
#       urllib.request.urlretrieve(source_url, temp_file_name)
#       tf.gfile.Copy(temp_file_name, filepath)
#       with tf.gfile.GFile(filepath) as f:
#         size = f.Size()
#       print('Successfully downloaded', filename, size, 'bytes.')
#   return filepath

# def _read32(bytestream):
#   dt = np.dtype(np.uint32).newbyteorder('>')
#   return np.frombuffer(bytestream.read(4), dtype=dt)[0]

# def extract_images(filename):
#   """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
#   #print('Extracting', filename)
#   with tf.gfile.Open(filename, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
#     magic = _read32(bytestream)
#     if magic != 2051:
#       raise ValueError(
#           'Invalid magic number %d in MNIST image file: %s' %
#           (magic, filename))
#     num_images = _read32(bytestream)
#     rows = _read32(bytestream)
#     cols = _read32(bytestream)
#     buf = bytestream.read(rows * cols * num_images)
#     data = np.frombuffer(buf, dtype=np.uint8)
#     data = data.reshape(num_images, rows, cols, 1)
#     return data

# def extract_labels(filename, one_hot=False, num_classes=10):
#   """Extract the labels into a 1D uint8 numpy array [index]."""
#   #print('Extracting', filename)
#   with tf.gfile.Open(filename, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
#     magic = _read32(bytestream)
#     if magic != 2049:
#       raise ValueError(
#           'Invalid magic number %d in MNIST label file: %s' %
#           (magic, filename))
#     num_items = _read32(bytestream)
#     buf = bytestream.read(num_items)
#     labels = np.frombuffer(buf, dtype=np.uint8)
#     if one_hot:
#       return dense_to_one_hot(labels, num_classes)
#     return labels

# SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

# def read_data_sets(train_dir, validation_size=5000, one_hot=False):
#   def int_to_float(images):
#     images = images.astype(np.float32)
#     return np.multiply(images, 1.0 / 255.0)
#   class DataSets(object):
#     pass
#   data_sets = DataSets()

#   TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
#   TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
#   TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
#   TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

#   local_file = maybe_download(TRAIN_IMAGES, train_dir, SOURCE_URL + TRAIN_IMAGES)
#   train_images = extract_images(local_file)

#   local_file = maybe_download(TRAIN_LABELS, train_dir, SOURCE_URL + TRAIN_LABELS)
#   train_labels = extract_labels(local_file, one_hot=one_hot)

#   local_file = maybe_download(TEST_IMAGES, train_dir, SOURCE_URL + TEST_IMAGES)
#   test_images = extract_images(local_file)

#   local_file = maybe_download(TEST_LABELS, train_dir, SOURCE_URL + TEST_LABELS)
#   test_labels = extract_labels(local_file, one_hot=one_hot)

#   train_images = int_to_float(train_images)
#   test_images = int_to_float(test_images)

#   validation_images = train_images[:validation_size]
#   validation_labels = train_labels[:validation_size]
#   train_images = train_images[validation_size:]
#   train_labels = train_labels[validation_size:]

#   data_sets.train_images = train_images
#   data_sets.train_labels = train_labels

#   data_sets.validation_images = validation_images
#   data_sets.validation_labels = validation_labels

#   data_sets.test_images = test_images
#   data_sets.test_labels = test_labels

#   return data_sets

