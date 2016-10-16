import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from toy_data import *
sys.path.insert(0, '/Users/jeff/Documents/Python/_projects/tdadl/')

def make_dir(path):
  """
    like os.makedirs(path) but avoids race conditions
  """
  try:
    os.makedirs(path)
  except OSError:
    if not os.path.isdir(path):
      raise

def run_tprop(beta=0):
  ## Get data from toy_data.py
  data = mnist_data()
  #data = xor_data()

  ## Model parameters
  m_dim = data.inputs.shape[1]
  p_dim = data.outputs.shape[1]

  # Layer naming convention
  # Layer 0: input vector space
  # Layer P: output vector space (same dimension as data.output)
  # Convention: "layers = P", so that len(l_dim)=P+1
  # The loss computes the error between x and y at layer P. The loss is not considered its own 'layer' 

  layers = 3
  l_dim = [m_dim] + (layers-1)*[240] + [p_dim]
  stddev = 0.05 # noise for weight inits
  b_init = 0.0 # init for bias terms
  alpha = 0.5 # x_tar[-1] = x[-1] - alpha*(dL/dx[-1])

  batch_size = 500

  ## Model
  tf.reset_default_graph()
  tf.set_random_seed(1234)

  # placeholders
  x_in = tf.placeholder(tf.float32, shape=[batch_size, m_dim]) # Input
  y = tf.placeholder(tf.float32, shape=[batch_size, p_dim]) # Output
  epoch = tf.placeholder(tf.float32, shape=None) # training iteration

  noise_inj = .1/(1.+epoch/200.) # stddev

  # Initialize lists.
  b = (layers+1)*[None] 
  W = (layers+1)*[None]
  x = (layers+1)*[None]

  L = (layers+1)*[None] # Local layer loss for training W and b
  L_inv = (layers+1)*[None] # Local inverse loss for training V and c
  C = (layers+1)*[None] # Local target loss for training x_target

  x_ = (layers+1)*[None] # targets
  V = (layers+1)*[None] # feedback matrix
  c = (layers+1)*[None] # feedback bias

  x_c = (layers+1)*[None] # x + noise
  fx_c = (layers+1)*[None] # f(x + noise)

  train_op_inv = (layers+1)*[None]
  train_op_L = (layers+1)*[None]
  train_op_C = (layers+1)*[None]

  def f(ll, zz):
    """map from layer ll-1 to ll"""
    return tf.nn.tanh(tf.matmul(zz, W[ll]) + b[ll], name='f')
  def f_stop(ll, zz):
    """like f, but with stop_gradients on parameters"""
    return tf.nn.tanh(tf.matmul(zz, tf.stop_gradient(W[ll]) + tf.stop_gradient(b[ll])), name='f_stop')

  def g(ll, zz):
    """map from layer ll to ll-1"""
    return tf.nn.tanh(tf.matmul(zz, V[ll]) + c[ll], name='g')

  # Forward graph
  x[0] = x_in
  for l in range(1, layers+1):
    with tf.name_scope('Layer_Forward'+str(l)):
      b[l] = tf.Variable(tf.constant(b_init, shape=[1, l_dim[l]]), name='b')
      #W[l] = tf.Variable(np.linalg.qr(np.random.randn(l_dim[l-1], l_dim[l]))[0].astype('float32'), name='W') #orthonormal initialization
      W[l] = tf.Variable(tf.truncated_normal([l_dim[l-1], l_dim[l]], stddev=np.sqrt(6./(l_dim[l-1]+l_dim[l]))), name='W') # Random initialization
      x[l] = f(l, x[l-1])

  # Top layer loss / top layer target
  L[-1] = tf.reduce_mean(-tf.reduce_sum(y*tf.log(tf.nn.softmax(x[-1])), reduction_indices=[1]))
  x_[-1] = x[-1] - alpha*tf.gradients(L[-1], [x[-1]])[0]

  # OPTION 1 Feedback graph
  # for l in range(layers, 1, -1):
  #   with tf.name_scope('Layer_Feedback'+str(l)):
  #     c[l] = tf.Variable(tf.constant(b_init, shape=[1, l_dim[l-1]]), name='c')
  #     V[l] = tf.Variable(tf.truncated_normal([l_dim[l], l_dim[l-1]], stddev=np.sqrt(6./(l_dim[l-1]+l_dim[l]))), name='V')
  #     x_[l-1] = x[l-1] - g(l, x[l]) + g(l, x_[l])

  # OPTION 2 Specify/optimize targets exactly.
  for l in range(layers-1, 0, -1):
    with tf.name_scope('Layer_targets'+str(l)):
      x_[l] = tf.Variable(tf.constant(0.1, shape=[batch_size, l_dim[l]]), name='x_target')

  # # Corrupted targets
  # for l in range(1, layers):
  #   with tf.name_scope('Corrupted'+str(l)):
  #     x_c[l] = tf.stop_gradient(tf.random_normal([1, l_dim[l]], mean=x[l], stddev=noise_inj), name='x_c')
  #     fx_c[l+1] = tf.stop_gradient(f(l+1, x_c[l]), name='fx_c')

  # Loss functions
  # OPTION 1
  for l in range(1, layers):
    with tf.name_scope('L'+str(l)):
      L[l] = tf.reduce_mean(0.5*(f(l, x[l-1]) - x_[l])**2, name='L')
  # OPTION 2
  #beta = 0
  for l in range(1, layers):
    with tf.name_scope('C'+str(l)):
      C[l] = tf.reduce_mean(0.5*(x_[l+1] - f(l+1, x_[l]))**2) + beta*tf.reduce_mean(0.5*(x[l] - x_[l])**2)
  # for i in range(2, layers+1):
  #   L_inv[i] = tf.reduce_mean(0.5*(g(i, fx_c[i]) - x_c[i-1])**2, name='L_inv')

  # Optimizers
  #opt = tf.train.AdamOptimizer(0.001)
  opt = tf.train.GradientDescentOptimizer(0.05)
  for l in range(1, layers+1):
    train_op_L[l] = tf.train.GradientDescentOptimizer(0.1).minimize(L[l], var_list=[W[l], b[l]])
  for l in range(1, layers): 
    train_op_C[l] = tf.train.GradientDescentOptimizer(0.1).minimize(C[l], var_list=[x_[l]])
  # for l in range(2, layers+1):
  #   train_op_inv[l] = opt.minimize(L_inv[l], var_list=[V[l], c[l]])

  # Backprop. for reference
  #train_bp = opt.minimize(L[-1], var_list=[i for i in W+b if i is not None])

  correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(x[-1]), 1), tf.argmax(y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # clean up
  train_op_L = [i for i in train_op_L if i is not None]
  #train_op_inv = [i for i in train_op_inv if i is not None]
  train_op_C = [i for i in train_op_C if i is not None]

  # Tensorboard
  for l in range(layers+1):
    if L[l] is not None:
      tf.scalar_summary('L'+str(l), L[l])
    if L_inv[l] is not None:
      tf.scalar_summary('L_inv'+str(l), L_inv[l])
  tf.scalar_summary('accuracy', accuracy)

  for var in tf.all_variables():
    tf.histogram_summary(var.name, var)
  merged_summary_op = tf.merge_all_summaries()

  #TODO: get prev run from directory/
  sess = tf.Session()
  sess.run(tf.initialize_all_variables())

  make_dir('/tmp/targ-prop/')
  run = len(os.listdir('/tmp/targ-prop'))+1
  summary_writer = tf.train.SummaryWriter('/tmp/targ-prop/'+str(beta), sess.graph)

  for i in range(1000):
    #x_batch, y_batch = data.rand_batch(batch_size)
    x_batch, y_batch = data.inputs[:batch_size], data.outputs[:batch_size]
    feed_dict = {x_in: x_batch, y: y_batch, epoch: i}
    #sess.run(train_op_inv, feed_dict=feed_dict)

    sess.run(train_op_C, feed_dict=feed_dict)
    sess.run(train_op_L, feed_dict=feed_dict)
    #sess.run(train_bp, feed_dict=feed_dict)

    if i % 10 == 0:
      loss_val, summary_str, acc_val = sess.run([L[-1], merged_summary_op, accuracy], feed_dict=feed_dict)
      summary_writer.add_summary(summary_str, i)

    if i % 100 == 0:
      print "iter:", "%04d" % (i), \
        "loss:", "{:.4f}".format(loss_val), \
        "accuracy:", "{:.4f}".format(acc_val)

  print "finished"



