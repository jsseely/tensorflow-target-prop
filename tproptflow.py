"""
  tensorflow implementation of difference target propagation
"""

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

def run_tprop(batch_size=100,
              t_steps=100,
              layers=5,
              alpha=0.1,
              beta=0,
              learning_rate=0.01,
              dtp_method=1,
              return_sess=False):
  """
    TODO:
      so much

    Args:
      batch_size: batch size
      t_steps: number of training steps
      layers: number of layers in the network
      alpha: in (0,1], scaling for top layer target;  x_tar[-1] = x[-1] - alpha*(dL/dx[-1])
      beta: regularization constant
      dtp_method: 1 or 2. 1 for Lee implementation, 2 for new implementation
  """

  ### DATA
  data = mnist_data()
  #data = xor_data()

  ### MODEL PARAMETERS
  # Layer naming convention
  # Layer 0: input vector space
  # Layer P: output vector space (same dimension as data.output)
  # Convention: "layers = P", so that len(l_dim)=P+1
  # The loss computes the error between x and y at layer P. The loss is not considered its own 'layer' 

  m_dim = data.inputs.shape[1]
  p_dim = data.outputs.shape[1]

  l_dim = [m_dim] + (layers-1)*[100] + [p_dim]
  stddev = 0.05 # noise for weight inits
  b_init = 0.0 # init for bias terms

  ### MODEL
  tf.reset_default_graph()
  tf.set_random_seed(1234)

  # placeholders
  x_in = tf.placeholder(tf.float32, shape=[batch_size, m_dim], name='x_in') # Input
  y = tf.placeholder(tf.float32, shape=[batch_size, p_dim], name='y') # Output
  epoch = tf.placeholder(tf.float32, shape=None, name='epoch') # training iteration

  # TODO: see paper for best value
  noise_inj = 0.1/(1.+epoch/200.) # stddev

  # Initialize lists.
  b = (layers+1)*[None] 
  W = (layers+1)*[None]
  x = (layers+1)*[None]

  L = (layers+1)*[None] # Local layer loss for training W and b
  L_inv = (layers+1)*[None] # Local inverse loss for training V and c

  x_ = (layers+1)*[None] # targets
  V = (layers+1)*[None] # feedback matrix
  c = (layers+1)*[None] # feedback bias

  eps = (layers+1)*[None] # noise in L_inv term
  scope = (layers+1)*[None] # store some scopes!
  
  train_op_inv = (layers+1)*[None]
  train_op_L = (layers+1)*[None]

  # Variable creation
  # Forward graph
  for l in range(1, layers+1):
    with tf.variable_scope('Layer'+str(l)) as scope[l]:
      b[l] = tf.get_variable( 'b', [1, l_dim[l]], tf.float32, tf.constant_initializer(b_init))
      W[l] = tf.get_variable( 'W', [l_dim[l-1], l_dim[l]], tf.float32, tf.orthogonal_initializer(0.5))
  # Feedback graph
  for l in range(layers, 1, -1):
    with tf.variable_scope('Layer'+str(l)):
      if dtp_method==1:
        c[l] = tf.get_variable( 'c', [1, l_dim[l-1]], tf.float32, tf.constant_initializer(b_init))
        V[l] = tf.get_variable( 'V', [l_dim[l], l_dim[l-1]], tf.float32, tf.orthogonal_initializer(0.5))
      if dtp_method==2:
        c[l] = tf.get_variable( 'c', [1, l_dim[l-1]], tf.float32, tf.constant_initializer(b_init))
        V[l] = tf.get_variable( 'V', [l_dim[l]+l_dim[l-1], l_dim[l-1]], tf.float32, tf.orthogonal_initializer(0.5))

  # Feedforward functions
  def f(layer, inp):
    """map from layer layer-1 to layer; inp is the inp"""
    with tf.variable_scope('Layer'+str(layer), reuse=True):
      W_ = tf.get_variable( 'W' )
      b_ = tf.get_variable( 'b' )
    return tf.nn.tanh(tf.matmul(inp, W_) + b_, name='f'+str(layer)) # debug: switch to identity...

  # Feedback functions
  def g(layer, inp):
    with tf.variable_scope('Layer'+str(layer), reuse=True):
      V_ = tf.get_variable( 'V' )
      c_ = tf.get_variable( 'c' )
    return tf.nn.tanh(tf.matmul(inp, V_) + c_, name='g'+str(layer))

  def g_full(layer, input1, input2):
    """ generalized g. g(x_[layer], x[layer-1]) -> x_[layer-1] """
    with tf.variable_scope('Layer'+str(layer), reuse=True):
      V_ = tf.get_variable( 'V' )
      c_ = tf.get_variable( 'c' )
    return tf.nn.tanh(tf.matmul(tf.concat( 1, [input1, input2] ), V_) + c_, name='g_full'+str(layer))

  # Forward propagation
  x[0] = x_in
  for l in range(1, layers+1):
    x[l] = tf.identity(f(l, x[l-1]), name='x'+str(l)) # TODO: remove identites
    # TOP LAYER SHOULD BE SOFTMAX


  # Top layer loss / top layer target
  # TODO: TOP LOSS SHOULD NOT INCLUDE REDUCE_MEAN -- MUST SEND A SET OF TARGETS BACK, NOT THE AVERAGE!
  #L[-1] = tf.reduce_mean(-tf.reduce_sum(y*tf.log(tf.nn.softmax(x[-1])), reduction_indices=[1]), name='global_loss')
  # L[-1] = tf.nn.softmax_cross_entropy_with_logits(x[-1], y)
  #x_[-1] = tf.identity(x[-1] - alpha*tf.gradients(L[-1], [x[-1]])[0], name='xtar'+str(layers))

  # UGH
  #L[-1] = L[-1] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(x[-1], y))
  x_[-1] = x[-1] - alpha*(x[-1] - y)


  # Feedback propagation
  for l in range(layers, 1, -1):
    if dtp_method==1:
      x_[l-1] = tf.identity(x[l-1] - g(l, x[l]) + g(l, x_[l]), name='xtar'+str(l-1))
    if dtp_method==2:
      x_[l-1] = g_full(l, x_[l], x[l-1])

  # Loss functions
  for l in range(1, layers+1): # FOR NOW; LAYERS+1, BUT SHOULD BE LAYERS
    L[l] = tf.reduce_mean(0.5*(x[l] - x_[l])**2, name='L'+str(l))
  for l in range(2, layers+1):
    if dtp_method==1:
      eps[l-1] = noise_inj*tf.random_normal([batch_size, l_dim[l-1]], mean=0, stddev=1., name='eps'+str(l-1)) # TODO: NOISE_INJ MULT OR SET AS STDDEV??/
      L_inv[l] = tf.reduce_mean(0.5*(g(l, f(l, x[l-1] + eps[l-1])) - (x[l-1] + eps[l-1]))**2, name='L_inv'+str(l))
    if dtp_method==2:
      L_inv[l] = tf.add(tf.reduce_mean(0.5*(f(l, g_full(l, x_[l], x[l-1])) - x_[l])**2), beta*tf.reduce_mean(0.5*(g_full(l, x_[l], x[l-1]) - x[l-1])**2), name='L_inv') # triple check -- where to put beta, where to put reducee_means? 

  # Optimizers
  #opt = tf.train.AdamOptimizer(0.001)
  #opt = tf.train.GradientDescentOptimizer(learning_rate)
  gate_gradients=tf.train.GradientDescentOptimizer(1).GATE_NONE # yo, whats this?
  for l in range(1, layers+1):
    train_op_L[l] = tf.train.GradientDescentOptimizer(learning_rate).minimize(L[l], var_list=[W[l], b[l]], gate_gradients=gate_gradients)
  for l in range(2, layers+1):
    train_op_inv[l] = tf.train.GradientDescentOptimizer(learning_rate).minimize(L_inv[l], var_list=[V[l], c[l]], gate_gradients=gate_gradients)

  # Backprop. for reference
  #train_bp = opt.minimize(L[-1], var_list=[i for i in W+b if i is not None])

  correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(x[-1]), 1), tf.argmax(y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # clean up
  train_op_L = [i for i in train_op_L if i is not None]
  train_op_inv = [i for i in train_op_inv if i is not None]

  # Tensorboard
  with tf.name_scope('summaries'):
    tf.summary.scalar('accuracy', accuracy)
    for l in range(layers+1):
      if L[l] is not None:
        tf.summary.scalar('L'+str(l), L[l])
      if L_inv[l] is not None:
        tf.summary.scalar('L_inv'+str(l), L_inv[l])
      if x_[l] is not None:
        tf.summary.scalar('x_target'+str(l), x_[l][0,0])

    for varlist in ['W', 'V']:
      for iv, var in enumerate(eval(varlist)):
        if var is not None:
          tf.summary.histogram(varlist+str(iv), var)
    
    merged_summary_op = tf.summary.merge_all()

  #TODO: get prev run from directory/
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  make_dir('/tmp/targ-prop/')
  run = str(len(os.listdir('/tmp/targ-prop'))+1)
  print 'Run: '+run
  #summary_writer = tf.summary.FileWriter('/tmp/targ-prop/'+str(beta), sess.graph)
  summary_writer = tf.summary.FileWriter('/tmp/targ-prop/'+str(run), sess.graph)

  for i in range(t_steps):
    x_batch, y_batch = data.next_batch(batch_size)
    #x_batch, y_batch = data.inputs[:batch_size], data.outputs[:batch_size]
    feed_dict = {x_in: x_batch, y: y_batch, epoch: i}
    sess.run(train_op_inv, feed_dict=feed_dict)
    sess.run(train_op_L, feed_dict=feed_dict)
    #sess.run(train_bp, feed_dict=feed_dict)

    if i % 100 == 0:
      loss_val, summary_str, acc_val = sess.run([L[-1], merged_summary_op, accuracy], feed_dict=feed_dict)
      summary_writer.add_summary(summary_str, i)

    if i % 1000 == 0:
      print "iter:", "%04d" % (i), \
        "loss:", "{:.4f}".format(loss_val), \
        "accuracy:", "{:.4f}".format(acc_val)

  print "finished"

  if return_sess:
    return sess
  else:
    sess.close()
    return

