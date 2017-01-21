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
              dtp_method=1):
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
  x_in = tf.placeholder(tf.float32, shape=[batch_size, m_dim]) # Input
  y = tf.placeholder(tf.float32, shape=[batch_size, p_dim]) # Output
  epoch = tf.placeholder(tf.float32, shape=None) # training iteration

  # TODO: see paper for best value
  noise_inj = 1./(1.+epoch/200.) # stddev

  # Initialize lists.
  b = (layers+1)*[None] 
  W = (layers+1)*[None]
  x = (layers+1)*[None]

  L = (layers+1)*[None] # Local layer loss for training W and b
  L_inv = (layers+1)*[None] # Local inverse loss for training V and c

  x_ = (layers+1)*[None] # targets
  V = (layers+1)*[None] # feedback matrix
  c = (layers+1)*[None] # feedback bias

  x_c = (layers+1)*[None] # x + noise
  fx_c = (layers+1)*[None] # f(x + noise)

  train_op_inv = (layers+1)*[None]
  train_op_L = (layers+1)*[None]

  # Variable creation
  # Forward graph
  for l in range(1, layers+1):
    with tf.name_scope('Layer_Forward'+str(l)):
      b[l] = tf.Variable(tf.constant(b_init, shape=[1, l_dim[l]]), name='b')
      #W[l] = tf.Variable(np.linalg.qr(np.random.randn(l_dim[l-1], l_dim[l]))[0].astype('float32'), name='W') #orthonormal initialization
      W[l] = tf.Variable(tf.truncated_normal([l_dim[l-1], l_dim[l]], stddev=np.sqrt(6./(l_dim[l-1]+l_dim[l]))), name='W') # Random initialization
  # Feedback graph
  for l in range(layers, 1, -1):
    with tf.name_scope('Layer_Feedback'+str(l)):
      if dtp_method==1:
        c[l] = tf.Variable(tf.constant(b_init, shape=[1, l_dim[l-1]]), name='c')
        V[l] = tf.Variable(tf.truncated_normal([l_dim[l], l_dim[l-1]], stddev=np.sqrt(6./(l_dim[l-1]+l_dim[l]))), name='V')
      if dtp_method==2:
        c[l] = tf.Variable(tf.constant(b_init, shape=[1, l_dim[l-1]]), name='c')
        V[l] = tf.Variable(tf.truncated_normal([l_dim[l]+l_dim[l-1], l_dim[l-1]], stddev=np.sqrt(6./(2*l_dim[l-1]+l_dim[l]))), name='V')

  # Feedforward functions
  def f(layer, inp):
    """map from layer layer-1 to layer; inp is the inp"""
    return tf.nn.tanh(tf.matmul(inp, W[layer]) + b[layer], name='f')
  def f_stop(layer, inp):
    """like f, but with stop_gradients on parameters"""
    return tf.nn.tanh(tf.matmul(inp, tf.stop_gradient(W[layer]) + tf.stop_gradient(b[layer])), name='f_stop')

  # Feedback functions
  def g(layer, inp):
    """map from layer layer to layer-1; inp is inp"""
    return tf.nn.tanh(tf.matmul(inp, V[layer]) + c[layer], name='g')
  def g_full(layer, input1, input2):
    """ generalized g. g(x_[layer], x[layer-1]) -> x_[layer-1] """
    return tf.nn.tanh(tf.matmul(tf.concat( 1, [input1, input2] ), V[layer]) + c[layer], name='g_full')

  # Forward propagation
  x[0] = x_in
  for l in range(1, layers+1):
    with tf.name_scope('Layer_Forward'+str(l)):
      x[l] = f(l, x[l-1])
  # Top layer loss / top layer target
  L[-1] = tf.reduce_mean(-tf.reduce_sum(y*tf.log(tf.nn.softmax(x[-1])), reduction_indices=[1]))
  x_[-1] = x[-1] - alpha*tf.gradients(L[-1], [x[-1]])[0]
  # Feedback propagation
  for l in range(layers, 1, -1):
    with tf.name_scope('Layer_Feedback'+str(l)):
      if dtp_method==1:
        x_[l-1] = x[l-1] - g(l, x[l]) + g(l, x_[l])
      if dtp_method==2:
        x_[l-1] = g_full(l, x_[l], x[l-1])

  # Corrupted targets
  for l in range(1, layers):
    with tf.name_scope('Corrupted'+str(l)):
      x_c[l] = tf.stop_gradient(tf.random_normal([1, l_dim[l]], mean=x[l], stddev=noise_inj), name='x_c')
      fx_c[l+1] = tf.stop_gradient(f(l+1, x_c[l]), name='fx_c')

  # Loss functions
  for l in range(1, layers):
    with tf.name_scope('L'+str(l)):
      # checked: both options equivalent
      #L[l] = tf.reduce_mean(0.5*(f(l, x[l-1]) - x_[l])**2, name='L')
      L[l] = tf.reduce_mean(0.5*(x[l] - x_[l])**2, name='L')
  for l in range(2, layers+1):
    if dtp_method==1:
      L_inv[l] = tf.reduce_mean(0.5*(g(l, fx_c[l]) - x_c[l-1])**2, name='L_inv')
    if dtp_method==2:
      L_inv[l] = tf.reduce_mean(0.5*(f(l, g_full(l, x_[l], x[l-1])) - x_[l])**2) + beta*tf.reduce_mean(0.5*(g_full(l, x_[l], x[l-1]) - x[l-1])**2) # triple check -- where to put beta, where to put reducee_means? 

  # Optimizers
  #opt = tf.train.AdamOptimizer(0.001)
  opt = tf.train.GradientDescentOptimizer(learning_rate)
  gate_gradients=opt.GATE_OP # yo, whats this?
  for l in range(1, layers+1):
    train_op_L[l] = opt.minimize(L[l], var_list=[W[l], b[l]], gate_gradients=gate_gradients)
  for l in range(2, layers+1):
    train_op_inv[l] = opt.minimize(L_inv[l], var_list=[V[l], c[l]], gate_gradients=gate_gradients)

  # Backprop. for reference
  #train_bp = opt.minimize(L[-1], var_list=[i for i in W+b if i is not None])

  correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(x[-1]), 1), tf.argmax(y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # clean up
  train_op_L = [i for i in train_op_L if i is not None]
  # OPTION 1
  train_op_inv = [i for i in train_op_inv if i is not None]
  # OPTION 2
  #train_op_C = [i for i in train_op_C if i is not None]

  # Tensorboard
  for l in range(layers+1):
    with tf.name_scope('Layer_loss'):
      if L[l] is not None:
        tf.summary.scalar('L'+str(l), L[l])
    with tf.name_scope('Layer_loss_inv'):
      if L_inv[l] is not None:
        tf.summary.scalar('L_inv'+str(l), L_inv[l])
    with tf.name_scope('targets'):
      if x_[l] is not None:
        tf.summary.scalar('x_target'+str(l), tf.reduce_mean(x_[l]))
  tf.summary.scalar('accuracy', accuracy)

  for varlist in ['W','b','V','c']:
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

    if i % 50 == 0:
      loss_val, summary_str, acc_val = sess.run([L[-1], merged_summary_op, accuracy], feed_dict=feed_dict)
      summary_writer.add_summary(summary_str, i)

    if i % 1000 == 0:
      print "iter:", "%04d" % (i), \
        "loss:", "{:.4f}".format(loss_val), \
        "accuracy:", "{:.4f}".format(acc_val)

  print "finished"
  sess.close()



