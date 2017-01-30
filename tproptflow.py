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
              beta0=0.,
              beta1=1.,
              beta2=0.,
              noise_str=0.5,
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

  # Params from conti_dtp.py -- unclear if this is one hyperparam search or the optimal one
  # alpha, L learning rate, L_inv learning rate, noise_inj
  # 0.327736332653, 0.0148893490317, 0.00501149118237, 0.359829566008

  ### DATA
  data = mnist_data()
  data_test = mnist_data_test()
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
  np.random.seed(1234)

  # placeholders
  x_in = tf.placeholder(tf.float32, shape=[None, m_dim], name='x_in') # Input
  y = tf.placeholder(tf.float32, shape=[None, p_dim], name='y') # Output
  epoch = tf.placeholder(tf.float32, shape=None, name='epoch') # training iteration

  # In dtp code: 0.5/(1 + epoch / 100)
  noise_inj = noise_str/(1.+epoch/200.) # stddev

  # Initialize lists.
  b = (layers+1)*[None] 
  W = (layers+1)*[None]
  x = (layers+1)*[None]

  L = (layers+1)*[None] # Local layer loss for training W and b
  L_inv = (layers+1)*[None] # Local inverse loss for training V and c
  L_inv0 = (layers+1)*[None]
  L_inv1 = (layers+1)*[None]
  L_inv2 = (layers+1)*[None]

  x_ = (layers+1)*[None] # targets
  V = (layers+1)*[None] # feedback matrix
  c = (layers+1)*[None] # feedback bias

  eps = (layers+1)*[None] # noise in L_inv term
  eps0 = (layers+1)*[None] # noise in L_inv term
  eps1 = (layers+1)*[None] # noise in L_inv term
  scope = (layers+1)*[None] # store some scopes!
  
  train_op_inv = (layers+1)*[None]
  train_op_L = (layers+1)*[None]

  from scipy import linalg
  # Let's init with numpy!
  for l in range(1, layers+1):
    low = -np.sqrt(6.0/(l_dim[l-1] + l_dim[l]))
    high = np.sqrt(6.0/(l_dim[l-1] + l_dim[l]))
    W[l] = np.random.uniform(low=low, high=high, size=(l_dim[l-1], l_dim[l])).astype('float32')
    #W[l] = linalg.orth(W[l]) # works because l_dim[l-1] > l_dim[l]
  for l in range(layers, 1, -1):
    if dtp_method==0 or dtp_method==1:
      V[l] = np.linalg.pinv(W[l])
    if dtp_method==2:
      pinv = np.linalg.pinv(W[l])
      V[l] = np.concatenate((pinv, np.eye(l_dim[l-1]) - np.dot(W[l], pinv)), axis=0).astype('float32')

  # Variable creation
  # Forward graph
  # weight initializer - Q: can I set each to w_in, or does that somehow make every weight have the same random inits?
  # w_in = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)
  # w_in = tf.orthogonal_initializer(0.5)
  # xavier:
  # tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)
  
  for l in range(1, layers+1):
    with tf.variable_scope('Layer'+str(l)) as scope[l]:
      b[l] = tf.get_variable( 'b', shape=[1, l_dim[l]], initializer=tf.constant_initializer(b_init))
      #W[l] = tf.get_variable( 'W', shape=[l_dim[l-1], l_dim[l]], initializer=W[l])
      W[l] = tf.get_variable( 'W', initializer=W[l])
  # Feedback graph
  for l in range(layers, 1, -1):
    with tf.variable_scope('Layer'+str(l)):
      if dtp_method==0 or dtp_method==1:
        c[l] = tf.get_variable( 'c', shape=[1, l_dim[l-1]], initializer=tf.constant_initializer(b_init))
        #V[l] = tf.get_variable( 'V', shape=[l_dim[l], l_dim[l-1]], initializer=V[l])
        V[l] = tf.get_variable( 'V', initializer=V[l])
      if dtp_method==2:
        c[l] = tf.get_variable( 'c', shape=[1, l_dim[l-1]],  initializer=tf.constant_initializer(b_init))
        #V[l] = tf.get_variable( 'V', shape=[l_dim[l]+l_dim[l-1], l_dim[l-1]], initializer=V[l])
        V[l] = tf.get_variable( 'V', initializer=V[l])

  # Feedforward functions
  def f(layer, inp, act=tf.nn.tanh):
    """map from layer layer-1 to layer; inp is the inp"""
    with tf.variable_scope('Layer'+str(layer), reuse=True):
      W_ = tf.get_variable( 'W' )
      b_ = tf.get_variable( 'b' )
    return act(tf.matmul(inp, W_) + b_, name='f'+str(layer))

  # Feedback functions
  def g(layer, inp, act=tf.nn.tanh):
    with tf.variable_scope('Layer'+str(layer), reuse=True):
      V_ = tf.get_variable( 'V' )
      c_ = tf.get_variable( 'c' )
    return act(tf.matmul(inp, V_) + c_, name='g'+str(layer))

  def g_full(layer, input1, input2, act=tf.nn.tanh):
    """ generalized g. g(x_[layer], x[layer-1]) -> x_[layer-1] """
    with tf.variable_scope('Layer'+str(layer), reuse=True):
      V_ = tf.get_variable( 'V' )
      c_ = tf.get_variable( 'c' )
    return act(tf.matmul(tf.concat( 1, [input1, input2] ), V_) + c_, name='g_full'+str(layer))

  def g_full2(layer, input1, input2, input3, act=tf.nn.tanh):
    """ generalized g. g(x_[layer], x[layer-1]) -> x_[layer-1] """
    with tf.variable_scope('Layer'+str(layer), reuse=True):
      V_ = tf.get_variable( 'V' )
      c_ = tf.get_variable( 'c' )
    return act(tf.matmul(tf.concat( 1, [input1, input2, input3] ), V_) + c_, name='g_full'+str(layer))

  # Forward propagation
  x[0] = x_in
  # all but top layer uses tanh
  act = tf.nn.tanh
  for l in range(1, layers):
    x[l] = f(l, x[l-1], act)
  # top layer uses softmax
  x[layers] = f(layers, x[layers-1], tf.nn.softmax)

  # Top layer loss / top layer target
  # L[-1] = tf.reduce_mean(-tf.reduce_sum(y*tf.log(x[-1]), reduction_indices=[1]), name='global_loss')
  # L[-1] = tf.nn.softmax_cross_entropy_with_logits(x[-1], y)
  # x_[-1] = tf.identity(x[-1] - alpha*tf.gradients(L[-1], [x[-1]])[0], name='xtar'+str(layers))
  
  # debugging, use mse?
  L[-1] = tf.reduce_mean(-tf.reduce_sum(y*tf.log(x[-1]), reduction_indices=[1]), name='global_loss')
  #L[-1] = tf.reduce_mean(0.5*(x[-1] - y)**2, name='global_loss')
  x_[-1] = x[-1] - alpha*(x[-1] - y)

  # Feedback propagation
  # TODO: get `targets` for backprop, for comparison.
  for l in range(layers, 1, -1):
    if dtp_method==0:
      # vanilla target prop
      x_[l-1] = g(l, x_[l], act)
    if dtp_method==1:
      # difference target prop
      x_[l-1] = x[l-1] - g(l, x[l], act) + g(l, x_[l], act)
    if dtp_method==2:
      # regularized target prop
      x_[l-1] = g_full(l, x_[l], x[l-1], act)

  # Errors for loss functions
  if dtp_method==1 or dtp_method==2:
    for l in range(1, layers+1):
      eps[l]  = noise_inj*tf.random_normal(tf.shape(x[l]), mean=0, stddev=1., name='eps'+str(l-1)) # uh, tf.shape(x[l-1]) right?
      eps0[l] = noise_inj*tf.random_normal(tf.shape(x[l]), mean=0, stddev=1., name='eps0'+str(l-1)) # uh, tf.shape(x[l-1]) right?
      eps1[l] = noise_inj*tf.random_normal(tf.shape(x[l]), mean=0, stddev=1., name='eps1'+str(l-1)) # uh, tf.shape(x[l-1]) right?
  # Loss functions
  for l in range(1, layers): # FOR NOW; LAYERS+1, BUT SHOULD BE LAYERS
    L[l] = tf.reduce_mean(0.5*(x[l] - x_[l])**2, name='L'+str(l))
  for l in range(2, layers+1):
    if dtp_method==0 or dtp_method==1:
      L_inv[l] = tf.reduce_mean(0.5*(g(l, f(l, x[l-1]+eps[l-1], act), act) - (x[l-1]+eps[l-1]))**2, name='L_inv'+str(l))
    if dtp_method==2:
      # L_inv0 - g as left inverse of f; regardless of what x_0 is, g should send f(x) to x. just use, for now, the activation x+eps
      L_inv0[l] = tf.reduce_mean(0.5*(g_full(l, f(l, x[l-1]+eps0[l-1], act), x[l-1], act) - (x[l-1]+eps0[l-1]))**2, name='L_inv0')
      # L_inv1 - g as the right invers of f; regardless of what x_0 is, f should send g(y) to y; make sure to use x_targ as y because that's what matters
      L_inv1[l] = tf.reduce_mean(0.5*(f(l, g_full(l, x_[l]+eps1[l], x[l-1], act), act) - (x_[l]+eps1[l]))**2, name='L_inv1')
      # L_inv2 - g should send y close to x_0
      L_inv2[l] = tf.reduce_mean(0.5*(g_full(l, x_[l], x[l-1], act) - x[l-1])**2, name='L_inv2')
      L_inv[l] = beta0*L_inv0[l] + beta1*L_inv1[l] + beta2*L_inv2[l]
      # L_inv[l] = tf.add(L_inv1[l], beta*L_inv2[l], name='L_inv')
      # L_inv[l] = tf.add(tf.reduce_mean(0.5*(f(l, g_full(l, x_[l]+eps[l], x[l], x[l-1])) - x_[l]-eps[l])**2), beta*tf.reduce_mean(0.5*(g_full(l, x_[l], x[l], x[l-1]) - x[l-1])**2), name='L_inv') # triple check -- where to put beta, where to put reduce_means? 

  # Optimizers
  #opt = tf.train.AdamOptimizer(0.001)
  #opt = tf.train.GradientDescentOptimizer(learning_rate)
  # gate_gradients=tf.train.AdamOptimizer(0.001).GATE_NONE # yo, whats this?
  for l in range(1, layers+1):
    train_op_L[l] = tf.train.RMSPropOptimizer(learning_rate, name='Opt'+str(l)).minimize(L[l], var_list=[W[l], b[l]])
  for l in range(2, layers+1):
    train_op_inv[l] = tf.train.RMSPropOptimizer(learning_rate, name='Opt_inv'+str(l)).minimize(L_inv[l], var_list=[V[l], c[l]])

  # Backprop. for reference
  #train_bp = opt.minimize(L[-1], var_list=[i for i in W+b if i is not None])

  correct_prediction = tf.equal(tf.argmax(x[-1], 1), tf.argmax(y,1)) # NOTE! normally, tf.nn.softmax(x[-1])
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # clean up
  train_op_L = [i for i in train_op_L if i is not None]
  train_op_inv = [i for i in train_op_inv if i is not None]

  # Tensorboard
  with tf.name_scope('key_summaries'):
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('global_loss', L[-1])
  with tf.name_scope('layer_losses'):
    for l in range(layers+1):
      if L[l] is not None:
        tf.summary.scalar('L'+str(l), L[l])
      if L_inv[l] is not None:
        tf.summary.scalar('L_inv'+str(l), L_inv[l])
      if L_inv1[l] is not None:
        tf.summary.scalar('L_inv_ratio'+str(l), L_inv1[l]/L_inv2[l])
      #if L_inv2[l] is not None:
      #  tf.summary.scalar('L_inv2'+str(l), L_inv2[l])
  with tf.name_scope('weights'):
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

    if i % 25 == 0:
      loss_val, summary_str, acc_val = sess.run([L[-1], merged_summary_op, accuracy], feed_dict=feed_dict)
      summary_writer.add_summary(summary_str, i)

    if i % 1000 == 0:
      x_test, y_test = data_test.inputs, data_test.outputs
      feed_dict = {x_in: x_test, y: y_test, epoch: i}
      loss_val_test, acc_val_test = sess.run([L[-1], accuracy], feed_dict=feed_dict)
      print "iter:", "%04d" % (i), \
        "| TRAINING ", \
        "loss:", "{:.4f}".format(loss_val), \
        "accuracy:", "{:.4f}".format(acc_val), \
        "| TEST ", \
        "loss:", "{:.4f}".format(loss_val_test), \
        "accuracy:", "{:.4f}".format(acc_val_test)

  print "finished"

  if return_sess:
    return sess
  else:
    sess.close()
    return

