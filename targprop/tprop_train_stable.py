"""
  trying to stabilize with duals, etc.

  target propagation.

    layer:     [  0  ]    [          1           ]    [          2           ]    [   3  ]    [   top    ]
    network:   [input] -> [affine -> nonlinearity] -> [affine -> nonlinearity] -> [affine] -> [loss layer]
  
  where [loss layer] = [softmax -> cross_entropy], [sigmoid -> cross_entropy], [sigmoid -> MSE], or just [MSE]
  i.e. the last nonlinearity is part of the [loss layer], not layer 3, and layer 3 is just affine.

  TODO: make sure everything is float32
  TODO: make_top_top should be consistent with other make_tf* functions in how it returns python variables containing tf objects, namescopes, etc.
  TODO: check dcgan paper for implementation of top layer loss...
"""
import os
import numpy as np
import targprop.datasets as ds
import targprop.operations as ops
from scipy import linalg
import tensorflow as tf

def make_dir(path):
  """ like os.makedirs(path) but avoids race conditions """
  try:
    os.makedirs(path)
  except OSError:
    if not os.path.isdir(path):
      raise

def make_tf_L(layer, W_shape, b_shape, lr, act=tf.nn.tanh):
  """
    TODO: implement initialization as input option
    builds graph for layer-local training of W and b
    args:
      layer (int): which layer
      W_shape:
      b_shape:
      lr: learning rate
      act: activation function
    returns:
      training op
      merged summaries of this layer
  """
  with tf.name_scope('layer'+str(layer)+'_ff') as scope:

    W = tf.get_variable(scope+'W', shape=W_shape, dtype=tf.float32, initializer=tf.orthogonal_initializer(0.95))
    #W = tf.get_variable(scope+'W', shape=W_shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
    b = tf.get_variable(scope+'b', shape=b_shape, dtype=tf.float32, initializer=tf.constant_initializer(0.))

    x_0 = tf.placeholder(tf.float32, shape=[None, W_shape[0]], name='input')
    y   = tf.placeholder(tf.float32, shape=[None, W_shape[1]], name='output')
    
    loss = 0.5*tf.reduce_mean((act(tf.matmul(x_0, W) + b) - y)**2, name='loss') 
    
    s1 = tf.summary.scalar('log_loss'+str(layer), tf.log(loss))
    s2 = tf.summary.histogram('W'+str(layer), W)
    s3 = tf.summary.histogram('b'+str(layer), b) 
    
    # opt = tf.train.RMSPropOptimizer(lr) # rmsprop works *way* better than adam for local loss functions. unclear why.
    opt = tf.train.GradientDescentOptimizer(lr) # rmsprop works *way* better than adam for local loss functions. unclear why.
    gvs = opt.compute_gradients(loss, var_list=[W, b])
    sg  = [tf.summary.scalar('norm_grad'+var.name[-3], tf.nn.l2_loss(grad)) for grad, var in gvs] # var.name = 'namescope/V:0' and we want just 'V'
    clipped_gvs = [(tf.clip_by_norm(grad, 100.), var) for grad, var in gvs] # hmmmmmm. clip by norm value?
    
    return opt.apply_gradients(clipped_gvs), tf.summary.merge([s1] + sg)

def make_tf_Linv(layer, V_shape, c_shape, lr, act=tf.nn.tanh):
  """ builds graph for layer-local training of V and c """
  with tf.name_scope('layer'+str(layer)+'_inv') as scope:

    V = tf.get_variable(scope+'V', shape=V_shape, dtype=tf.float32, initializer=tf.orthogonal_initializer(0.95))
    #V = tf.get_variable(scope+'V', shape=V_shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
    c = tf.get_variable(scope+'c', shape=c_shape, dtype=tf.float32, initializer=tf.constant_initializer(0.))
    
    W = tf.placeholder(tf.float32, shape=[V_shape[1], V_shape[0]], name='W')
    b = tf.placeholder(tf.float32, shape=[1, V_shape[0]], name='b')
    x_0 = tf.placeholder(tf.float32, shape=[None, V_shape[1]], name='input')
    
    fx = act(tf.matmul(x_0, W) + b)
    loss = 0.5*tf.reduce_mean((act(tf.matmul(fx, V) + c) - x_0)**2, name='loss')  
    
    s1 = tf.summary.scalar('log_loss'+str(layer), tf.log(loss))
    s2 = tf.summary.histogram('V'+str(layer), V)
    s3 = tf.summary.histogram('c'+str(layer), c) 
    
    opt = tf.train.RMSPropOptimizer(lr)
    gvs = opt.compute_gradients(loss, var_list=[V, c])
    sg  = [tf.summary.scalar('norm_grad'+var.name[-3], tf.nn.l2_loss(grad)) for grad, var in gvs] # var.name = 'namescope/V:0' and we want just 'V'
    clipped_gvs = [(tf.clip_by_norm(grad, 100.), var) for grad, var in gvs]
    
    return opt.apply_gradients(clipped_gvs), tf.summary.merge([s1] + sg)

def make_tf_top(x_shape, loss='sigmoid_ce'):
  """
    builds the top layer, i.e. the loss layer. 
  """
  with tf.name_scope('top') as scope:
    x = tf.placeholder(tf.float32, shape=x_shape, name='input')
    y = tf.placeholder(tf.float32, shape=x_shape, name='output')

    if loss=='sigmoid_ce':
      L = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(x, y))
      correct_prediction = tf.equal(tf.round( tf.sigmoid(x) ), tf.round( y ))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      accuracy_summary = [tf.summary.scalar('accuracy', accuracy)]
    elif loss=='softmax_ce':
      L = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(x, y))
      correct_prediction = tf.equal(tf.argmax( tf.nn.softmax(x), 1 ), tf.argmax( y, 1 ))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      accuracy_summary = [tf.summary.scalar('accuracy', accuracy)]
    elif loss=='sigmoid_l2':
      L = tf.nn.l2_loss(tf.sigmoid(x) - y)
      accuracy = None
      accuracy_summary = []
    elif loss=='l2':
      L = tf.nn.l2_loss(x - y)
      accuracy = None
      accuracy_summary = []

    loss_summary = tf.summary.scalar('log_loss', tf.log(L))
    dx = tf.gradients(L, x)[0]

    return L, dx, tf.summary.merge([loss_summary] + accuracy_summary), accuracy


def train_net(batch_size=100,
              t_steps=200,
              l_dim=[100, 50, 5, 50, 100],
              activation='tanh',
              gamma=0.001,
              alpha_t=0.1,
              noise_str=0.1,
              err_alg=0,
              learning_rate=0.003,
              learning_rate_inv=0.003,
              learning_rate_rinv=0.1,
              num_steps_rinv=2,
              top_loss='sigmoid_ce',
              mode='autoencoder',
              dataset='mnist',
              SGD=True,
              preprocess=False,
              tb_path='/tmp/targprop/'):
  """
    Args:
      batch_size (int, > 0): the number of examples in each training batch
      t_steps (int, > 0): the number of training steps
      l_dim (list of ints): the layer dimensions
      activation (tanh, linear, sigmoid, relu): activation functions of network
      gamma (float, > 0): regularization parameter for regularized target prop
      alpha_t (float, (0, 1)): the 'learning rate' in target propagation, i.e. the top layer target is x - alpha_t* dL/dx
      err_alg (int, in [0, 1, 2, 3]): which error propagation algorithm to use
        0: backprop
        1: constrained least-squares target prop (essentially op-by-op difference target prop)
        2: regularized least-squares target prop (op-by-op)
        3: difference target prop using L_inv (close to a carbon copy of Lee et al)
      learning_rate (float, > 0): the learning rate in gradient descent.
      learning_rate_inv (float, > 0): the learning rate for L_inv if err_alg==3
      top_loss ('sigmoid_ce', softmax_ce', 'sigmoid_l2', 'l2'): the top-layer, defined by pre-loss nonlinearity and loss function
      mode ('autoencoder', 'classification'):
        'autoencoder': outputs are set to inputs
        'classification': outputs are set to labels
      dataset ('mnist', 'cifar'): which dataset to use. 
      SGD (bool): stochastic gradient descent. Should be True. False can be useful for debugging and seeing if algorithms converge on a single batch.
      preprocess (bool): preprocess the data with PCA + whitening. 
    Returns:
      output_dict
        output_dict['L']: list. loss for each training step
        output_dict['L_test']: float. loss for test data at final training step
        output_dict['accuracy']: accuracy of classification
        output_dict['accuracy_test']: accuracy on test set
        output_dict['actvs']: activations of last layer. for autoencoder mode. 
  """

  # data
  if dataset == 'cifar':
    data = ds.cifar10_data()
    data_test = ds.cifar10_data_test()
  elif dataset == 'mnist':
    data = ds.mnist_data()
    data_test = ds.mnist_data_test()
  else:
    # set train and test the same. change later.
    data = dataset
    data_test = dataset

  if preprocess:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1000, whiten=True)
    data.inputs = pca.fit_transform(data.inputs)
    data_test.inputs = pca.transform(data_test.inputs)

  # autoencoderify
  if mode == 'autoencoder':
    data.outputs = data.inputs
    data_test.outputs = data_test.inputs

  # model parameters / architecture
  m_dim = data.inputs.shape[1] # input dimension
  p_dim = data.outputs.shape[1] # output dimension
  
  l_dim = [m_dim] + l_dim + [p_dim] # layer dimensions
  layers = len(l_dim)-1

  # operations from operations.py
  lin = ops.linear()
  add = ops.addition()

  # set activation function
  if activation == 'tanh':
    tf_act = tf.nn.tanh
    op_act = ops.tanh()
  elif activation == 'linear':
    tf_act = tf.identity
    op_act = ops.identity()
  elif activation == 'sigmoid':
    tf_act = tf.nn.sigmoid
    op_act = ops.sigmoid()
  elif activation == 'relu':
    tf_act = tf.nn.relu
    op_act = ops.relu()

  # put activations in lists
  acts    = (layers+1)*[None] # activation functions
  tf_acts = (layers+1)*[None] # activation functions
  for l in range(1, layers):
    acts[l]    = op_act
    tf_acts[l] = tf_act
  acts[-1]    = ops.identity() # last activation function is just identity, so we can offload the pre-loss nonlinearity to the 'loss' layer
  tf_acts[-1] = tf.identity

  def nonlin_layer(x_in, W_in, b_in):
    return tf_act(tf.matmul(x_in, W_in) + b_in)
  def affine_layer(x_in, W_in, b_in):
    return tf.matmul(x_in, W_in) + b_in

  # put op functions in lists...
  f = (layers+1)*[None]
  for l in range(1, layers):
    f[l] = nonlin_layer
  f[-1] = affine_layer

  # initialize variable lists
  W = (layers+1)*[None] # forward weights
  b = (layers+1)*[None] # biases
  
  train_op_W = (layers+1)*[None]
  train_op_p = (layers+1)*[None]
  train_op_tx = (layers+1)*[None]

  summary_ops = (layers+1)*[None]
  
  # initialize activation lists
  x = (layers+1)*[None]
  tx = (layers+1)*[None]
  p = (layers+1)*[None]

  loss = (layers+1)*[None]
  tloss = (layers+1)*[None]
  ploss = (layers+1)*[None]



  # create tensorflow graph with layer-local loss functions
  tf.reset_default_graph()
  
  # placeholders
  x[0] = tf.placeholder(tf.float32, shape=[None, l_dim[0]], name='input')
  tx[-1] = tf.placeholder(tf.float32, shape=[None, l_dim[-1]], name='output') 

  in_shape = x[0].get_shape()

  # 0 layer stuff
  tx[0] = tf.get_variable('layer0_ffx_tar', shape=[batch_size, l_dim[0]], dtype=tf.float32, initializer=tf.constant_initializer(0.))

  loss[0] = 0.
  tloss[0] = 0.
  ploss[0] = 0.

  opt = tf.train.RMSPropOptimizer(learning_rate)

  for l in range(1, layers+1):
    with tf.name_scope('layer'+str(l)+'_ff') as scope:
      W[l] = tf.get_variable(scope+'W', shape=[l_dim[l-1], l_dim[l]], dtype=tf.float32, initializer=tf.orthogonal_initializer(0.95))
      b[l] = tf.get_variable(scope+'b', shape=[1, l_dim[l]], dtype=tf.float32, initializer=tf.constant_initializer(0.))
      x[l] = f[l](x[l-1], W[l], b[l])

      tx[l] = tf.get_variable(scope+'x_tar', shape=[batch_size, l_dim[l]], dtype=tf.float32, initializer=tf.constant_initializer(0.))
      p[l] = tf.get_variable(scope+'p', shape=[batch_size, l_dim[l]], dtype=tf.float32, initializer=tf.constant_initializer(0.))

      if l == layers:
        # loss[l] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=x[l], labels=tx[l]))
        # correct_prediction = tf.equal(tf.argmax( tf.nn.softmax(x[l]), 1 ), tf.argmax( tx[l], 1 ))
        loss[l] = 0.5*tf.reduce_mean( (x[l] - tx[l])**2. )
        correct_prediction = tf.equal(tf.argmax( x[l], 1 ), tf.argmax( tx[l], 1 ))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_summary = tf.summary.scalar('accuracy', accuracy)
      elif l < layers:
        loss[l] = 0.5*tf.reduce_mean((x[l] - tx[l])**2.)

      summary_ops[l] = tf.summary.scalar('log_loss'+str(l), tf.log(loss[l]))

      # target loss term
      tloss[l] = 0.5*gamma*tf.nn.l2_loss(f[l](tx[l-1], W[l], b[l]) - tx[l])

      # Lagrange multiplier term
      ploss[l] = tf.reduce_sum(tf.multiply(p[l], f[l](tx[l-1], W[l], b[l]) - tx[l]))

      train_op_W[l] = opt.minimize(loss[l] + tloss[l] + ploss[l], var_list=[W[l], b[l]])
      train_op_p[l] = tf.train.GradientDescentOptimizer(gamma).minimize(-ploss[l], var_list=[p[l]])

  for l in range(0, layers):
    train_op_tx[l] = opt.minimize(loss[l] + tloss[l] + tloss[l+1] + ploss[l] + ploss[l+1], var_list=[tx[l]])

  merged = tf.summary.merge_all()
  writer = tf.summary.FileWriter(tb_path)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  for t in range(t_steps+1):
    if SGD:
      x0, y = data.next_batch(batch_size)
    else:
      x0 = data.inputs[:batch_size]
      y = data.outputs[:batch_size]

    feed_dict = {x[0]: x0, tx[-1]: y}

    sess.run(train_op_tx[0], feed_dict=feed_dict)
    for l in range(1, layers):  
      sess.run(train_op_tx[l], feed_dict=feed_dict)
      sess.run(train_op_W[l], feed_dict=feed_dict)
    sess.run(train_op_W[-1], feed_dict=feed_dict)

    if t % 5 == 0:
      for l in range(1, layers+1):
        sess.run(train_op_p[l], feed_dict=feed_dict)

    if t % 1 == 0:
      writer.add_summary(sess.run(merged, feed_dict=feed_dict), t)
    if t % 20 == 0:
      print 'Iter: ', t, 'Loss, accuracy: ', sess.run([loss[-1], accuracy], feed_dict=feed_dict)

  # ( V ^__^) V   training complete   V (^__^ V )

  #feed_dict = {x[0]: data_test.inputs, tx[-1]: data_test.outputs}
  #L_test, accuracy_test = sess.run([loss[-1], accuracy], feed_dict=feed_dict)

  # prepare the output dictionary
  output_dict = {}
  #output_dict['L_test'] = L_test
  #output_dict['accuracy_test'] = accuracy_test

  # if mode == 'autoencoder':
  #   if top_loss == 'sigmoid_ce':
  #     output_dict['reconstruction'] = sess.run(tf.sigmoid(x3_test[-1][:20]))
  #   else:
  #     output_dict['reconstruction'] = x3_test[-1][:20] # save final layer activations (reconstructions)

  sess.close() # (= _ =) ..zzZZ

  return output_dict
