"""
  tensorflow implementation of difference target propagation
"""

import os
import numpy as np
import tensorflow as tf
import datasets as ds
import operations as ops

def make_dir(path):
  """
    like os.makedirs(path) but avoids race conditions
  """
  try:
    os.makedirs(path)
  except OSError:
    if not os.path.isdir(path):
      raise

def train_net(batch_size=100,
              t_steps=100,
              l_dim=8*[240],
              act=tf.nn.tanh,
              alpha=0.1,
              beta0=0.,
              beta1=1.,
              beta2=0.,
              noise_str=0.5,
              learning_rate=0.01,
              learning_rate_inv=0.01,
              err_alg=1,
              mode='autoencoder',
              dataset='mnist',
              preprocess=False,
              return_sess=False):
  """
    Args:
      batch_size: batch size
      t_steps: number of training steps
      l_dim: list of network architecture / dimension of 'hidden' layers, not including input and output layer.
      alpha: in (0,1], scaling for top layer target;  x_tar[-1] = x[-1] - alpha*(dL/dx[-1])
      beta0: regularization constant
      beta1: regularization constant
      beta2: regularization constant
      noise_str: value of standard dev of noise injected into neurons, but only for the L_inv loss functions, and for t_step=0 (decays through training)
      learning_rate: learning rate for optimization
      err_alg: error propagation method. 0 for difference target prop. 1 for regularized target prop. 2 for reg target prop with learnable inverses. 3 for backprop.
      mode: 'autoencoder' or 'classification'
      dataset: 'mnist' or 'cifar'
      preprocess: bool. PCA+whiten the data? Good for cifar but whatevs for mnist
      return_sess: should we return the tf session?
    Returns:
      sess: the tf session if return_sess is True
  """

  # Params from conti_dtp.py -- unclear if this is one hyperparam search or the optimal one
  # alpha, L learning rate, L_inv learning rate, noise_inj
  # 0.327736332653, 0.0148893490317, 0.00501149118237, 0.359829566008

  ### DATA ###
  if dataset == 'cifar':
    data = ds.cifar10_data()
    data_test = ds.cifar10_data_test()
  elif dataset == 'mnist':
    data = ds.mnist_data()
    data_test = ds.mnist_data_test()

  if preprocess:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1000, whiten=True)
    data.inputs = pca.fit_transform(data.inputs)
    data_test.inputs = pca.transform(data_test.inputs)

  if mode == 'autoencoder':
    # autoencoderify
    data.outputs = data.inputs
    data_test.outputs = data_test.inputs

  m_dim = data.inputs.shape[1] # input dimension
  p_dim = data.outputs.shape[1] # output dimension

  l_dim = [m_dim] + l_dim + [p_dim] # layer dimensions
  layers = len(l_dim)-1

  ### MODEL ###
  tf.reset_default_graph()
  tf.set_random_seed(1234)
  np.random.seed(1234)

  # placeholders
  x_in = tf.placeholder(tf.float32, shape=[None, m_dim], name='x_in') # Input
  y = tf.placeholder(tf.float32, shape=[None, p_dim], name='y') # Output
  epoch = tf.placeholder(tf.float32, shape=None, name='epoch') # training iteration

  # in dtp code, 0.5/(1 + epoch / 100)
  noise_inj = noise_str/(1.+epoch/100.) # std dev of noise in L_inv loss

  # initialize lists
  x = (layers+1)*[None] # activations
  W = (layers+1)*[None] # feedforward matrix
  b = (layers+1)*[None] # feedforward bias

  x_ = (layers+1)*[None] # targets
  V = (layers+1)*[None] # feedback matrix
  c = (layers+1)*[None] # feedback bias

  L = (layers+1)*[None] # local layer loss for training W and b
  L_inv = (layers+1)*[None] # local inverse loss for training V and c
  L_inv0 = (layers+1)*[None] # (testing)
  L_inv1 = (layers+1)*[None]
  L_inv2 = (layers+1)*[None]
  eps = (layers+1)*[None] # noise in L_inv term
  eps0 = (layers+1)*[None] # (testing)
  eps1 = (layers+1)*[None]

  vscope = (layers+1)*[None] # variable scopes

  train_op_L = (layers+1)*[None] # training op
  train_op_inv = (layers+1)*[None] # training op

  # init with numpy arrays
  from scipy import linalg
  for l in range(1, layers+1):
    low = -np.sqrt(6.0/(l_dim[l-1] + l_dim[l]))
    high = np.sqrt(6.0/(l_dim[l-1] + l_dim[l]))
    W[l] = np.random.uniform(low=low, high=high, size=(l_dim[l-1], l_dim[l])).astype('float32')
    if l_dim[l-1] >= l_dim[l]:
      W[l] = 1.0*linalg.orth(W[l])

  # transpose for autoencoder
  if mode == 'autoencoder':
    for l in range(layers/2+1, layers+1):
      W[l] = W[layers+1-l].T

  for l in range(layers, 1, -1):
    if err_alg==0 or err_alg==1:
      #V[l] = np.linalg.pinv(W[l])
      low = -np.sqrt(6.0/(l_dim[l-1] + l_dim[l]))
      high = np.sqrt(6.0/(l_dim[l-1] + l_dim[l]))
      V[l] = np.random.uniform(low=low, high=high, size=(l_dim[l], l_dim[l-1])).astype('float32')
      if l_dim[l] >= l_dim[l-1]:
        V[l] = 1.0*linalg.orth(V[l])
    if err_alg==2:
      pinv = np.linalg.pinv(W[l])
      V[l] = np.concatenate((pinv, np.eye(l_dim[l-1]) - np.dot(W[l], pinv)), axis=0).astype('float32')

  # Variable creation
  # xavier:
  # tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)
  # orth:
  # tf.orthogonal_initializer(0.5)

  # feedforward variables
  for l in range(1, layers+1):
    with tf.variable_scope('vars_Layer'+str(l)) as vscope[l]:
      b[l] = tf.get_variable( 'b', shape=[1, l_dim[l]], initializer=tf.constant_initializer(0.0))
      W[l] = tf.get_variable( 'W', shape=[l_dim[l-1], l_dim[l]], initializer=tf.orthogonal_initializer())
      #W[l] = tf.get_variable( 'W', initializer=W[l])

  # feedback variables
  for l in range(layers, 1, -1):
    with tf.variable_scope(vscope[l]):
      if err_alg==0 or err_alg==1:
        c[l] = tf.get_variable( 'c', shape=[1, l_dim[l-1]], initializer=tf.constant_initializer(0.0))
        V[l] = tf.get_variable( 'V', shape=[l_dim[l], l_dim[l-1]], initializer=tf.orthogonal_initializer())
        #V[l] = tf.get_variable( 'V', initializer=V[l])
      if err_alg==2:
        c[l] = tf.get_variable( 'c', shape=[1, l_dim[l-1]],  initializer=tf.constant_initializer(0.0))
        V[l] = tf.get_variable( 'V', shape=[l_dim[l]+l_dim[l-1], l_dim[l-1]], initializer=tf.orthogonal_initializer())
        #V[l] = tf.get_variable( 'V', initializer=V[l])

  # feedforward functions
  def f(layer, x_in, act=tf.nn.tanh):
    with tf.variable_scope(vscope[layer], reuse=True):
      # note: could also just use W[l] and b[l]
      W_ = tf.get_variable('W')
      b_ = tf.get_variable('b')
    return act(tf.add(tf.matmul(x_in, W_), b_), name='x')

  # Feedback functions
  def g(layer, x_target, act=tf.nn.tanh):
    with tf.variable_scope(vscope[layer], reuse=True):
      V_ = tf.get_variable('V')
      c_ = tf.get_variable('c')
    return act(tf.add(tf.matmul(x_target, V_), c_), name='x_')

  def g_dtp(layer, x1_target, x1_activation, x0_activation, act=tf.nn.tanh):
    with tf.variable_scope(vscope[layer], reuse=True):
      V_ = tf.get_variable('V')
      c_ = tf.get_variable('c')
    return tf.add(x0_activation,
                  tf.sub(act(tf.add(tf.matmul(x1_target,     V[layer], name='x3_'), c[layer], name='x2_'), name='x1_'),
                         act(tf.add(tf.matmul(x1_activation, V[layer], name='x3_'), c[layer], name='x2_'), name='x1_')), name='x_target')

  def g_rinv(layer, x1_target, x0_activation):
    with tf.variable_scope(vscope[layer], reuse=True):
      V_ = tf.get_variable('V')
      c_ = tf.get_variable('c')
    relu_inv = tf.py_func(ops.relu().f_inv, [x1_target, x0_activation], [tf.float32], name='x3_')[0]
    add_inv = tf.sub(relu_inv, b[layer], name='x2_')
    return tf.py_func(ops.linear().f_inv, [add_inv,  x0_activation, W[layer]], [tf.float32], name='x1_')[0]

  # TESTING
  # def g_full(layer, input1, input2, act=tf.nn.tanh):
  #   """ generalized g. g(x_[layer], x[layer-1]) -> x_[layer-1] """
  #   with tf.name_scope(scope[l]):
  #     V[layer] = tf.get_variable( 'V' )
  #     c[layer] = tf.get_variable( 'c' )
  #     return act(tf.matmul(tf.concat( 1, [input1, input2] ), V[layer]) + c[layer], name='g_full')

  # def g_full2(layer, input1, input2, input3, act=tf.nn.tanh):
  #   """ generalized g. g(x_[layer], x[layer-1]) -> x_[layer-1] """
  #   with tf.name_scope('Layer'):
  #     V[layer] = tf.get_variable( 'V' )
  #     c[layer] = tf.get_variable( 'c' )
  #   return act(tf.matmul(tf.concat( 1, [input1, input2, input3] ), V[layer]) + c[layer], name='g_full')
  # /TESTING

  # forward propagation
  x[0] = x_in
  for l in range(1, layers+1):
    with tf.name_scope('layer'+str(l)+'_ff'):
      if l==layers and mode=='classification':
        # last layer
        x[layers] = f(layers, x[layers-1], tf.nn.softmax)
      else:
        # other layers
        x[l] = f(l, x[l-1], act)

  # top layer loss / top layer target
  # L[-1] = tf.nn.softmax_cross_entropy_with_logits(x[-1], y)
  with tf.name_scope('top_layer'):
    if mode == 'classification':
      #L[-1] = tf.reduce_mean(-tf.reduce_sum(y*tf.log(x[-1] + 1e-10), reduction_indices=[1]), name='global_loss') # add 1e-10 so you don't get nan'd
      L[-1] = tf.reduce_mean((x[-1] - y)**2, name='global_loss')
    elif mode == 'autoencoder':
      L[-1] = tf.reduce_mean((x[-1] - y)**2, name='global_loss')
    x_[-1] = tf.sub(x[-1], alpha*(x[-1] - y), name='x_target_top') 

  # feedback propagation
  for l in range(layers, 1, -1):
    with tf.name_scope('layer'+str(l)+'_fb'):
      if err_alg==0:
        x_[l-1] = tf.add(x[l-1] - g(l, x[l], act), g(l, x_[l], act), name='x_target')
      if err_alg==1:
        x_[l-1] = g_rinv(l, x_[l], x[l-1])

  # noise terms for loss functions
  if err_alg==0 or err_alg==2:
    for l in range(1, layers+1):
      with tf.name_scope('layer'+str(l)+'_eps'):
        eps[l]  = tf.random_normal(tf.shape(x[l]), mean=0, stddev=noise_inj, name='eps'+str(l-1))
        #eps0[l] = noise_inj*tf.random_normal(tf.shape(x[l]), mean=0, stddev=1., name='eps0'+str(l-1)) # uh, tf.shape(x[l-1]) right?
        #eps1[l] = noise_inj*tf.random_normal(tf.shape(x[l]), mean=0, stddev=1., name='eps1'+str(l-1)) # uh, tf.shape(x[l-1]) right?

  # loss functions
  for l in range(1, layers): # FOR NOW; LAYERS+1, BUT SHOULD BE LAYERS
    with tf.name_scope('layer'+str(l)+'_loss'):
      if err_alg!=3:
        L[l] = tf.reduce_mean((x[l] - tf.stop_gradient(x_[l]))**2, name='Loss') # note: stop_gradients not necessary
  for l in range(2, layers+1):
    with tf.name_scope('layer'+str(l)+'_loss_inv'):
      if err_alg==0:
        L_inv[l] = tf.reduce_mean((g(l, tf.stop_gradient(f(l, x[l-1]+eps[l-1], act)), act) - tf.stop_gradient(x[l-1]+eps[l-1]))**2, name='L_inv')
      if err_alg==1:
        pass
      if err_alg==2:
        # STILL TESTING
        # L_inv0 - g as left inverse of f; regardless of what x_0 is, g should send f(x) to x. just use, for now, the activation x+eps
        L_inv0[l] = tf.reduce_mean((g_full(l, f(l, x[l-1]+eps0[l-1], act), x[l-1], act) - (x[l-1]+eps0[l-1]))**2, name='L_inv0')
        # L_inv1 - g as the right inverse of f; regardless of what x_0 is, f should send g(y) to y; make sure to use x_targ as y because that's what matters
        L_inv1[l] = tf.reduce_mean((f(l, g_full(l, x_[l]+eps1[l], x[l-1], act), act) - (x_[l]+eps1[l]))**2, name='L_inv1')
        # L_inv2 - g should send y close to x_0
        L_inv2[l] = tf.reduce_mean((g_full(l, x_[l], x[l-1], act) - x[l-1])**2, name='L_inv2')
        L_inv[l] = beta0*L_inv0[l] + beta1*L_inv1[l] + beta2*L_inv2[l]
        # L_inv[l] = tf.add(L_inv1[l], beta*L_inv2[l], name='L_inv')
        # L_inv[l] = tf.add(tf.reduce_mean(0.5*(f(l, g_full(l, x_[l]+eps[l], x[l], x[l-1])) - x_[l]-eps[l])**2), beta*tf.reduce_mean(0.5*(g_full(l, x_[l], x[l], x[l-1]) - x[l-1])**2), name='L_inv') # triple check -- where to put beta, where to put reduce_means? 

  # optimizers
  if err_alg!=3:
    for l in range(1, layers+1):
      with tf.name_scope('layer'+str(l)+'_opts'):
        train_op_L[l] = tf.train.RMSPropOptimizer(learning_rate, name='Opt').minimize(L[l], var_list=[W[l], b[l]])
  if err_alg==0 or err_alg==2:
    for l in range(2, layers+1):
      with tf.name_scope('layer'+str(l)+'_opts_inv'):
        train_op_inv[l] = tf.train.RMSPropOptimizer(learning_rate_inv, name='Opt_inv').minimize(L_inv[l], var_list=[V[l], c[l]])
  if err_alg==3:
    train_op_L[-1] = tf.train.RMSPropOptimizer(learning_rate, name='Opt').minimize(L[-1], var_list=[i for i in W+b if i is not None])

  if mode == 'classification':
    correct_prediction = tf.equal(tf.argmax(x[-1], 1), tf.argmax(y,1)) # note: normally, tf.nn.softmax(x[-1]), but we already softmax'd
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  elif mode == 'autoencoder':
    accuracy = tf.constant(0) # :(

  # clean up
  train_op_L = [i for i in train_op_L if i is not None]
  train_op_inv = [i for i in train_op_inv if i is not None]

  # tensorboard
  with tf.name_scope('key_summaries'):
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('global_loss', L[-1])
  with tf.name_scope('layer_losses'):
    for l in range(layers+1):
      if L[l] is not None:
        tf.summary.scalar('L'+str(l), L[l])
      if L_inv[l] is not None:
        tf.summary.scalar('L_inv'+str(l), L_inv[l])
  with tf.name_scope('weights'):
    for varlist in ['W', 'V', 'b', 'c']:
      for iv, var in enumerate(eval(varlist)):
        if var is not None:
          tf.summary.histogram(varlist+str(iv), var)
  with tf.name_scope('grads'):
    for varlist in ['W', 'b']:
      for iv, var in enumerate(eval(varlist)):
        if var is not None and L[iv] is not None:
          tf.summary.histogram('grad'+varlist+str(iv), tf.gradients(L[iv], [var])[0]) # does this actually recompute gradients? if so, whatevs
    for varlist in ['V', 'c']:
      for iv, var in enumerate(eval(varlist)):
        if var is not None and L_inv[iv] is not None:
          tf.summary.histogram('grad'+varlist+str(iv), tf.gradients(L_inv[iv], [var])[0])

  merged_summary_op = tf.summary.merge_all()

  ### TRAIN ###
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  make_dir('/tmp/targ-prop/')
  run = str(len(os.listdir('/tmp/targ-prop'))+1)
  print 'Run: '+run
  summary_writer = tf.summary.FileWriter('/tmp/targ-prop/'+str(run), sess.graph)

  for i in range(t_steps):
    x_batch, y_batch = data.next_batch(batch_size)
    feed_dict = {x_in: x_batch, y: y_batch, epoch: i}    
    sess.run(train_op_inv, feed_dict=feed_dict)
    sess.run(train_op_L, feed_dict=feed_dict)

    if i % 25 == 0:
      loss_val, summary_str, acc_val = sess.run([L[-1], merged_summary_op, accuracy], feed_dict=feed_dict)
      summary_writer.add_summary(summary_str, i)

    if i % 200 == 0:
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

