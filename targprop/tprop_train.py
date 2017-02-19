"""
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

    #W = tf.get_variable(scope+'W', shape=W_shape, dtype=tf.float32, initializer=tf.orthogonal_initializer(0.95))
    W = tf.get_variable(scope+'W', shape=W_shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
    b = tf.get_variable(scope+'b', shape=b_shape, dtype=tf.float32, initializer=tf.constant_initializer(0.))

    x_0 = tf.placeholder(tf.float32, shape=[None, W_shape[0]], name='input')
    y   = tf.placeholder(tf.float32, shape=[None, W_shape[1]], name='output')
    
    loss = 0.5*tf.reduce_mean((act(tf.matmul(x_0, W) + b) - y)**2, name='loss') 
    
    s1 = tf.summary.scalar('loss'+str(layer), loss)
    s2 = tf.summary.histogram('W'+str(layer), W)
    s3 = tf.summary.histogram('b'+str(layer), b) 
    
    opt = tf.train.AdamOptimizer(lr)
    gvs = opt.compute_gradients(loss, var_list=[W, b])
    sg  = [tf.summary.scalar('norm_grad'+var.name[-3], tf.nn.l2_loss(grad)) for grad, var in gvs] # var.name = 'namescope/V:0' and we want just 'V'
    clipped_gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs] # hmmmmmm. clip by norm value?
    
    return opt.apply_gradients(clipped_gvs), tf.summary.merge([s1] + sg)

def make_tf_Linv(layer, V_shape, c_shape, lr, act=tf.nn.tanh):
  """ builds graph for layer-local training of V and c """
  with tf.name_scope('layer'+str(layer)+'_inv') as scope:

    #V = tf.get_variable(scope+'V', shape=V_shape, dtype=tf.float32, initializer=tf.orthogonal_initializer(0.95))
    V = tf.get_variable(scope+'V', shape=V_shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
    c = tf.get_variable(scope+'c', shape=c_shape, dtype=tf.float32, initializer=tf.constant_initializer(0.))
    
    W = tf.placeholder(tf.float32, shape=[V_shape[1], V_shape[0]], name='W')
    b = tf.placeholder(tf.float32, shape=[1, V_shape[0]], name='b')
    x_0 = tf.placeholder(tf.float32, shape=[None, V_shape[1]], name='input')
    
    fx = act(tf.matmul(x_0, W) + b)
    loss = 0.5*tf.reduce_mean((act(tf.matmul(fx, V) + c) - x_0)**2, name='loss')  
    
    s1 = tf.summary.scalar('loss'+str(layer), loss)
    s2 = tf.summary.histogram('V'+str(layer), V)
    s3 = tf.summary.histogram('c'+str(layer), c) 
    
    opt = tf.train.AdamOptimizer(lr)
    gvs = opt.compute_gradients(loss, var_list=[V, c])
    sg  = [tf.summary.scalar('norm_grad'+var.name[-3], tf.nn.l2_loss(grad)) for grad, var in gvs] # var.name = 'namescope/V:0' and we want just 'V'
    clipped_gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs]
    
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

    loss_summary = tf.summary.scalar('loss', L)
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

  # initialize variable lists
  W = (layers+1)*[None] # forward weights
  b = (layers+1)*[None] # biases

  if err_alg == 3:
    V = (layers+1)*[None]
    c = (layers+1)*[None]
    train_op_L_inv = (layers+1)*[None] # training op
    summary_ops_inv = (layers+1)*[None] # tensorboard
  
  # initialize train and summary op lists
  train_op_L = (layers+1)*[None] # training op
  summary_ops = (layers+1)*[None] # for tensorboard
  
  # create tensorflow graph with layer-local loss functions
  tf.reset_default_graph()
  # set random seed?
  for l in range(1, layers+1):
    train_op_L[l], summary_ops[l] = make_tf_L(l, W_shape=(l_dim[l-1], l_dim[l]), 
                                                 b_shape=(1, l_dim[l]),
                                                 lr=learning_rate,
                                                 act=tf_acts[l])
    if err_alg == 3 and l > 1:
      train_op_L_inv[l], summary_ops_inv[l] = make_tf_Linv(l, V_shape=(l_dim[l], l_dim[l-1]),
                                                              c_shape=(1, l_dim[l-1]),
                                                              lr=learning_rate_inv,
                                                              act=tf_act)
  
  # loss layer
  global_loss, dldx, global_summaries, global_accuracy = make_tf_top( x_shape=[None, l_dim[-1]], loss=top_loss )

  # initialize activation lists
  x1 = (layers+1)*[None] # x1 = W*x3
  x2 = (layers+1)*[None] # x2 = x1 + b
  x3 = (layers+1)*[None] # x3 = f(x2)

  # errors / gradients
  dx3 = (layers+1)*[None]
  dx2 = (layers+1)*[None]
  dx1 = (layers+1)*[None]

  # targets
  tx3 = (layers+1)*[None]
  tx2 = (layers+1)*[None]
  tx1 = (layers+1)*[None]

  # (  >__<)  start training (>__< )
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  summary_writer = tf.summary.FileWriter(tb_path)

  # get initialized variables
  for l in range(1, layers+1):
    nscope = 'layer'+str(l)+'_ff/'
    W[l], b[l] = sess.run([nscope+'W:0', nscope+'b:0'])
    if l > 1 and err_alg == 3:
      nscope = 'layer'+str(l)+'_inv/'
      V[l], c[l] = sess.run([nscope+'V:0', nscope+'c:0'])

  # training steps
  for t in range(t_steps+1):

    if err_alg==3:
      noise_inj = noise_str/(1.+t/100.) # std dev of noise in L_inv loss
    
    # get data
    if SGD:
      x0, y = data.next_batch(batch_size)
    else:
      x0 = data.inputs[:batch_size]
      y = data.outputs[:batch_size]
    
    # STEP 1: forward pass
    x3[0] = x0
    for l in range(1, layers+1):
      x1[l] =     lin.f( x3[l-1], W[l] )
      x2[l] =     add.f( x1[l],   b[l] )
      x3[l] = acts[l].f( x2[l]         ) 

    # loss layer
    dx3[-1] = sess.run(dldx, feed_dict={'top/input:0': x3[-1], 'top/output:0': y})

    # STEP 2: backward pass
    # top layer errors and targets
    tx3[-1] = x3[-1] - alpha_t*dx3[-1] # top layer target

    for l in range(layers, 0, -1):
      
      if err_alg == 0:
        # backprop
        dx2[l]   = acts[l].df( dx3[l], x2[l]         )
        dx1[l]   =     add.df( dx2[l], x1[l],   b[l] )
        dx3[l-1] =     lin.df( dx1[l], x3[l-1], W[l] )

        # backprop 'targets' 
        tx2[l] = x2[l] - dx2[l] # TODO: use learning_rate*dx2[l] or not?
        tx1[l] = x1[l] - dx1[l]
        tx3[l] = x3[l] - dx3[l]

      elif err_alg == 1:
        # least-squares target prop, i.e. op-by-op difference target prop
        tx2[l]   = acts[l].f_inv( tx3[l], x2[l],        )
        tx1[l]   =     add.f_inv( tx2[l], x1[l],   b[l] )
        tx3[l-1] =     lin.f_inv( tx1[l], x3[l-1], W[l] )

        # target prop 'errors'
        dx2[l]   = x2[l]   - tx2[l]
        dx1[l]   = x1[l]   - tx1[l]
        dx3[l-1] = x3[l-1] - tx3[l-1]

      elif err_alg == 2:
        # regularized least-squares target prop
        tx2[l]   = acts[l].f_rinv( tx3[l], x2[l],         gamma, lr=learning_rate_rinv, num_steps=num_steps_rinv )
        tx1[l]   =     add.f_rinv( tx2[l], x1[l],   b[l], gamma )
        tx3[l-1] =     lin.f_rinv( tx1[l], x3[l-1], W[l], gamma )

        # target prop 'errors'
        dx2[l]   = x2[l]   - tx2[l]
        dx1[l]   = x1[l]   - tx1[l]
        dx3[l-1] = x3[l-1] - tx3[l-1]

      elif err_alg == 3:
        # difference target propagation using tx[l-1] = x[l-1] - g(x[l]) + g(tx[l])
        if l>1:
          tx3[l-1] = x3[l-1] - op_act.f( add.f( lin.f(x3[l] , V[l]),  c[l]) ) + op_act.f( add.f( lin.f(tx3[l] , V[l]),  c[l]) )

    # STEP 3: update variables
    for l in range(1, layers+1):
        
      # first update V and c if err_alg==3
      if err_alg == 3 and l > 1:
        nscope = 'layer'+str(l)+'_inv/' # namescope
        feed_dict={nscope+'input:0': x3[l-1] + noise_inj*np.random.randn(*x3[l-1].shape),
                   nscope+'W:0': W[l],
                   nscope+'b:0': b[l]}
        sess.run(train_op_L_inv[l], feed_dict=feed_dict)
        V[l], c[l] = sess.run([nscope+'V:0', nscope+'c:0'])
        
        if t % 200 == 0: # tensorboard
            summary_str = sess.run(summary_ops_inv[l], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, t)
      
      # now update W and b
      nscope = 'layer'+str(l)+'_ff/'
      feed_dict={nscope+'input:0': x3[l-1],
                 nscope+'output:0': tx3[l]} # use tx2 if not using activation in L[l]... 
      sess.run(train_op_L[l], feed_dict=feed_dict)
      W[l], b[l] = sess.run([nscope+'W:0', nscope+'b:0'])

      if t % 200 == 0: # tensorboard
        summary_writer.add_summary(sess.run(summary_ops[l], feed_dict=feed_dict), t)

    # after one training step, save accuracy to tensorboard
    if t % 200 == 0:
      summary_writer.add_summary(sess.run(global_summaries, feed_dict={'top/input:0': x3[-1], 'top/output:0': y}), t)

    if t % 500 == 0:
      print 'Iter: ', t, 'Loss, accuracy: ', sess.run([global_loss, global_accuracy], feed_dict={'top/input:0': x3[-1], 'top/output:0': y})

  # ( V ^__^) V training complete V (^__^ V )

  # test set activations
  x1_test = (layers+1)*[None]
  x2_test = (layers+1)*[None]
  x3_test = (layers+1)*[None]

  # feedforward pass with test data
  x3_test[0] = data_test.inputs
  for l in range(1, layers+1):
    x1_test[l] =     lin.f( x3_test[l-1], W[l] )
    x2_test[l] =     add.f( x1_test[l],   b[l] )
    x3_test[l] = acts[l].f( x2_test[l]         )

  L_test, accuracy_test = sess.run([global_loss, global_accuracy], feed_dict={'top/input:0': x3_test[-1], 'top/output:0': data_test.outputs})

  # prepare the output dictionary
  output_dict = {}
  output_dict['L_test'] = L_test
  output_dict['accuracy_test'] = accuracy_test

  if mode == 'autoencoder':
    if top_loss == 'sigmoid_ce':
      output_dict['acvts'] = sess.run(tf.sigmoid(x3_test[-1][:20]))
    else:
      output_dict['actvs'] = x3_test[-1][:20] # save final layer activations (reconstructions)


  sess.close() # (= _ =) ..zzZZ

  return output_dict
