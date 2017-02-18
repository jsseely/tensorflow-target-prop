"""
  target propagation.
  TODO: make sure everything is float32
  TODO: get_variable tf initializers...
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

def softmax(x):
  """softmax"""
  maxx = np.max(x, axis=1, keepdims=True) # prevents overflow, but not underflow
  return np.exp(x-maxx)/np.sum(np.exp(x-maxx), axis=1, keepdims=True)
def cross_entropy(y, x):
  """cross entropy"""
  return np.mean(-np.sum(y*np.log(x), axis=1, keepdims=True))
def mse(y, x):
  """mean-squared error"""
  return 0.5*np.mean((x - y)**2)

def clip(x, th):
  """piecewise linear clipping function"""
  return np.piecewise(x, [x <= -th, x > -th, x >= th], [-th, lambda x_: x_, th])

def make_tf_L(layer, W_init, b_init, lr, act=tf.nn.tanh):
  """ builds graph for layer-local training of W and b """
  with tf.name_scope('layer'+str(layer)+'_ff') as scope:
    # W = tf.Variable(W_init, name='W')
    # b = tf.Variable(b_init, name='b')

    W = tf.get_variable(scope+'W', shape=W_init.shape, dtype=tf.float32, initializer=tf.orthogonal_initializer(0.95))
    b = tf.get_variable(scope+'b', shape=b_init.shape, dtype=tf.float32, initializer=tf.constant_initializer(0.))

    x_0 = tf.placeholder(tf.float32, shape=[None, W_init.shape[0]], name='input')
    y   = tf.placeholder(tf.float32, shape=[None, W_init.shape[1]], name='output')
    
    loss = 0.5*tf.reduce_mean((act(tf.matmul(x_0, W) + b) - y)**2, name='loss') 
    
    s1 = tf.summary.scalar('loss'+str(layer), loss)
    s2 = tf.summary.histogram('W'+str(layer), W)
    s3 = tf.summary.histogram('b'+str(layer), b) 
    
    opt = tf.train.AdamOptimizer(lr)
    gvs = opt.compute_gradients(loss, var_list=[W, b])
    sg = [tf.summary.scalar('mean_grad'+var.name[-3], tf.reduce_mean(grad)) for grad, var in gvs] # var.name = 'namescope/V:0' and we want just 'V'
    sg += [tf.summary.scalar('norm_grad'+var.name[-3], tf.reduce_sum(grad**2)**0.5) for grad, var in gvs] # var.name = 'namescope/V:0' and we want just 'V'
    clipped_gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs] # hmmmmmm. clip by norm value?
    return opt.apply_gradients(clipped_gvs), tf.summary.merge([s1] + sg)

def make_tf_Linv(layer, V_init, c_init, lr, act=tf.nn.tanh):
  """ builds graph for layer-local training of V and c """
  with tf.name_scope('layer'+str(layer)+'_inv') as scope:
    # V = tf.Variable(V_init, name='V')
    # c = tf.Variable(c_init, name='c')

    V = tf.get_variable(scope+'V', shape=V_init.shape, dtype=tf.float32, initializer=tf.orthogonal_initializer(0.95))
    c = tf.get_variable(scope+'c', shape=c_init.shape, dtype=tf.float32, initializer=tf.constant_initializer(0.))
    
    W = tf.placeholder(tf.float32, shape=V_init.T.shape, name='W')
    b = tf.placeholder(tf.float32, shape=[1, V_init.shape[0]], name='b')
    x_0 = tf.placeholder(tf.float32, shape=[None, V_init.shape[1]], name='input')
    
    fx = act(tf.matmul(x_0, W) + b)
    loss = 0.5*tf.reduce_mean((act(tf.matmul(fx, V) + c) - x_0)**2, name='loss')  
    
    s1 = tf.summary.scalar('loss'+str(layer), loss)
    s2 = tf.summary.histogram('V'+str(layer), V)
    s3 = tf.summary.histogram('c'+str(layer), c) 
    
    opt = tf.train.AdamOptimizer(lr)
    gvs = opt.compute_gradients(loss, var_list=[V, c])
    sg = [tf.summary.scalar('mean_grad'+var.name[-3], tf.reduce_mean(grad)) for grad, var in gvs] # var.name = 'namescope/V:0' and we want just 'V'
    sg += [tf.summary.scalar('norm_grad'+var.name[-3], tf.reduce_sum(grad**2)**0.5) for grad, var in gvs] # var.name = 'namescope/V:0' and we want just 'V'
    clipped_gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs]
    return opt.apply_gradients(clipped_gvs), tf.summary.merge([s1] + sg)

def make_tf_top(loss='sigmoid_ce'):
  with tf.name_scope('top') as scope:
    x = tf.placeholder(tf.float32, shape=[batch_size, l_dim[-1]], name='input')
    y = tf.placeholder(tf.float32, shape=[batch_size, l_dim[-1]], name='output')

    if loss=='sigmoid_ce':
      L = tf.nn.sigmoid_cross_entropy_with_logits(x, y)
      correct_prediction = np.equal(np.argmax(tf.sigmoid(x[-1]), axis=1), np.argmax(y, axis=1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      accuracy_summary = [tf.summary.scalar('accuracy', accuracy)]
    elif loss=='softmax_ce':
      L = tf.nn.softmax_cross_entropy_with_logits(x, y)
      correct_prediction = np.equal(np.argmax(tf.nn.softmax(x[-1]), axis=1), np.argmax(y, axis=1))
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

    return L, dx, tf.merge([loss_summary] + accuracy_summary), accuracy



def train_net(batch_size=100,
              t_steps=200,
              l_dim=[100, 50, 5, 50, 100],
              act='tanh',
              gamma=0.001,
              alpha_t=0.1,
              noise_str=0.1,
              err_alg=0,
              learning_rate=0.003,
              learning_rate_inv=0.003,
              top_loss='sigmoid_ce',
              mode='autoencoder',
              dataset='mnist',
              update_implementation='numpy',
              SGD=True,
              preprocess=False,
              tb_path='/tmp/targprop/'):
  """
    Args:
      batch_size (int, > 0): the number of examples in each training batch
      t_steps (int, > 0): the number of training steps
      l_dim (list of ints): the layer dimensions
      act (tanh, linear, sigmoid, relu): activation functions of network
      gamma (float, > 0): regularization parameter for regularized target prop
      alpha_t (float, (0, 1)): the 'learning rate' in target propagation, i.e. the target is set to alpha_t times the distance between the estimated output and the actual output.
      err_alg (int, in [0, 1, 2, 3]): which error propagation algorithm to use
        0: backprop
        1: constrained least-squares target prop (essentially difference target prop)
        2: regularized least-squares target prop
        3: difference target prop using L_inv (Lee et al)
      learning_rate (float, > 0): the learning rate in gradient descent.
      learning_rate_inv (float, > 0): the learning rate for L_inv if err_alg==3
      mode ('autoencoder', 'classification'):
        'autoencoder': outputs are set to inputs, loss is MSE
        'classification': outputs are set to labels, loss is cross entropy
      dataset ('mnist', 'cifar'): which dataset to use. 
      update_implementation ('numpy', 'tf'): whether to update variables using numpy or tensorflow.
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
    pca = PCA(n_components=100, whiten=True)
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
  if act == 'tanh':
    tf_act = tf.nn.tanh
    nln = ops.tanh()
  elif act == 'linear':
    tf_act = tf.identity
    nln = ops.identity()
  elif act == 'sigmoid':
    tf_act = tf.nn.sigmoid
    nln = ops.sigmoid()
  elif act == 'relu':
    tf_act = tf.nn.relu
    nln = ops.relu()

  F   = (layers+1)*[None] # activation functions
  Ftf = (layers+1)*[None] # activation functions
  for l in range(1, layers):
    F[l]   = nln
    Ftf[l] = tf_act
  F[-1]   = ops.identity()
  Ftf[-1] = tf.identity

  # initialize lists
  W = (layers+1)*[None] # forward weights
  b = (layers+1)*[None] # biases

  b_init = 0.0 # init for bias terms

  if err_alg == 3:
    V = (layers+1)*[None]
    c = (layers+1)*[None]
    train_op_L_inv = (layers+1)*[None] # training op
    summary_ops_inv = (layers+1)*[None] # tensorboard

  train_op_L = (layers+1)*[None] # training op
  summary_ops = (layers+1)*[None] # for tensorboard

  # initialize variables
  for l in range(1, layers+1):
    np.random.seed(l) # different random seed for each layer. Otherwise, same. For replicability.
    low = -np.sqrt(6.0/(l_dim[l-1] + l_dim[l]))
    high = np.sqrt(6.0/(l_dim[l-1] + l_dim[l]))
    W[l] = np.random.uniform(low=low, high=high, size=(l_dim[l-1], l_dim[l])).astype('float32') # Xavier initialization
    # if l_dim[l-1] >= l_dim[l]:
    #   W[l] = 1.0*linalg.orth(W[l]) # orth init
    b[l] = b_init*np.ones((1, l_dim[l])).astype('float32')

  # if mode == 'autoencoder':
  #   # init with transposes
  #   for l in range(layers/2+1, layers+1):
  #     W[l] = W[layers+1-l].T

  if err_alg == 3:
    for l in range(layers, 1, -1):
      low = -np.sqrt(6.0/(l_dim[l-1] + l_dim[l]))
      high = np.sqrt(6.0/(l_dim[l-1] + l_dim[l]))
      V[l] = np.random.uniform(low=low, high=high, size=(l_dim[l], l_dim[l-1])).astype('float32') # Xavier initialization
      # if l_dim[l] >= l_dim[l-1]:
      #   V[l] = 1.0*linalg.orth(W[l]) # orth init
      #V[l] = np.linalg.pinv(W[l])
      c[l] = b_init*np.ones((1, l_dim[l-1])).astype('float32')

    # if mode == 'autoencoder':
    #   # init with transposes
    #   for l in range(layers/2+1, layers+1):
    #     V[layers+1-l] = V[l].T

  # create tensorflow graph with layer-local loss functions
  if update_implementation == 'tf':
    tf.reset_default_graph()
    for l in range(1, layers+1):
      train_op_L[l], summary_ops[l] = make_tf_L(l, W[l], b[l], learning_rate, Ftf[l])
      if err_alg == 3 and l > 1:
        train_op_L_inv[l], summary_ops_inv[l] = make_tf_Linv(l, V[l], c[l], learning_rate_inv, tf_act)
    
    # loss layer
    global_loss, dL_dx, global_summaries, global_accuracy = make_tf_top( top_loss )

    # add some basic summaries 
    #loss_sum = tf.summary.scalar('global_loss', tf.placeholder(tf.float32, shape=None, name='global_loss_placeholder'))
    #if mode == 'classification':
    #  acc_sum = tf.summary.scalar('accuracy', tf.placeholder(tf.float32, shape=None, name='accuracy_placeholder'))
    #  global_sums = tf.summary.merge((acc_sum, loss_sum))
    #elif mode == 'autoencoder':
    #  global_sums = tf.summary.merge((loss_sum, ))

  # initialize activation lists
  x1 = (layers+1)*[None] # x1 = W*x3
  x2 = (layers+1)*[None] # x2 = x1 + b
  x3 = (layers+1)*[None] # x3 = f(x2)

  # test set activations
  x1_test = (layers+1)*[None]
  x2_test = (layers+1)*[None]
  x3_test = (layers+1)*[None]

  # errors / gradients
  dx3 = (layers+1)*[None]
  dx2 = (layers+1)*[None]
  dx1 = (layers+1)*[None]

  # targets
  tx3 = (layers+1)*[None]
  tx2 = (layers+1)*[None]
  tx1 = (layers+1)*[None]

  # start training
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  summary_writer = tf.summary.FileWriter(tb_path)
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
    for l in range(1, layers+1): # up to but not including the last layer
      x1[l] =  lin.f( x3[l-1], W[l] )
      x2[l] =  add.f( x1[l],   b[l] )
      x3[l] = F[l].f( x2[l]         ) 

    # loss layer
    dx3[-1] = sess.run(dL_dx, feed_dict={'top_input:0': x3[-1], 'top_output:0': y})

    # STEP 2: backward pass
    # top layer errors and targets
    tx3[-1] = x3[-1] - alpha_t*dx3[-1] # top layer target

    for l in range(layers, 0, -1):
      
      if err_alg == 0:
        # backprop
        dx2[l]   = F[l].df( dx3[l], x2[l]         )
        dx1[l]   =  add.df( dx2[l], x1[l],   b[l] )
        dx3[l-1] =  lin.df( dx1[l], x3[l-1], W[l] )

        # backprop 'targets' 
        tx2[l] = x2[l] - dx2[l] # TODO: use learning_rate*dx2[l] or not?
        tx1[l] = x1[l] - dx1[l]
        tx3[l] = x3[l] - dx3[l]

      elif err_alg == 1:
        # least-squares target prop, i.e. op-by-op difference target prop
        tx2[l]   = F[l].f_inv( tx3[l], x2[l],        )
        tx1[l]   =  add.f_inv( tx2[l], x1[l],   b[l] )
        tx3[l-1] =  lin.f_inv( tx1[l], x3[l-1], W[l] )

        # target prop 'errors'
        dx2[l]   = x2[l]   - tx2[l]
        dx1[l]   = x1[l]   - tx1[l]
        dx3[l-1] = x3[l-1] - tx3[l-1]

      elif err_alg == 2:
        # regularized least-squares target prop
        tx2[l]   = F[l].f_rinv( tx3[l], x2[l],         gamma )
        tx1[l]   =  add.f_rinv( tx2[l], x1[l],   b[l], gamma )
        tx3[l-1] =  lin.f_rinv( tx1[l], x3[l-1], W[l], gamma )

        # target prop 'errors'
        dx2[l]   = x2[l]   - tx2[l]
        dx1[l]   = x1[l]   - tx1[l]
        dx3[l-1] = x3[l-1] - tx3[l-1]

      elif err_alg == 3:
        # difference target propagation using tx[l-1] = x[l-1] - g(x[l]) + g(tx[l])
        if l>1:
          tx3[l-1]   = x3[l-1] - nln.f( add.f( lin.f(x3[l] , V[l]),  c[l]) ) + nln.f( add.f( lin.f(tx3[l] , V[l]),  c[l]) )

    # STEP 3: update variables
    for l in range(1, layers+1):

      if update_implementation=='numpy':
        pass
        # # gradient descent
        # dW[l] = -learning_rate*np.dot(x3[l-1].T, dx1[l])/batch_size
        # db[l] = -learning_rate*np.mean(dx2[l], axis=0)
        # # clip gradients
        # dW[l] = clip(dW[l], 1e4)
        # db[l] = clip(db[l], 1e4)
        # # update variables
        # W[l] = W[l] + dW[l]
        # b[l] = b[l] + db[l]

      elif update_implementation=='tf':
        
        # first update V and c if err_alg==3
        if err_alg == 3 and l>1:
          nscope = 'layer'+str(l)+'_inv/' # namescope
          feed_dict={nscope+'input:0': x3[l-1] + noise_inj*np.random.randn(*x3[l-1].shape),
                     nscope+'W:0': W[l],
                     nscope+'b:0': b[l]}
          sess.run(train_op_L_inv[l], feed_dict=feed_dict)
          V[l], c[l] = sess.run([nscope+'V:0', nscope+'c:0'])
          
          if t % 100 == 0: # tensorboard
              summary_str = sess.run(summary_ops_inv[l], feed_dict=feed_dict)
              summary_writer.add_summary(summary_str, t)
        
        # now update W and b
        nscope = 'layer'+str(l)+'_ff/'
        feed_dict={nscope+'input:0': x3[l-1],
                   nscope+'output:0': tx3[l]} # use tx2 if not using activation in L[l]... 
        sess.run(train_op_L[l], feed_dict=feed_dict)
        W[l], b[l] = sess.run([nscope+'W:0', nscope+'b:0'])

        if t % 100 == 0: # tensorboard
          summary_writer.add_summary(sess.run(summary_ops[l], feed_dict=feed_dict), t)

    # after one training step, save accuracy to tensorboard
    if t % 100 == 0:
      summary_writer.add_summary(sess.run(global_summaries, feed_dict={'top_input:0': x3[-1], 'top_output:0': y}), t)

    if t % 500 == 0:
      print 'Iter: ', t, 'Loss: ', L[t]

  # ( V ^__^) V training complete V (^__^ V )

  # feedforward pass with test data
  x3_test[0] = data_test.inputs
  for l in range(1, layers+1):
    x1_test[l] =  lin.f( x3_test[l-1], W[l] )
    x2_test[l] =  add.f( x1_test[l],   b[l] )
    x3_test[l] = F[l].f( x2_test[l]         )

  L_test, accuracy_test = sess.run([global_loss, global_accuracy], feed_dict={'top_input:0': x3[-1], 'top_output:0': data_test.outputs})

  # test set loss and accuracy
  # if mode=='autoencoder':
  #   L_test = mse( data_test.outputs, x3_test[-1] )
  # elif mode=='classification':
  #   L_test = cross_entropy( data_test.outputs, softmax(x3_test[-1]) + 1e-10 )
  #   correct_prediction = np.equal(np.argmax(softmax(x3_test[-1]), axis=1), np.argmax(data_test.outputs, axis=1))
  #   accuracy_test = np.mean(correct_prediction.astype('float'))

  # prepare the output dictionary
  output_dict = {}
  #output_dict['L'] = L # too big
  output_dict['L_test'] = L_test
  output_dict['accuracy_test'] = accuracy_test

  if mode == 'autoencoder':
    output_dict['actvs'] = x3_test[-1][:20] # save final layer activations (reconstructions)
  # elif mode == 'classification':
    #output_dict['accuracy'] = accuracy # too big

  sess.close() # (= _ =) ..zzZZ
  return output_dict
