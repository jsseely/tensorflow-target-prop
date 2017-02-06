import numpy as np
import scipy as sp
import datasets as ds
import operations as ops
from tqdm import tqdm
from scipy import linalg

# Loss-related ops
def softmax(x):
  return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)
def cross_entropy(y, x):
  return np.sum(-y*np.log(softmax(x)), axis=1, keepdims=True).mean()
def mse(y, x):
  return np.mean((x - y)**2)

# clipping function
def clip(x, th):
  return np.piecewise(x, [x <= -th, x > -th, x >= th], [-th, lambda x_: x_, th])

def train_net(batch_size=100,
              t_steps=200,
              l_dim=[100, 50, 5, 50, 100],
              alpha=0.003,
              alpha_t=0.1,
              gamma=0.001,
              err_alg=0,
              mode='autoencoder',
              dataset='mnist',
              SGD=True,
              preprocess=False):
  '''
    Train 
    Args:
      batch_size (int, > 0): the number of examples in each training batch
      t_steps (int, > 0): the number of training steps
      l_dim (list of ints): the layer dimensions
      alpha (float, > 0): the learning rate in gradient descent.
      alpha_t (float, (0, 1)): the 'learning rate' in target propagation, i.e. the target is set to alpha_t times the distance between the estimated output and the actual output.
      gamma (float, > 0): regularization parameter for regularized target prop
      err_alg (int, in [0, 1, 2, 3]): which error propagation algorithm to use
        0: backprop
        1: constrained least-squares target prop (essentially difference target prop)
        2: regularized least-squares target prop
      mode ('autoencoder', 'classification'):
        'autoencoder': outputs are set to inputs, loss is MSE
        'classification': outputs are set to labels, loss is cross entropy
      dataset ('mnist', 'cifar'): which dataset to use. 
      SGD (bool): stochastic gradient descent. Should be True. False can be useful for debugging and seeing if algorithms converge on a single batch.
      preprocess (bool): preprocess the data with PCA + whitening. 
    Returns:
     L (list of floats): loss as a function of training step
     L_test (float): loss of test data
     actvs: output layer activations for visualizing reconstruction (autoencoder)
  '''

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

  # Model parameters
  m_dim = data.inputs.shape[1]
  p_dim = data.outputs.shape[1]
  
  l_dim = [m_dim] + l_dim + [p_dim]
  layers = len(l_dim)-1

  b_init = 0.0 # init for bias terms

  # Set up np.zero arrays for all variables.
  # Forward weights
  W  = np.zeros((layers+1), dtype=object) # weights
  dW = np.zeros((layers+1), dtype=object) # dl/dW
  # Biases
  b  = np.zeros((layers+1), dtype=object) # biases
  db = np.zeros((layers+1), dtype=object) # dl/db
  # Loss
  L = np.zeros((t_steps+1)) # loss
  if mode == 'classification':
    accuracy = np.zeros((t_steps+1)) # accuracy 

  # Loss for test data
  L_test = np.zeros((1))


  # Initialize variables
  for l in range(1, layers+1):
    np.random.seed(l) # different random seed for each layer. Otherwise, same. For replicability.
    low = -np.sqrt(6.0/(l_dim[l-1] + l_dim[l]))
    high = np.sqrt(6.0/(l_dim[l-1] + l_dim[l]))
    W[l] = np.random.uniform(low=low, high=high, size=(l_dim[l-1], l_dim[l])).astype('float32') # Xavier initialization
    if l_dim[l-1] >= l_dim[l]:
      W[l] = 0.95*linalg.orth(W[l]) # orth init + symmetry breaking noise
    b[l] = b_init*np.ones((1, l_dim[l])).astype('float32')

  if mode == 'autoencoder':
    # init with transposes
    for l in range(layers/2+1, layers+1):
      W[l] = W[layers+1-l].T

  # Activations
  x1 = np.zeros((layers+1), dtype=object) # x1 = W*x3
  x2 = np.zeros((layers+1), dtype=object) # x2 = x1 + b
  x3 = np.zeros((layers+1), dtype=object) # x3 = f(x2)

  # Test set activations
  x1_test = np.zeros((layers+1), dtype=object)
  x2_test = np.zeros((layers+1), dtype=object)
  x3_test = np.zeros((layers+1), dtype=object)

  # Errors / gradients
  dx3 = np.zeros((layers+1), dtype=object)
  dx2 = np.zeros((layers+1), dtype=object)
  dx1 = np.zeros((layers+1), dtype=object)

  # Targets
  tx3 = np.zeros((layers+1), dtype=object)
  tx2 = np.zeros((layers+1), dtype=object)
  tx1 = np.zeros((layers+1), dtype=object)

  # Operations
  lin = ops.linear()
  nln = ops.relu()
  add = ops.addition()

  # Training
  for t in tqdm(range(t_steps+1)):
    # Get data
    if SGD:
      x0, y = data.next_batch(batch_size)
    else:
      x0 = data.inputs[:batch_size]
      y = data.outputs[:batch_size]
    # STEP 1: Forward pass
    x3[0] = x0
    for l in range(1, layers+1):
      x1[l] = lin.f( x3[l-1], W[l] )
      x2[l] = add.f( x1[l],   b[l] )
      x3[l] = nln.f( x2[l]         ) #NOTE: not implementing top nonlin as softmax as in original dtp paper.
    if mode == 'autoencoder':
      L[t] = mse( y, x3[-1] )
    elif mode == 'classification':
      L[t] = cross_entropy( y, softmax(x3[-1]) )
      correct_prediction = np.equal(np.argmax(softmax(x3[-1]), axis=1), np.argmax(y, axis=1))
      accuracy[t] = np.mean(correct_prediction.astype('float'))

    # STEP 2: Backward pass
    # Top layer errors and targets
    dx3[-1] = x3[-1] - y # applies to both MSE and cross-entropy softmax
    tx3[-1] = x3[-1] - alpha_t*(x3[-1] - y) # top layer target
    for l in range(layers, 0, -1):
      if err_alg == 0:
        # Backprop
        dx2[l]   = nln.df( dx3[l], x2[l]         )
        dx1[l]   = add.df( dx2[l], x1[l],   b[l] )
        dx3[l-1] = lin.df( dx1[l], x3[l-1], W[l] )

      elif err_alg == 1:
        # least-squares target prop, i.e. difference target prop
        tx2[l]   = nln.f_inv( tx3[l], x2[l],        )
        tx1[l]   = add.f_inv( tx2[l], x1[l],   b[l] )
        tx3[l-1] = lin.f_inv( tx1[l], x3[l-1], W[l] )

        # Target prop 'errors'
        dx2[l]   = x2[l]   - tx2[l]
        dx1[l]   = x1[l]   - tx1[l]
        dx3[l-1] = x3[l-1] - tx3[l-1]

      elif err_alg == 2:
        # Regularized least-squares target prop
        tx2[l]   = nln.f_rinv( tx3[l], x2[l],         gamma )
        tx1[l]   = add.f_rinv( tx2[l], x1[l],   b[l], gamma )
        tx3[l-1] = lin.f_rinv( tx1[l], x3[l-1], W[l], gamma )

        # Target prop 'errors'
        dx2[l]   = x2[l]   - tx2[l]
        dx1[l]   = x1[l]   - tx1[l]
        dx3[l-1] = x3[l-1] - tx3[l-1]

    # STEP 3: Update variables
    # TODO: do this part in tf... 
    for l in range(1, layers+1):
      # Gradient descent
      dW[l] = -alpha*np.dot(x3[l-1].T, dx1[l])/batch_size
      db[l] = -alpha*np.mean(dx2[l], axis=0)
      # Clip gradients
      dW[l] = clip(dW[l], 1e4)
      db[l] = clip(db[l], 1e4)
      # Update variables
      W[l] = W[l] + dW[l]
      b[l] = b[l] + db[l]

  # After training, get test error
  x3_test[0] = data_test.inputs
  for l in range(1, layers+1):
    x1_test[l] = lin.f( x3_test[l-1], W[l] )
    x2_test[l] = add.f( x1_test[l],   b[l] )
    x3_test[l] = nln.f( x2_test[l]         )
  L_test = mse( data_test.outputs, x3_test[-1] )

  output_dict = {'L': L, 'L_test': L_test}
  if mode == 'autoencoder':
    # save final layer activations (reconstructions)
    output_dict['actvs'] = x3_test[-1][:20]
  elif mode == 'classification':
    # save accuracy of classification
    output_dict['accuracy'] = accuracy

  return output_dict

