import numpy as np
import scipy as sp
from toy_data import *

# Activation function.
def nl(x, nonlinearity='tanh'):
  ''' forward pass '''
  if nonlinearity == 'tanh':
    return tanh(x)
  elif nonlinearity == 'sigmoid':
    return sigmoid(x)
  elif nonlinearity == 'relu':
    return relu(x)
def nl_inv(x, nonlinearity='tanh', th=1e-2):
  ''' generalized inverse '''
  if nonlinearity == 'tanh':
    return tanh_inv(x, th=th)
  elif nonlinearity == 'sigmoid':
    return sigmoid_inv(x, th=th)
  elif nonlinearity == 'relu': # no relu_inv
    return relu_rinv(x, 0, 0.01)
def nl_rinv(x, x0, nonlinearity='tanh', beta=0.1):
  ''' regulzarized inverse '''
  if nonlinearity == 'tanh': # no tanh_rinv
    return tanh_inv(x, th=0.01)
  elif nonlinearity == 'sigmoid': # no sigmoid_rinv
    return sigmoid_inv(x, th=0.01)
  elif nonlinearity == 'relu':
    return relu_rinv(x, x0=x0, beta=beta)
def dnl(x, nonlinearity='tanh'):
  ''' derivative '''
  if nonlinearity == 'tanh':
    return dtanh(x)
  elif nonlinearity == 'sigmoid':
    return dsigmoid(x)
  elif nonlinearity == 'relu':
    return drelu(x)

# Activation function operations.
def sigmoid(x):
  return 1. / (1. + np.exp(-x))
def dsigmoid(x):
  ''' derivative of sigmoid() '''
  return sigmoid(x)*(1.0 - sigmoid(x))
def sigmoid_inv(x, th=1e-2):
  ''' Inverse of sigmoid(). Input to sigmoid_inv() is clipped first
      to lie within the image of sigmoid(). Approximates a generalized
      inverse. '''
  x = np.piecewise(x, [x <= th, x > th, x >= (1-th)], [th, lambda x_: x_, 1-th])
  return -np.log(1./x - 1.)
def sigmoid_rinv(x):
  return None

def tanh(x):
  return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
def dtanh(x):
  ''' derivative of tanh() '''
  return 1. - tanh(x)**2
def tanh_inv(x, th=1e-2):
  ''' Inverse of tanh(). Input to tanh_inv() is clipped first
      to lie within the image of tanh(). Approximates a generalized
      inverse. '''
  x = np.piecewise(x, [x <= (-1+th), x > (-1+th), x >= (1-th)], [-1+th, lambda x_: x_, 1-th])
  return 0.5*np.log((1. + x)/(1. - x))
def tanh_rinv(x):
  return None

def relu(x):
  return x*(x>0)
def drelu(x):
  return (x>0).astype('float')
def relu_inv(x):
  return None
def relu_rinv(y, x0, beta=0.1):
  """
    implementation of relu regularized inverse
    returns the solution to
      argmin_x C(x)
    where
      C(x) = (relu(x) - y)**2 + beta*(x - x0)**2
    Solves by setting dC/dx = 0. When there are two possible solutions,
    compute C(x) explicitly for both and take the smaller one.
  """
  def cost(x):
    return (relu(x)-y)**2 + beta*(x-x0)**2
  x_out = np.zeros(y.shape)
  x_1 = 1./(1.+beta)*(beta*x0 + y) # possible solution 1
  x_2 = x0 # possible solution 2 -- only viable if beta is not 0
  costs = np.array([cost(x_1), cost(x_2)])
  case_1 = (x_1 > 0)*np.logical_not(x_2 < 0)
  case_2 = (x_2 < 0)*np.logical_not(x_1 > 0)
  case_3 = (x_1 > 0)*(x_2 < 0)
  inds = np.argmin(costs, axis=0)
  x_3 = (inds==0)*x_1 + (inds==1)*x_2
  x_out = case_1*x_1 + case_2*x_2 + case_3*x_3
  return x_out

# Matrix multiplication operations
def matmul(x, W):
  return np.dot(x, W)
def matmul_inv(x, W):
  return np.dot(x, np.linalg.pinv(W))
def matmul_pinv(x, W, rcond = 1e-2):
  return np.dot(x, np.linalg.pinv(W, rcond=rcond))
def matmul_rinv(x, y, W, beta):
  return np.dot(np.linalg.pinv(beta*np.eye(x.shape[1]) + np.dot(x.T, x), rcond=1e-4), np.dot(x.T, y) + beta*W)

# Add/subtract ops (being overly explicit)
# TODO: add_rinv
def add(x, b):
  return x + b
def add_inv(x, b):
  return x - b
def add_rinv(x, y, b, beta):
  return (y - x + beta*b)/(1. + beta)

# Loss-related ops
def softmax(x):
  return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)
def cross_entropy(y, x):
  return np.sum(-y*np.log(softmax(x)), axis=1, keepdims=True).mean()
def mse(y, x):
  return np.mean((x - y)**2)

def run_tprop_aunc(batch_size=100,
                    t_steps=200,
                    SGD=True,
                    layers=6,
                    h_dim=10,
                    nonlinearity='tanh',
                    alpha=0.003,
                    alpha_t=0.1,
                    alpha_inv=0.5,
                    beta_t=0.1,
                    beta_W=0.1,
                    beta_b=0.1,
                    pinv_rcond=1e-3,
                    nonlin_thresh=1e-2,
                    err_algs=[0,1,2],
                    training_algs=[0,1,2],
                    dataset='mnist',
                    preprocess=False):
  '''
    batch_size (int, > 0): the number of examples in each training batch
    t_steps (int, > 0): the number of training steps
    SGD (bool): stochastic gradient descent. Should be True. False can be useful for debugging and seeing if algorithms converge.

    layers (int, > 0): the number of layers in the deep network. the input 'layer' is not counted as a layer.
    nonlinearity ('tanh', 'sigmoid', 'relu'):

    alpha (float, > 0): the learning rate in gradient descent.
    alpha_t (float, (0, 1)): the 'learning rate' in target propagation, i.e. the target is set to alpha_t times the distance between the estimated output and the actual output.
    alpha_inv (float, (0, 1)): the 'learning rate' in the pseudoinverse update methods, i.e. instead of setting W_i to the new psuedoinverse solution, we move a per cent distance toward the new solution.

    beta_t (float, > 0): regularization for targets
    beta_W (float, > 0): regularization for W updates
    beta_b (float, > 0): regularization for b updates

    pinv_rcond (float, > 0): rcond value for np.linalg.pinv(), used in a couple places in the code. 
    nonlin_thresh (float, >0): threshold cutoffs for defining the generalized inverses for sigmoid and tanh. 

    dataset ('mnist', 'cifar'): which dataset to use. 
    preprocess (bool): preprocess the data with PCA + whitening. 
  '''

  if dataset == 'cifar':
    data = cifar10_data()
    data_test = cifar10_data_test()
  elif dataset == 'mnist':
    data = mnist_data()
    data_test = mnist_data_test()
  
  if preprocess:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1000, whiten=True)
    data.inputs = pca.fit_transform(data.inputs)
    data_test.inputs = pca.transform(data_test.inputs)

  # autoencoder
  data.outputs = data.inputs
  data_test.outputs = data_test.inputs

  # Model parameters
  m_dim = data.inputs.shape[1]
  p_dim = data.outputs.shape[1]
  
  #err_algs = [0, 1, 2] # Which set of error prop algs to use
  #training_algs = [0, 1, 2] # Which set of weight update algs to use

  # TODO: should ben len(err_algs), but this messes up the ipynb workflow...
  n_err_algs = 3 # there are three error prop algs: backprop, tprop, regularized tprop
  n_training_algs = 3 # there are three weight update algs: gradient descent, p_inv, regularized inv

  # Old option
  # if layers == 4:
  #   l_dim = [m_dim] + [128, 32, 128] + [p_dim]
  # elif layers == 6:
  #   l_dim = [m_dim] + [128, 64, 32, 64, 128] + [p_dim]
  # elif layers == 8:
  #   l_dim = [m_dim] + [256, 128, 64, 32, 64, 128, 256] + [p_dim]

  # New option
  if layers == 4:
    l_dim = [m_dim] + [50] + [h_dim] + [50] + [p_dim]
  elif layers == 6:
    l_dim = [m_dim] + [50, 50] + [h_dim] + [50, 50] + [p_dim]
  elif layers == 8:
    l_dim = [m_dim] + [50, 50, 50] + [h_dim] + [50, 50, 50] + [p_dim]

  # Set up np.zero arrays for all variables.
  # Forward weights
  W  = np.zeros((n_err_algs, n_training_algs, layers+1), dtype=object) # weights
  dW = np.zeros((n_err_algs, n_training_algs, layers+1), dtype=object) # dl/dW
  # Biases
  b  = np.zeros((n_err_algs, n_training_algs, layers+1), dtype=object) # biases
  db = np.zeros((n_err_algs, n_training_algs, layers+1), dtype=object) # dl/db
  # Loss
  L = np.zeros((n_err_algs, n_training_algs, t_steps+1)) # loss
  # Loss for test data
  L_test = np.zeros((n_err_algs, n_training_algs))

  # Initialize
  for k in err_algs:
    for j in training_algs:
      for l in range(1, layers+1):
        # TODO: make orth() robust to l_dim[2] > l_dim[1]. Actually -- Orth should be on transpose of matrix. 
        np.random.seed(l) # different random seed for each layer. Otherwise, same. For replicability.
        W[k, j, l] = np.random.randn(l_dim[l-1], l_dim[l])/np.sqrt(l_dim[l-1] + l_dim[l])
        #W[k, j, l] = 0.9*sp.linalg.orth(np.random.randn(l_dim[l-1], l_dim[l]))
        b[k, j, l] = 0*np.ones((1, l_dim[l]))

  # Activations
  x_1 = np.zeros((n_err_algs, n_training_algs, layers+1), dtype=object) # x_1 = W*x_3
  x_2 = np.zeros((n_err_algs, n_training_algs, layers+1), dtype=object) # x_2 = x_1 + b
  x_3 = np.zeros((n_err_algs, n_training_algs, layers+1), dtype=object) # x_3 = f(x_2)

  # Test set activations
  x_1_test = np.zeros((n_err_algs, n_training_algs, layers+1), dtype=object)
  x_2_test = np.zeros((n_err_algs, n_training_algs, layers+1), dtype=object)
  x_3_test = np.zeros((n_err_algs, n_training_algs, layers+1), dtype=object)

  # Errors / gradients
  dx_3 = np.zeros((n_err_algs, n_training_algs, layers+1), dtype=object)
  dx_2 = np.zeros((n_err_algs, n_training_algs, layers+1), dtype=object)
  dx_1 = np.zeros((n_err_algs, n_training_algs, layers+1), dtype=object)

  # Targets
  tx_3 = np.zeros((n_err_algs, n_training_algs, layers+1), dtype=object)
  tx_2 = np.zeros((n_err_algs, n_training_algs, layers+1), dtype=object)
  tx_1 = np.zeros((n_err_algs, n_training_algs, layers+1), dtype=object)

  # Training
  for t in range(t_steps+1):

    # print progress
    if t % 10 == 0:
      print t

    # Get data
    if SGD:
      x0, y = data.next_batch(batch_size)
    else:
      x0 = data.inputs[:batch_size]
      y = data.outputs[:batch_size]

    # Iterate through err_algs and training_algs
    for k in err_algs:
      for j in training_algs:
        
        # STEP 1: Forward pass
        x_3[k, j, 0] = x0
        for l in range(1, layers+1):
          x_1[k, j, l] = matmul(x_3[k, j, l-1], W[k, j, l])
          x_2[k, j, l] = add(x_1[k, j, l], b[k, j, l])
          x_3[k, j, l] = nl(x_2[k, j, l], nonlinearity)
        L[k, j, t] = mse(y, x_3[k, j, -1])

        # STEP 2: Backward pass
        # Top layer errors and targets
        dx_3[k, j, -1] = x_3[k, j, -1] - y # applies to both MSE and cross-entropy softmax
        tx_3[k, j, -1] = x_3[k, j, -1] - alpha_t*(x_3[k, j, -1] - y) # top layer target
        for l in range(layers, 0, -1):
          
          if k == 0:
            # Error propagation algs
            # Backprop (errors / derivatives)
            dx_2[k, j, l]   = dnl(x_2[k, j, l], nonlinearity) * dx_3[k, j, l]
            dx_1[k, j, l]   = dx_2[k, j, l]
            dx_3[k, j, l-1] = matmul(dx_1[k, j, l], W[k, j, l].T)

            # Backprop 'targets' (used for pinv weight update method)
            tx_2[k, j, l] = x_2[k, j, l] - alpha*dx_2[k, j, l]
            tx_1[k, j, l] = x_1[k, j, l] - alpha*dx_1[k, j, l]
            tx_3[k, j, l] = x_3[k, j, l] - alpha*dx_3[k, j, l]

          elif k == 1:
            # Difference target prop
            # Target prop targets
            # TODO, check tx_1
            tx_2[k, j, l]   = x_2[k, j, l] + nl_inv(tx_3[k, j, l], nonlinearity, th=nonlin_thresh) - nl_inv(x_3[k, j, l], nonlinearity, th=nonlin_thresh)
            tx_1[k, j, l]   = tx_2[k, j, l] - b[k, j, l]
            tx_3[k, j, l-1] = x_3[k, j, l-1] + matmul_pinv(tx_1[k, j, l], W[k, j, l], rcond=pinv_rcond) - matmul_pinv(x_1[k, j, l], W[k, j, l], rcond=pinv_rcond)

            # Target prop 'errors / derivatives'
            dx_2[k, j, l]   = x_2[k, j, l] - tx_2[k, j, l]
            dx_1[k, j, l]   = x_1[k, j, l] - tx_1[k, j, l]
            dx_3[k, j, l-1] = x_3[k, j, l-1] - tx_3[k, j, l-1] # TODO: check, set to l-1 or l?

          elif k == 2:
            # Regularized target prop
            # tx_2 and tx_1 same as above.
            tx_2[k, j, l]   = nl_rinv(tx_3[k, j, l], x_2[k, j, l], nonlinearity, beta_t)
            tx_1[k, j, l]   = 1./(beta_t + 1.)*(tx_2[k, j, l] - b[k, j, l] + beta_t*x_1[k, j, l])
            tx_3[k, j, l-1] = np.dot(np.dot(tx_1[k, j, l], W[k, j, l].T) + beta_t*x_3[k, j, l-1], np.linalg.inv(np.dot(W[k, j, l], W[k, j, l].T) + beta_t*np.eye(l_dim[l-1])))
           
            # Target prop 'errors / derivatives' (used for gradient descent weight update method)
            dx_2[k, j, l]   = x_2[k, j, l] - tx_2[k, j, l]
            dx_1[k, j, l]   = x_1[k, j, l] - tx_1[k, j, l]
            dx_3[k, j, l-1] = x_3[k, j, l-1] - tx_3[k, j, l-1] # TODO: check, set to l-1 or l?

        # STEP 3: Update variables
        for l in range(1, layers+1):

          if j == 0:
            # TODO: check convex combinations...
            # Gradient descent
            dW[k, j, l] = -alpha*np.dot(x_3[k, j, l-1].T, dx_1[k, j, l])/batch_size
            db[k, j, l] = -alpha*np.mean(dx_2[k, j, l], axis=0)
          elif j == 1:
            # Psuedoinverse solution / convex combination
            dW[k, j, l] = -alpha_inv*np.dot(np.linalg.pinv(x_3[k, j, l-1], rcond=pinv_rcond), (x_1[k, j, l] - tx_1[k, j, l])) # TODO: check 1/batch_size term.
            db[k, j, l] = -alpha_inv*(np.mean(dx_2[k, j, l], axis=0))
          elif j == 2:
            # regularized pseudoinverse solution / convex combination
            dW[k, j, l] = -alpha_inv*(W[k, j, l] - matmul_rinv(x_3[k, j, l-1], tx_1[k, j, l], W[k, j, l], beta_W)) # subtract W[k, j, l] because of the weight update step in the next line. 
            db[k, j, l] = -alpha_inv*(b[k, j, l] - np.mean(add_rinv(x_1[k, j, l], tx_2[k, j, l], b[k, j, l], beta_b), axis=0))
          
          # Update variables
          W[k, j, l] = W[k, j, l] + dW[k, j, l]
          b[k, j, l] = b[k, j, l] + db[k, j, l]

          # If parameters are exploding, just set to 0 so it doesn't effect performance of other methods.
          if np.abs(W[k, j, l]).max() > 1e4:
            W[k, j, l] = np.zeros((l_dim[l-1], l_dim[l]))
          if np.abs(b[k, j, l]).max() > 1e4:
            b[k, j, l] = np.zeros((1, l_dim[l]))

  # After training, get test error
  for k in err_algs:
    for j in training_algs:

      x_3_test[k, j, 0] = data_test.inputs
      for l in range(1, layers+1):
        x_1_test[k, j, l] = matmul(x_3_test[k, j, l-1], W[k, j, l])
        x_2_test[k, j, l] = add(x_1_test[k, j, l], b[k, j, l])
        x_3_test[k, j, l] = nl(x_2_test[k, j, l], nonlinearity)
      L_test[k, j] = mse(data_test.outputs, x_3_test[k, j, -1])

  # Activations to save
  actvs = np.zeros((n_err_algs, n_training_algs), dtype=object)
  for k in err_algs:
    for j in training_algs:
      actvs[k, j] = x_3_test[k, j, -1][:20]

  return L, L_test, actvs


# Plotting functions. For ipynb.
def square_axes(fig, i=0):
  ''' make axis look nice. useful for notebook visualizations '''
  fig.axes[i].axhline(0, color='w', linewidth=3.5, alpha=0.25)
  fig.axes[i].axvline(0, color='w', linewidth=3.5, alpha=0.25)
  fig.axes[i].set_xlim(-1.5,1.5)
  fig.axes[i].set_ylim(-1.5,1.5)
  fig.axes[i].set_aspect(1)
  fig.axes[i].get_xaxis().set_ticklabels([])
  fig.axes[i].get_yaxis().set_ticklabels([])

def get_preimg(img, func, eps=0.1):
  """ computes the preimage a of set img through func, i.e.
    img=func(a)"""
  c1 = np.linspace(-2, 2, 201)
  c2 = np.linspace(-2, 2, 201)
  x1, y1 = np.meshgrid(c1, c2)
  a = np.stack((x1.flatten(), y1.flatten()))
  d = np.zeros(a.shape[1])
  b = func(a)
  for i in range(a.shape[1]):
    d[i] = np.min(np.linalg.norm(b[:,i] - img))
  inds = d < eps
  return a[:,inds], b[:,inds]

def get_circle(center, radius, points=100):
  x = radius*np.cos(np.linspace(0, 2*np.pi, num=points)) + center[0]
  y = radius*np.sin(np.linspace(0, 2*np.pi, num=points)) + center[1]
  return np.stack((x,y))

def get_ce_grid(y):
  c = np.linspace(-1, 2, 401)
  x1, x2 = np.meshgrid(c, c)
  x_ = np.stack((x1.flatten(), x2.flatten()))
  ce = cross_entropy(y, x_).reshape([401, 401])
  return x1, x2, ce
