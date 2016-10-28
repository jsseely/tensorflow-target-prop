import os
import sys
import pdb
import numpy as np
from toy_data import *

def square_axes(fig, i=0):
  ''' make axis look nice. useful for notebook visualizations '''
  fig.axes[i].axhline(0, color='w', linewidth=3.5, alpha=0.25)
  fig.axes[i].axvline(0, color='w', linewidth=3.5, alpha=0.25)
  fig.axes[i].set_xlim(-1.5,1.5)
  fig.axes[i].set_ylim(-1.5,1.5)
  fig.axes[i].set_aspect(1)
  fig.axes[i].get_xaxis().set_ticklabels([])
  fig.axes[i].get_yaxis().set_ticklabels([])

# sigmoid ops
def sigmoid(x):
  return 1. / (1. + np.exp(-x))
def sigmoid_inv(x, th=1e-2):
  ''' inverse of sigmoid(). input to sigmoid_inv is clipped first
      to lie within the image of sigmoid() '''
  x = np.piecewise(x, [x <= th, x > th, x >= (1-th)], [th, lambda x_: x_, 1-th])
  return -np.log(1./x - 1.)
def dsigmoid(x):
  ''' derivative of sigmoid() '''
  return sigmoid(x)*(1.0 - sigmoid(x))

# tanh ops
def tanh(x):
  return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
def tanh_inv(x, th=1e-2):
  ''' inverse of tanh(). input to tanh_inv is clipped first
      to lie within the image of tanh() '''
  x = np.piecewise(x, [x <= (-1+th), x > (-1+th), x >= (1-th)], [-1+th, lambda x_: x_, 1-th])
  return 0.5*np.log((1. + x)/(1. - x))
def dtanh(x):
  ''' derivative of tanh() '''
  return 1. - tanh(x)**2

# nonlin op
def nl(x, nonlinearity='tanh'):
  if nonlinearity == 'tanh':
    return tanh(x)
  elif nonlinearity == 'sigmoid':
    return sigmoid(x)
def nl_inv(x, nonlinearity='tanh', th=1e-2):
  if nonlinearity == 'tanh':
    return tanh_inv(x, th=th)
  elif nonlinearity == 'sigmoid':
    return sigmoid_inv(x, th=th)
def dnl(x, nonlinearity='tanh'):
  if nonlinearity == 'tanh':
    return dtanh(x)
  elif nonlinearity == 'sigmoid':
    return dsigmoid(x)

# relu ops
def relu(x):
  return x*(x>0)
def drelu(x):
  return (x>0).astype('float')

# Matrix multiplication ops
def matmul(x, W):
  return np.dot(x, W)
def matmul_inv(x, W):
  return np.dot(x, np.linalg.pinv(W))
def matmul_pinv(x, W, rcond = 1e-2):
  return np.dot(x, np.linalg.pinv(W, rcond=rcond))

# Add/subtract ops (being overly explicit)
def add(x, b):
  return x + b
def add_inv(x, b):
  return x - b

# Loss-related ops
def softmax(x):
  return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)
def cross_entropy(y, x):
  return np.sum(-y*np.log(softmax(x)), axis=1, keepdims=True).mean()

def run_tprop(batch_size=100,
              t_steps=200,
              layers=3,
              alpha=0.003,
              alpha_t=0.1,
              SGD=True,
              pinv_rcond=1e-2,
              nonlin_thresh=1e-3,
              beta_1=0.1,
              beta_2=0.1,
              nonlinearity='tanh'):
  '''
    main function
  '''
  data = mnist_data()
  data_test = mnist_data_test()

  # Model parameters
  m_dim = data.inputs.shape[1]
  p_dim = data.outputs.shape[1]

  algs = 3 # there are three error prop algs: backprop, tprop, regularized tprop
  training_algs = 3 # there are three weight update algs: gradient descent, p_inv, regularized inv
  l_dim = [m_dim] + (layers-1)*[240] + [p_dim] # dimensions of each layer stored as a list

  # Forward weights
  W  = np.zeros((algs, training_algs, layers+1), dtype=object)
  dW = np.zeros((algs, training_algs, layers+1), dtype=object)
  # Biases
  b  = np.zeros((algs, training_algs, layers+1), dtype=object)
  db = np.zeros((algs, training_algs, layers+1), dtype=object)
  # Loss
  L = np.zeros((algs, training_algs, t_steps+1))
  accuracy = np.zeros((algs, training_algs, t_steps+1))
  # Loss for test data
  L_test = np.zeros((algs, training_algs))
  accuracy_test = np.zeros((algs, training_algs))

  # Initialize
  for k in range(algs):
    for j in range(training_algs):
      for l in range(1, layers+1):
        np.random.seed(l) # different random seed for each layer. Otherwise, same.
        W[k, j, l] = np.random.randn(l_dim[l-1], l_dim[l])/np.sqrt(l_dim[l-1])
        b[k, j, l] = 0*np.ones((1, l_dim[l]))

  # Activations
  x_1 = np.zeros((algs, training_algs, layers+1), dtype=object) # x_1 = W*x_3
  x_2 = np.zeros((algs, training_algs, layers+1), dtype=object) # x_2 = x_1 + b
  x_3 = np.zeros((algs, training_algs, layers+1), dtype=object) # x_3 = f(x_2)

  # Test Activations
  x_1_test = np.zeros((algs, training_algs, layers+1), dtype=object)
  x_2_test = np.zeros((algs, training_algs, layers+1), dtype=object)
  x_3_test = np.zeros((algs, training_algs, layers+1), dtype=object)

  # Errors
  dx_3 = np.zeros((algs, training_algs, layers+1), dtype=object)
  dx_2 = np.zeros((algs, training_algs, layers+1), dtype=object)
  dx_1 = np.zeros((algs, training_algs, layers+1), dtype=object)

  # Targets
  tx_3 = np.zeros((algs, training_algs, layers+1), dtype=object)
  tx_2 = np.zeros((algs, training_algs, layers+1), dtype=object)
  tx_1 = np.zeros((algs, training_algs, layers+1), dtype=object)

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

    # iterate through algs and training_algs
    for k in range(algs):
      for j in range(training_algs):

        # Forward pass
        x_3[k, j, 0] = x0
        for l in range(1, layers+1):
          x_1[k, j, l] = matmul(x_3[k, j, l-1], W[k, j, l])
          x_2[k, j, l] = add(x_1[k, j, l], b[k, j, l])
          x_3[k, j, l] = nl(x_2[k, j, l], nonlinearity)
        L[k, j, t] = cross_entropy(y, x_3[k, j, -1])
        correct_prediction = np.equal(np.argmax(softmax(x_3[k, j, -1]), axis=1), np.argmax(y, axis=1))
        accuracy[k, j, t] = np.mean(correct_prediction.astype('float'))

        # Backward pass
        # Top layer errors and targets
        dx_3[k, j, -1] = x_3[k, j, -1] - y # applies to both MSE and cross-entropy softmax
        tx_3[k, j, -1] = x_3[k, j, -1] - alpha_t*(x_3[k, j, -1] - y) # top layer target
        for l in range(layers, 0, -1):
          
          # Backprop
          if k == 0:
            # Backprop (errors / derivatives)
            dx_2[k, j, l]   = dnl(x_2[k, j, l], nonlinearity) * dx_3[k, j, l]
            dx_1[k, j, l]   = dx_2[k, j, l]
            dx_3[k, j, l-1] = matmul(dx_1[k, j, l], W[k, j, l].T)

            # Backprop 'targets' (used for pinv weight update method)
            tx_2[k, j, l] = x_2[k, j, l] - alpha_t*dx_2[k, j, l]
            tx_1[k, j, l] = x_1[k, j, l] - alpha_t*dx_1[k, j, l]
            tx_3[k, j, l] = x_3[k, j, l] - alpha_t*dx_3[k, j, l]

          # Target prop
          elif k == 1:
            # Target prop targets
            tx_2[k, j, l]   = x_2[k, j, l] + nl_inv(tx_3[k, j, l], nonlinearity, th=nonlin_thresh) - nl_inv(x_3[k, j, l], nonlinearity, th=nonlin_thresh)
            tx_1[k, j, l]   = x_1[k, j, l] + add_inv(tx_2[k, j, l], b[k, j, l]) - add_inv(x_2[k, j, l], b[k, j, l]) # overly explicit. just: tx_1 = tx_2 - b
            tx_3[k, j, l-1] = x_3[k, j, l-1] + matmul_pinv(tx_1[k, j, l], W[k, j, l], rcond=pinv_rcond) - matmul_pinv(x_1[k, j, l], W[k, j, l], rcond=pinv_rcond)

            # Target prop 'errors / derivatives' (used for gradient descent weight update method)
            dx_2[k, j, l]   = x_2[k, j, l] - tx_2[k, j, l]
            dx_1[k, j, l]   = x_1[k, j, l] - tx_1[k, j, l]
            dx_3[k, j, l-1] = x_3[k, j, l] - tx_3[k, j, l]

          # Regularized target prop
          elif k == 2:
            # tx_2 and tx_1 same as above.
            tx_2[k, j, l]   = x_2[k, j, l] + nl_inv(tx_3[k, j, l], nonlinearity, th=nonlin_thresh) - nl_inv(x_3[k, j, l], nonlinearity, th=nonlin_thresh)
            tx_1[k, j, l]   = x_1[k, j, l] + add_inv(tx_2[k, j, l], b[k, j, l]) - add_inv(x_2[k, j, l], b[k, j, l]) # overly explicit. just: tx_1 = tx_2 - b
            tx_3[k, j, l-1] = np.dot(np.dot(tx_1[k, j, l], W[k, j, l].T) + beta_1*x_3[k, j, l-1], np.linalg.pinv(np.dot(W[k, j, l], W[k, j, l].T) + beta_1*np.eye(l_dim[l-1]))) # use pinv to avoid very illconditioned matrices
           
            # Target prop 'errors / derivatives' (used for gradient descent weight update method)
            dx_2[k, j, l]   = x_2[k, j, l] - tx_2[k, j, l]
            dx_1[k, j, l]   = x_1[k, j, l] - tx_1[k, j, l]
            dx_3[k, j, l-1] = x_3[k, j, l] - tx_3[k, j, l]

        # Update variables
        for l in range(1, layers+1):
          if j == 0:
            # Gradient descent
            dW[k, j, l] = -alpha*np.dot(x_3[k, j, l-1].T, dx_1[k, j, l])/batch_size
            db[k, j, l] = -alpha*np.mean(dx_2[k, j, l], axis=0)
          elif j == 1:
            # Psuedoinverse solution
            db[k, j, l] = np.mean(tx_2[k, j, l] - x_2[k, j, l], axis=0)
            dW[k, j, l] = np.dot(np.linalg.pinv(x_3[k, j, l-1], rcond=pinv_rcond), (tx_1[k, j, l] - x_1[k, j, l]))
          elif j == 2:
            db[k, j, l] = np.mean(tx_2[k, j, l] - x_2[k, j, l], axis=0)
            dW[k, j, l] = np.dot(np.linalg.pinv(beta_2*np.eye(l_dim[l-1]) + np.dot(x_3[k, j, l-1].T, x_3[k, j, l-1])), np.dot(x_3[k, j, l-1].T, tx_1[k, j, l]) + beta_2*W[k, j, l]) - W[k, j, l] # subtract W[k, j, l] because of the weight update step in the next line. Use pinv to avoid very ill conditioned matrices
          # Update variables
          W[k, j, l] = W[k, j, l] + dW[k, j, l]
          b[k, j, l] = b[k, j, l] + db[k, j, l]

  # Now get test error
  for k in range(algs):
    for j in range(training_algs):
      # Forward pass test
      x_3_test[k, j, 0] = data_test.inputs
      for l in range(1, layers+1):
        x_1_test[k, j, l] = matmul(x_3_test[k, j, l-1], W[k, j, l])
        x_2_test[k, j, l] = add(x_1_test[k, j, l], b[k, j, l])
        x_3_test[k, j, l] = nl(x_2_test[k, j, l], nonlinearity)
      L_test[k, j] = cross_entropy(data_test.outputs, x_3_test[k, j, -1])
      correct_prediction = np.equal(np.argmax(softmax(x_3_test[k, j, -1]), axis=1), np.argmax(data_test.outputs, axis=1))
      accuracy_test[k, j] = np.mean(correct_prediction.astype('float'))

  return L, accuracy, L_test, accuracy_test



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
