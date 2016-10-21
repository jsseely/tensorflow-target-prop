import os
import sys
import numpy as np
sys.path.insert(0, '/Users/jeff/Documents/Python/_projects/tdadl/')
from toy_data import *

def square_axes(i=0):
  ''' make axis look nice '''
  fig.axes[i].axhline(0, color='w', linewidth=3.5, alpha=0.25)
  fig.axes[i].axvline(0, color='w', linewidth=3.5, alpha=0.25)
  fig.axes[i].set_xlim(-1.5,1.5)
  fig.axes[i].set_ylim(-1.5,1.5)
  fig.axes[i].set_aspect(1)
  fig.axes[i].get_xaxis().set_ticklabels([])
  fig.axes[i].get_yaxis().set_ticklabels([])

def sigmoid(x):
  return 1. / (1. + np.exp(-x))
def sigmoid_inv(x, th = 0.001):
  x = x * (x > th)
  x = x * (x < (1-th))
  x = x + th*(x <= th)
  x = x + (1-th)*(x >= (1-th))
  return -np.log(1./x - 1.)
def dsigmoid(x):
  return sigmoid(x)*(1.0 - sigmoid(x))

def tanh(x):
  return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
def tanh_inv(x, th = 0.001):
  x = x * (x > (-1+th))
  x = x * (x < (1-th))
  x = x + (-1+th)*(x <= (-1+th))
  x = x + (1-th)*(x >= (1-th))
  return -0.5*np.log((1. + x)/(1. - x))
def dtanh(x):
  return 1. - tanh(x)**2

def relu(x):
  return x*(x>0)
def drelu(x):
  return (x>0).astype('float')

# d_op for multiinputs...
def matmul(x, W):
  return np.dot(x, W)
def matmul_inv(x, W):
  return np.dot(x, np.linalg.inv(W))
def matmul_pinv(x, W, rcond = 1e-1):
  return np.dot(x, np.linalg.pinv(W, rcond=rcond))

def add(x, b):
  return x + b
def add_inv(x, b):
  return x - b

def softmax(x):
  return np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)
def cross_entropy(y, x):
  return np.sum(-y*np.log(softmax(x)), axis=1, keepdims=True).mean()

def run_tprop():
  data = mnist_data()

  batch_size = 1000

  ## Model parameters
  m_dim = data.inputs.shape[1]
  p_dim = data.outputs.shape[1]

  t_steps = 500
  algs = 2
  training_algs = 2
  layers = 3
  l_dim = [m_dim] + (layers-1)*[240] + [p_dim]
  alpha = 0.003
  alpha_t = 0.01

  # Forward weights
  W  = np.zeros((algs, layers+1), dtype=object)
  dW = np.zeros((algs, layers+1), dtype=object)
  # Biases
  b  = np.zeros((algs, layers+1), dtype=object)
  db = np.zeros((algs, layers+1), dtype=object)
  L = np.zeros((algs, t_steps+1))
  accuracy = np.zeros((algs, t_steps+1))

  # Initialize
  for k in range(algs):
    for l in range(1, layers+1):
      W[k, l] = np.random.randn(l_dim[l-1], l_dim[l])/np.sqrt(l_dim[l-1])
      b[k, l] = 0*np.ones((1, l_dim[l]))

  # Activations
  x_1 = np.zeros((algs, layers+1), dtype=object)
  x_2 = np.zeros((algs, layers+1), dtype=object)
  x_3 = np.zeros((algs, layers+1), dtype=object)

  # Errors
  dx_3 = np.zeros((algs, layers+1), dtype=object)
  dx_2 = np.zeros((algs, layers+1), dtype=object)
  dx_1 = np.zeros((algs, layers+1), dtype=object)

  # Targets
  tx_3 = np.zeros((algs, layers+1), dtype=object)
  tx_2 = np.zeros((algs, layers+1), dtype=object)
  tx_1 = np.zeros((algs, layers+1), dtype=object)

  # Training
  for t in range(t_steps+1):
    if t % 100 == 0:
      print t
    # Get data
    x0, y = data.rand_batch(batch_size)
    #x0 = data.inputs[:batch_size]
    #y = data.outputs[:batch_size]
    for k in range(algs):
      # Forward pass
      x_3[k, 0] = x0
      for l in range(1, layers+1):
        x_1[k, l] = matmul(x_3[k, l-1], W[k, l])
        x_2[k, l] = add(x_1[k, l], b[k, l])
        x_3[k, l] = tanh(x_2[k, l])

      # Backward pass
      dx_3[k, -1] = x_3[k, -1] - y # true for both MSE and cross-entropy softmax (?)
      tx_3[k, -1] = x_3[k, -1] - alpha_t*(x_3[k, -1] - y)
      for l in range(layers, 0, -1):
        # TODO: separate targ_algs and training_algs (training_algs / gd or pinv)
        if k == 0:
          # Backprop
          dx_2[k, l]   = dtanh(x_2[k, l]) * dx_3[k, l]
          dx_1[k, l]   = dx_2[k, l]
          dx_3[k, l-1] = matmul(dx_1[k, l], W[k, l].T)
        elif k == 1:
          # Target prop using inverses
          tx_2[k, l]   = x_2[k, l] + tanh_inv(tx_3[k, l]) - tanh_inv(x_3[k, l])
          tx_1[k, l]   = x_1[k, l] + add_inv(tx_2[k, l], b[k, l]) - add_inv(x_2[k, l], b[k, l])
          tx_3[k, l-1] = x_3[k, l-1] + matmul_pinv(tx_1[k, l], W[k, l]) - matmul_pinv(x_1[k, l], W[k, l])

      # Update parameters
      for l in range(1, layers+1):
        if k == 0:
          dW[k, l] = -alpha*np.dot(x_3[k, l-1].T, dx_1[k, l])/batch_size
          db[k, l] = -alpha*np.mean(dx_2[k, l], axis=0)
        elif k == 1:
          #dW[k, l] = -alpha*np.dot(x_3[k, l-1].T, x_1[k, l] - tx_1[k, l])/batch_size #why does not having a negative sign work???
          db[k, l] = -alpha*np.mean(x_2[k, l] - tx_2[k, l], axis=0)
          dW[k, l] = np.dot(np.linalg.pinv(x_3[k, l-1], rcond=0.01), (x_1[k, l] - tx_1[k, l]))
        W[k, l] = W[k, l] + dW[k, l]
        b[k, l] = b[k, l] + db[k, l]

      # Loss 
      L[k, t] = cross_entropy(y, x_3[k, -1])
      correct_prediction = np.equal(np.argmax(softmax(x_3[k, -1]), axis=1), np.argmax(y,axis=1))
      accuracy[k, t] = np.mean(correct_prediction.astype('float'))

  return L, accuracy

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

def prop(init, L=1):
  x = []
  x.append(init)
  for l in range(1, L+1):
    x.append(f1(x[-1], l))
    x.append(f2(x[-1]))
  return x

def back(init, L=1):
  y = []
  y.append(init)
  for l in range(L, 0, -1):
    y.append(f2_inv(y[-1]))
    y.append(f1_inv(y[-1], l))
  y.reverse()
  return y

def bprop(init, L=1):
  d = []
  d.append(init)
  for l in range(L, 0, -1):
    d.append(df2(d[-1]))
    d.append(df1(d[-1], l))
  d.reverse()
  return d

