"""
  A simple Operation class. 
  For a function y=f(x), f: Rn -> Rm we associate three separate ways to define an 
  'inverse' of f, such that x_new = f_inv(y)
"""

import numpy as np
from scipy.optimize import fmin
import tensorflow as tf

def tf_rinv(y, x_0, func, func_inv, gamma=1e-2, lr=0.1, num_steps=2):
  """ slow implementation, because it has to create a new graph each time it is called. """
  y = y.astype('float32')
  x_0 = x_0.astype('float32')
  g = tf.Graph()
  with g.as_default():
    x_val = func_inv(y, x_0, th=1e-2)+ 0.0*np.random.randn(*x_0.shape)
    x = tf.Variable(x_val)
    L = tf.reduce_mean((func(x) - y)**2. + gamma*(x - x_0)**2.)
    opt = tf.train.GradientDescentOptimizer(lr).minimize(L)
    fdiff = np.inf
    xdiff = np.inf
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      f_val = sess.run(L) 
      counter = 0
      while counter < num_steps:
        counter += 1
        sess.run(opt)
        f_val_, x_val_ = sess.run([L, x])
        fdiff = np.abs(f_val-f_val_)
        xdiff = np.max(np.linalg.norm(x_val - x_val_, axis=1, keepdims=True))
        x_val = x_val_
        f_val = f_val_
      return x_val

def fmin_rinv(y, x_0, func, func_inv, gamma=1e-2):
  def cost(x, y, x_0):
    return np.sum((func(x) - y)**2 + gamma*(x - x_0)**2, axis=1)
  x = func_inv(y, x_0, th=1e-2)
  for i in range(y.shape[0]):
    x[i] = fmin(cost, x[i], args=(y[i, np.newaxis], x_0[i, np.newaxis]), xtol=1e-4, ftol=1e-4, maxiter=50, disp=0)

# this implementation didn't work really...
def tf_rinv_NOPE(y, x_0, func, func_inv, gamma=1e-2):
  y = y.astype('float32')
  x_0 = x_0.astype('float32')
  x_init = func_inv(y, x_0, th=1e-4) 
  x_out = np.zeros_like(x_0)
  g = tf.Graph()
  with g.as_default():
    tf.reset_default_graph()
    x   = (x_0.shape[0])*[None]
    L   = (x_0.shape[0])*[None]
    opt = (x_0.shape[0])*[None]
    for n in range(x_0.shape[0]):
      x[n] = tf.Variable(x_init[n,:,None])
      L[n] = tf.reduce_sum((func(x[n]) - y[n,:,None])**2 + gamma*(x[n] - x_0[n,:,None])**2)
      opt[n] = tf.train.RMSPropOptimizer(0.01).minimize(L[n], var_list=[x[n]])
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for n in range(x_0.shape[0]):
        fdiff = np.inf
        xdiff = np.inf
        x_val = sess.run(x[n])
        f_val = sess.run(L[n]) 
        counter = 0
        while fdiff > 1e-2 or xdiff > 1e-2:
          counter += 1
          sess.run(opt[n])
          f_val_, x_val_ = sess.run([L[n], x[n]])
          fdiff = np.abs(f_val-f_val_)
          xdiff = np.linalg.norm(x_val - x_val_, axis=1, keepdims=True)
          x_val = x_val_
          f_val = f_val_
        x_out[n] = x_val
  return x_out

class Op(object):
  def __init__(self, f, df, f_inv, f_rinv):
    """
      Constructs a Op object. Specify the function, its derivative, its least-squares inverse, and its regularized inverse.
      
      f: Rn -> Rm
      df: Rm x Rn -> Rn
      f_inv: Rm x Rn -> Rn
      f_rinv: Rm x Rn -> Rn

      df, f_inv, and f_rinv are three approaches to sending an output y through some kind of 
      'inverse' of f, to produce an input x, that is possibly close to some x_0.

      Args
        f: the function.
          y = f(x)
        df: derivative of f(x) wrt x and one step of the chain rule
          x = df(y, x, params)
          basically implements
          dl/dx = dl/dy*df(x)/dx where l is some global loss.
        f_inv: least-squares inverse of f
          x = f_inv(y, x_0, params)
          Solution to
          argim_x norm( x - x_0 ) s.t. x in {x | norm( f(x) - y ) is minimal }
        f_rinv: regularized least-squares inverse of f
          x = f_rinv(y, x_0, params)
          Solution to
          argmin_x norm( x - x_0 ) + gamma*norm( f(x) - y )
    """
    self._f = f
    self._df = df
    self._f_inv = f_inv
    self._f_rinv = f_rinv

  @property
  def f(self):
    return self._f

  @property
  def df(self):
    return self._df

  @property
  def f_inv(self):
    return self._f_inv

  @property
  def f_rinv(self):
    return self._f_rinv

  @f.setter
  def f(self, value):
    self._f = value

  @df.setter
  def df(self, value):
    self._df = value

  @f_inv.setter
  def f_inv(self, value):
    self._f_inv = value

  @f_rinv.setter
  def f_rinv(self, value):
    self._f_rinv = value


def sigmoid():
  def f(x):
    return 1./(1. + np.exp(-x))
  def df(y, x):
    return f(x)*(1. - f(x))*y
  def f_inv(y, x_0, th=1e-2):
    y = np.piecewise(y, [y <= th, y > th, y >= (1 - th)], [th, lambda y_: y_, 1 - th])
    return -np.log(1./y - 1.)
  def f_rinv(y, x_0, gamma=1e-2, lr=0.1, num_steps=2):
    return tf_rinv(y, x_0, tf.nn.sigmoid, f_inv, gamma=gamma)
  return Op(f, df, f_inv, f_rinv)

def tanh():
  def f(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
  def df(y, x):
    return y*(1. - f(x)**2)
  def f_inv(y, x_0, th=1e-2):
    y = np.piecewise(y, [y <= (-1+th), y > (-1+th), y >= (1-th)], [-1+th, lambda y_: y_, 1-th])
    return 0.5*np.log((1. + y)/(1. - y))
  def f_rinv(y, x_0, gamma=1e-2, lr=0.1, num_steps=2):
    return tf_rinv(y, x_0, tf.nn.tanh, f_inv, gamma=gamma)
  return Op(f, df, f_inv, f_rinv)

def relu():
  def f(x):
    return x*(x > 0)
  def df(y, x):
    return (x > 0).astype('float')*y
  def f_inv(y, x_0, th=None):
    case_1 = y > 0
    case_2 = (y <= 0)*(x_0 >= 0)
    case_3 = (y <= 0)*(x_0 < 0)
    x_1 = y
    x_2 = 0
    x_3 = x_0
    x_out = case_1*x_1 + case_2*x_2 + case_3*x_3
    return x_out
  def f_rinv(y, x0, gamma=1e-2, lr=None, num_steps=None):
    def cost(x):
      return (f(x) - y)**2 + gamma*(x - x0)**2
    x_out = np.zeros(y.shape)
    x_1 = 1./(1. + gamma)*(gamma*x0 + y) # possible solution 1
    x_2 = x0 # possible solution 2 -- only viable if gamma is not 0
    costs = np.array([cost(x_1), cost(x_2)])
    case_1 = (x_1 > 0)*np.logical_not(x_2 < 0)
    case_2 = (x_2 < 0)*np.logical_not(x_1 > 0)
    case_3 = (x_1 > 0)*(x_2 < 0)
    inds = np.argmin(costs, axis=0)
    x_3 = (inds == 0)*x_1 + (inds == 1)*x_2
    x_out = case_1*x_1 + case_2*x_2 + case_3*x_3
    return x_out
  return Op(f, df, f_inv, f_rinv)

def linear():
  def f(x, A):
    return np.dot(x, A)
  def df(y, x, A):
    return np.dot(y, A.T)
  def f_inv(y, x_0, A):
    Apinv = np.linalg.pinv(A)
    return x_0 - np.dot(np.dot(x_0, A), Apinv) + np.dot(y, Apinv)
  def f_rinv(y, x_0, A, gamma=1e-2, lr=None, num_steps=None):
    # assert gamma > 0
    return np.dot( np.dot(y, A.T) + gamma*x_0 , np.linalg.inv(np.dot(A, A.T) + gamma*np.eye(x_0.shape[1])) )
  return Op(f, df, f_inv, f_rinv)

def addition():
  def f(x, b):
    return x + b
  def df(y, x=None, b=None):
    return y
  def f_inv(y, x_0, b):
    return y - b
  def f_rinv(y, x_0, b, gamma=1e-2, lr=None, num_steps=None):
    return (y - b + gamma*x_0)/(1. + gamma)
  return Op(f, df, f_inv, f_rinv)

def identity():
  def f(x):
    return x
  def df(y, x=None):
    return y # to clarify, df=1, so return y*df by the chain rule. 
  def f_inv(y, x_0):
    return y
  def f_rinv(y, x_0, gamma=1e-2, lr=None, num_steps=None):
    # actual rinv not implemented...
    return y
  return Op(f, df, f_inv, f_rinv)

