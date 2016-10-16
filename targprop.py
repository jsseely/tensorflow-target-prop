import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

def square_axes(i=0):
  ''' make axis look nice '''
  fig.axes[i].axhline(0, color='w', linewidth=3.5, alpha=0.25)
  fig.axes[i].axvline(0, color='w', linewidth=3.5, alpha=0.25)
  fig.axes[i].set_xlim(-1.5,1.5)
  fig.axes[i].set_ylim(-1.5,1.5)
  fig.axes[i].set_aspect(1)
  fig.axes[i].get_xaxis().set_ticklabels([])
  fig.axes[i].get_yaxis().set_ticklabels([])

def get_matrix(layer=1):
  if layer == 1:
    A = np.array([[1.5, .5],[.5, 1.5]])
  else:
    A = np.array([[2, 2],[0, 2]])
  return A

def get_bias(layer=1):
  if layer == 1:
    b = np.array([[0],[0]])
  else:
    b = np.array([[0],[0]])
  return b

def f1(x,layer):
  return np.dot(get_matrix(layer), x) + get_bias(layer)
def f1_inv(x,layer):
  A = get_matrix(layer)
  return np.dot(np.linalg.inv(A), x - get_bias(layer))

def f2(x):
  return 2./(1 + np.exp(-x))-1
def f2_inv(x):
  return -np.log(2./(1+x) - 1)

def df2(x):
  return (1 - f2(x)**2)*x
def df1(x, layer):
  return np.dot(get_matrix(layer).T, x)

def get_preimg(img, func, eps=0.1):
  """ computes the preimage a of set img through func, i.e.
      img=func(a)"""
  c1 = np.linspace(-5, 5, 100)
  c2 = np.linspace(-5, 5, 100)
  x1, y1 = np.meshgrid(np.append(c1, 999), c2)
  x1[x1==999] = None
  y1[y1==999] = None
  #x2[x2==999] = None
  #y2[y2==999] = None
  a = np.stack((x1.flatten(), y1.flatten()))
  d = np.zeros(a.shape[1])
  b = func(a)
  for i in range(a.shape[1]):
      d[i] = np.min(np.linalg.norm(a[:,i] - img))
  inds = d < eps
  return a[:,inds], b[:,inds]

def get_circle(center, radius, points=100):
  x = radius*np.cos(np.linspace(0, 2*np.pi, num=points)) + center[0]
  y = radius*np.sin(np.linspace(0, 2*np.pi, num=points)) + center[1]
  return np.stack((x,y))

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

