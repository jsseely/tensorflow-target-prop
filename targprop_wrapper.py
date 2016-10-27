import os
import sys
import pickle
import numpy as np

from sklearn.grid_search import ParameterGrid
from targprop import *

# Iterable parameters
param_grid = {}

# # RNN and training parameters
# train every nth
# save random seed / np and in tf

param_grid['layers'] = np.random.randint(2, 8, size=1)
param_grid['alpha'] = 10**np.random.uniform(-4, -1, size=1).astype('float32')
param_grid['alpha_t'] = np.random.uniform(0, 1, size=1).astype('float32')
param_grid['pinv_rcond'] = 10**np.random.uniform(-3, 1, size=1).astype('float32')
param_grid['nonlin_thresh'] = 10**np.random.uniform(-3, 0, size=1).astype('float32')

# Use parameter_grid_split if you want to search only along coordinates.
# param_grid = parameter_grid_split(param_grid)

i = int(sys.argv[1])
#j = np.random.randint(0, len(list(ParameterGrid(param_grid)))-1)

# use i if doing a grid search. use j if doing a random search.
cur_params = ParameterGrid(param_grid)[0]

# Fixed parameters
BATCH_SIZE = 100
T_STEPS = 2000

# Current run and paths
CUR_RUN = str(sys.argv[2])

SAVE_PATH = os.getcwd()+'/saves/'+CUR_RUN+'/'

def make_dir(path):
  """
    like os.makedirs(path) but avoids race conditions
  """
  try:
    os.makedirs(path)
  except OSError:
    if not os.path.isdir(path):
      raise

make_dir(SAVE_PATH)

print('Current Run: '+CUR_RUN)

print(cur_params)

L, acc, L_test, acc_test = run_tprop(batch_size=BATCH_SIZE,
                                     t_steps=T_STEPS,
                                     layers=cur_params['layers'],
                                     alpha=cur_params['alpha'],
                                     alpha_t=cur_params['alpha_t'],
                                     SGD=True,
                                     pinv_rcond=cur_params['pinv_rcond'],
                                     nonlin_thresh=cur_params['nonlin_thresh'],
                                     nonlinearity='tanh')

pickle.dump([cur_params, L, acc, L_test, acc_test], open(SAVE_PATH+str(i)+'.pickle', 'wb'))
