import os
import sys
import pickle
import numpy as np

from sklearn.grid_search import ParameterGrid
from targprop_aunc import *

# Iterable parameters
param_grid = {}

param_grid['batch_size']    = np.random.choice([100], size=1)
param_grid['layers']        = np.random.choice([6], size=1)
param_grid['h_dim']         = np.random.choice([2, 3, 5, 10], size=1)

param_grid['nonlinearity']  = np.random.choice(['relu'], size=1)

param_grid['alpha']         = 10**np.random.uniform(-3, 0, size=1)
param_grid['alpha_inv']     = np.random.uniform(0, 1, size=1)

param_grid['beta_t']        = 10**np.random.uniform(-4, 3, size=1)
param_grid['beta_W']        = 10**np.random.uniform(-4, 3, size=1)
param_grid['beta_b']        = 10**np.random.uniform(-4, 3, size=1)

param_grid['err_algs']      = np.random.choice([0, 1, 2], size=1)
param_grid['training_algs'] = np.random.choice([0], size=1)

i = int(sys.argv[1])

# use i if doing a grid search. use j if doing a random search.
cur_params = ParameterGrid(param_grid)[0]

# Fixed parameters
#BATCH_SIZE = 100
T_STEPS = 5000

# Current run and paths
CUR_RUN = str(sys.argv[2])
DATASET = str(sys.argv[3])
if DATASET == 'cifar':
  preprocess = True
elif DATASET == 'mnist':
  preprocess = False

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

L, L_test, activations    = run_tprop_aunc(batch_size=cur_params['batch_size'],
                                           t_steps=T_STEPS,
                                           SGD=True,
                                           layers=cur_params['layers'],
                                           h_dim=cur_params['h_dim'],
                                           nonlinearity=cur_params['nonlinearity'],
                                           alpha=cur_params['alpha'],
                                           alpha_t=1,
                                           alpha_inv=cur_params['alpha_inv'],
                                           beta_t=cur_params['beta_t'],
                                           beta_W=cur_params['beta_W'],
                                           beta_b=cur_params['beta_b'],
                                           pinv_rcond=1e-3,
                                           nonlin_thresh=1e-2,
                                           err_algs=[cur_params['err_algs']],
                                           training_algs=[cur_params['training_algs']],
                                           dataset=DATASET,
                                           preprocess=preprocess)

pickle.dump([cur_params, L, L_test, activations], open(SAVE_PATH+str(i)+'.pickle', 'wb'))
