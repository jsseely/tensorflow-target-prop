"""
  A wrapper function to specify hyperparameters and loop through train_net()

  usage:
  python wrapper.py 1 "20170101" "C"

  Args:
    i, int
    CUR_RUN, string

"""
import os
import sys
import pickle
import numpy as np

from targprop.tprop_train import train_net

# parameters
cur_params = {}
cur_params['batch_size']         = 100
cur_params['t_steps']            = 20000

cur_params['gamma']              = 10**np.random.uniform(-4, 1)
cur_params['alpha_t']            = np.random.uniform(0., 1.)
cur_params['noise_str']          = 10**np.random.uniform(-4, 2)

cur_params['learning_rate']      = 10**np.random.uniform(-5, -1)
cur_params['learning_rate_inv']  = 10**np.random.uniform(-5, -1)
cur_params['learning_rate_rinv'] = 10**np.random.uniform(-2, 1)
cur_params['num_steps_rinv']     = np.random.randint(1, 10)

cur_params['SGD']                = True

# input parameters
CUR_SIM = int(sys.argv[1])
CUR_RUN = str(sys.argv[2])

MODE = str(sys.argv[3])
ACT = str(sys.argv[4])

# input-dependent parameters
if MODE == 'autoencoder':
  cur_params['top_loss'] = 'sigmoid_ce'
  cur_params['l_dim'] = [200, 100, 3, 100, 200]
elif MODE == 'classification':
  cur_params['top_loss'] = 'softmax_ce'
  cur_params['l_dim'] = 5*[200]

DATASET = 'mnist'
if DATASET == 'cifar':
  PREPROCESS = True
elif DATASET == 'mnist':
  PREPROCESS = False

SAVE_PATH = './saves/'+CUR_RUN+'/'
TB_PATH = './saves/'+CUR_RUN+'/tb/'

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
make_dir(TB_PATH)

# save params
pickle.dump(cur_params, open(SAVE_PATH+str(CUR_SIM)+'params.pickle', 'wb'))

print 'Current Run: '+CUR_RUN
print cur_params

out_dict = []
for err_alg in [0, 2, 3]:
  out_dict_ = train_net(err_alg=err_alg, 
                        tb_path=TB_PATH+str(CUR_SIM)+'_'+str(err_alg),
                        **cur_params)
  out_dict.append(out_dict_)

pickle.dump(out_dict, open(SAVE_PATH+str(CUR_SIM)+'.pickle', 'wb'))
