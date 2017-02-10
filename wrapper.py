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

# Iterable parameters
cur_params = {}
cur_params['batch_size']        = 100
cur_params['gamma']             = 10**np.random.uniform(-6, -1)
cur_params['alpha_t']           = np.random.uniform(0, 1)
cur_params['noise_str']         = 10**np.random.uniform(-4, 1)
cur_params['learning_rate']     = 10**np.random.uniform(-6, -1)
cur_params['learning_rate_inv'] = 10**np.random.uniform(-6, -1)

T_STEPS = 20000
UPDATES = 'tf'
SGD = True

CUR_SIM = int(sys.argv[1])
CUR_RUN = str(sys.argv[2])

MODE = str(sys.argv[3])
ACT = str(sys.argv[4])

DATASET = 'mnist'

if MODE == 'classification':
  cur_params['l_dim'] = 5*[200]
elif MODE == 'autoencoder':
  cur_params['l_dim'] = [200, 100, 5, 100, 200]

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
  out_dict_ = train_net(batch_size=cur_params['batch_size'],
                         t_steps=T_STEPS,
                         l_dim=cur_params['l_dim'],
                         act=ACT,
                         gamma=cur_params['gamma'],
                         alpha_t=cur_params['alpha_t'],
                         noise_str=cur_params['noise_str'],
                         err_alg=err_alg,
                         learning_rate=cur_params['learning_rate'],
                         learning_rate_inv=cur_params['learning_rate_inv'],
                         mode=MODE,
                         dataset=DATASET,
                         update_implementation=UPDATES,
                         SGD=SGD,
                         preprocess=PREPROCESS,
                         tb_path=TB_PATH+str(CUR_SIM)+'_'+str(err_alg))
  out_dict.append(out_dict_)

pickle.dump(out_dict, open(SAVE_PATH+str(CUR_SIM)+'.pickle', 'wb'))
