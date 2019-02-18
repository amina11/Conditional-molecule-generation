# this file is used to generate one hot encoding of the smile and masking from QM9.csv file 
# it also do train test sepratrion and save the data in ./data/QM9_clean_smi_test.h5, QM9_clean_smi_train.h5 as well as their labels.

from __future__ import print_function
from past.builtins import range
import os
import sys
import numpy as np
import math
import random
from tqdm import tqdm
from joblib import Parallel, delayed
import h5py
from cmd_args import cmd_args

sys.path.append('%s/./util/' % os.path.dirname(os.path.realpath('__file__')))
import cfg_parser as parser
from mol_tree import AnnotatedTree2MolTree
from mol_util import DECISION_DIM
from attribute_tree_decoder import create_tree_decoder
from batch_make_att_masks import batch_make_att_masks
from tree_walker import OnehotBuilder 

cmd_args.data = "./data/qm9.csv"
cmd_args.data_save_dir = "./data/data_" + str(cmd_args.max_decode_steps)
cmd_args.smiles_file = cmd_args.data_save_dir + '/QM9_clean_smi.smi'


def process_chunk(smiles_list):
    grammar = parser.Grammar(cmd_args.grammar_file)

    cfg_tree_list = []
    for smiles in smiles_list:
        ts = parser.parse(smiles, grammar)
        assert isinstance(ts, list) and len(ts) == 1

        n = AnnotatedTree2MolTree(ts[0])
        cfg_tree_list.append(n)

    walker = OnehotBuilder()
    tree_decoder = create_tree_decoder()
    onehot, masks = batch_make_att_masks(cfg_tree_list, tree_decoder, walker, dtype=np.byte)

    return (onehot, masks)

def run_job(L):
    chunk_size = 5000
    
    
    list_binary = Parallel(n_jobs=cmd_args.data_gen_threads, verbose=50)(
        delayed(process_chunk)(L[start: start + chunk_size])
        for start in range(0, len(L), chunk_size)
    )

    
    #process_chunk(L[start: start + chunk_size] for start in range(0, len(L), chunk_size))
   
    
    all_onehot = np.zeros((len(L), cmd_args.max_decode_steps, DECISION_DIM), dtype=np.byte)
    all_masks = np.zeros((len(L), cmd_args.max_decode_steps, DECISION_DIM), dtype=np.byte)

    for start, b_pair in zip( range(0, len(L), chunk_size), list_binary ):
        all_onehot[start: start + chunk_size, :, :] = b_pair[0]
        all_masks[start: start + chunk_size, :, :] = b_pair[1]

    f_smiles = '.'.join(cmd_args.smiles_file.split('/')[-1].split('.')[0:-1])
    out_file = '%s/%s-%d.h5' % (cmd_args.data_save_dir, f_smiles, cmd_args.skip_deter)
    h5f = h5py.File(out_file, 'w')
    h5f.create_dataset('x', data=all_onehot)
    h5f.create_dataset('masks', data=all_masks)
    h5f.close()

    
## load the QM9 dataset
import pandas as pd 
data = pd.read_csv(cmd_args.data, header=None)
data = np.array(data)
QM9_clean_smi = data[:,1]
QM9_clean_label= data[:,2:]
## normalize the sa_score and logp
sascore_normalizer = (np.absolute(QM9_clean_label[:,1])).max()
logp_normalizer = (np.absolute(QM9_clean_label[:,2])).max()


#save the smile files seperately to make run_job consistent
if not os.path.exists(cmd_args.data_save_dir):
    os.makedirs(cmd_args.data_save_dir)
    
    
with open(cmd_args.smiles_file,'w') as f:
    for row in QM9_clean_smi:
        f.write(row)
        f.write('\n')
f.close()    

# run to binarize
run_job(QM9_clean_smi)

## shuffle and train test seperation
import random
h5f = h5py.File(cmd_args.data_save_dir  + '/QM9_clean_smi-0.h5', 'r')
binary = h5f['x'][:]
masks = h5f['masks'][:]

#shuffle
num = np.arange(binary.shape[0])
random.shuffle(num)
binary_shuffle = binary[num,:,:]
masks_shuffle = masks[num,:,:]
y_shuffle = QM9_clean_label[num,:]
smile_shuffle = QM9_clean_smi[num]

#seperate train_test
#n_test = np.int(binary.shape[0]* 0.2)
n_test = 10000
train_x = binary_shuffle[n_test:, :,:]
train_x_mask = masks_shuffle[n_test:, :,:]
train_y = y_shuffle[n_test:, :]
train_smiles = smile_shuffle[n_test:]

test_x = binary_shuffle[:n_test, :,:]
test_x_mask = masks_shuffle[:n_test, :,:]
test_y = y_shuffle[:n_test,:]
test_smiles = smile_shuffle[:n_test]

#normalize
train_normalized_y = np.copy(train_y)
train_normalized_y[:, 1]= train_y[:,1] / sascore_normalizer
train_normalized_y[:,2] = train_y[:,2] / logp_normalizer

test_normalized_y = np.copy(test_y)
test_normalized_y[:, 1]= test_y[:,1] / sascore_normalizer
test_normalized_y[:,2] = test_y[:,2] / logp_normalizer


#train
train = h5py.File(cmd_args.data_save_dir  + '/QM9_clean_smi_train.h5', 'w')
train.create_dataset('x_train', data=train_x)
train.create_dataset('masks_train', data=train_x_mask)
train.close()
np.save(cmd_args.data_save_dir  + '/QM9_clean_smi_train_y.npy', train_y)
np.save(cmd_args.data_save_dir  + '/QM9_clean_smi_train_smile.npy', train_smiles)
np.save(cmd_args.data_save_dir  + '/QM9_normalized_train_y.npy', train_normalized_y)

#test
test = h5py.File(cmd_args.data_save_dir + 'QM9_clean_smi_test.h5', 'w')
test.create_dataset('x_test', data=test_x)
test.create_dataset('masks_test', data=test_x_mask)
test.close()

np.save(cmd_args.data_save_dir  + '/QM9_clean_smi_test_y.npy', test_y)
np.save(cmd_args.data_save_dir + '/QM9_clean_smi_test_smile.npy', test_smiles)
np.save(cmd_args.data_save_dir  + '/QM9_normalized_test_y.npy', test_normalized_y)

np.save(cmd_args.data_save_dir + '/QM9_sascore_normalizer.npy', sascore_normalizer)
np.save(cmd_args.data_save_dir + '/QM9_logp_normalizer.npy', logp_normalizer)