#!/usr/bin/python
#2019.1.16
#vanilla supervised vae pretrained 100 epoch
## regressor pretrained on the decoder result of vanilla supervised_vae pretrained till converge(500 epoch)
## the regressor and main model updated continous 5 epochs for each epoch

import matplotlib
import matplotlib.pyplot as plt
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
import h5py
import pdb
import numpy as np
import argparse
import random
import sys, os
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.append('%s/./util/' % os.path.dirname(os.path.realpath('__file__')))
from cmd_args import cmd_args
import cfg_parser as parser

#parameters need to be specified
#cmd_args.regressor_saved_dir = '/work/molecule_generation/sdvae/mol_vae_clean/zinc_data_experiments/regressor/regressor_pretrained_pretrained_vae_output'
#cmd_args.vanilla_vae_save_dir = '/work/molecule_generation/sdvae/mol_vae_clean/ICML_2019/vanila_supervised_vae/zinc/vanilla_supervised_vae'
#cmd_args.training_data_dir = '/work/molecule_generation/sdvae/mol_vae_clean/zinc_data_experiments/data/data_278'
#cmd_args.info_folder = '/work/molecule_generation/sdvae/mol_vae_clean/ICML_2019/dropbox'



# constant params
cmd_args.save_dir = './model/vae_pretrained_100_epoch_reg_param_'+ str(cmd_args.RL_param)
if not os.path.exists(cmd_args.save_dir):
    os.makedirs(cmd_args.save_dir)
seed = 19260817
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


from model import Regressor
from train_util import Prepare_data
from model import MolVAE
from model import train_regressor, epoch_train
#define regressor
regressor = Regressor(cmd_args.max_decode_steps, cmd_args.latent_dim)
regressor.cuda()

#define vae model
ae = MolVAE()  
if cmd_args.mode == 'gpu':
    ae = ae.cuda()    
assert cmd_args.encoder_type == 'cnn'    


# load the data
getting_data = Prepare_data(cmd_args)
train_binary_x, train_masks, valid_binary_x, valid_masks, train_y, valid_y = getting_data.load_data()
print('num_train: %d\tnum_valid: %d' % (train_y.shape[0], valid_binary_x.shape[0]))    


# load the pretrained vae model
vae_pretrained_model = cmd_args.vanilla_vae_save_dir + '/epoch-40.model'
if vae_pretrained_model  is not None and vae_pretrained_model != '':
        if os.path.isfile(vae_pretrained_model):
            print('loading model from %s' % vae_pretrained_model)
            ae.load_state_dict(torch.load(vae_pretrained_model))


# load the pretrained regressor            
cmd_args.regressor_saved_model = cmd_args.regressor_saved_dir + '/epoch-best.model'
if cmd_args.regressor_saved_model is not None and cmd_args.regressor_saved_model != '':
        if os.path.isfile(cmd_args.regressor_saved_model):
            print('loading model from %s' % cmd_args.regressor_saved_model)
            regressor.load_state_dict(torch.load(cmd_args.regressor_saved_model))


#optimizer
optimizer_vae = optim.Adam(ae.parameters(), lr=cmd_args.learning_rate)
optimizer_regularizer = optim.Adam(ae.state_decoder.parameters(), lr = cmd_args.learning_rate)
lr_scheduler = ReduceLROnPlateau(optimizer_vae, 'min', factor=0.5, patience=5, verbose=True, min_lr=0.001)
optimizer_regressor = optim.Adam(regressor.parameters(), lr=cmd_args.learning_rate)


#train
sample_idxes = list(range(train_binary_x.shape[0]))
best_valid_loss = None
kl = []
prep = []
original_reward = []
permuted_reward = []

for epoch in range(cmd_args.num_epochs):
    random.shuffle(sample_idxes)
    
      
    if epoch > -1:
        ## update the reggressor:
        for epoch_2 in range(5):
            regressor, regressor_loss, reg_1= train_regressor(epoch_2, optimizer_regressor, ae, regressor,sample_idxes,  train_binary_x, train_y)
            print('>>>>average regressor \033[92mtraining\033[0m of epoch %d: loss %.5f, reg1 %.5f' % (epoch_2, regressor_loss, reg_1))  
     


        
    ## update the vae:
    for epoch_vae in range(5):
        ae, vae_loss = epoch_train('train',epoch_vae, ae, regressor, sample_idxes, train_binary_x, train_masks, train_y,cmd_args, optimizer_vae, optimizer_regularizer)    
        print('>>>>average \033[92mtraining\033[0m of epoch %d: vae_loss loss %.5f regularizer_loss %.5f original_reward %.5f permuted_reward %.5f   prep %.5f kl prep %.5f' % (epoch_vae, vae_loss[0], vae_loss[1], vae_loss[2], vae_loss[3], vae_loss[4], vae_loss[5]))   
        kl.append(vae_loss[5])
        prep.append(vae_loss[4])
        permuted_reward.append(vae_loss[3])
        original_reward.append(vae_loss[2])
        

    if epoch % 1 == 0:
        _, valid_loss = epoch_train('valid', epoch,  ae, regressor, list(range(valid_binary_x.shape[0])), valid_binary_x, valid_masks, valid_y,cmd_args)
        print('>>>>average \033[92mtraining\033[0m of epoch %d: vae_loss loss %.5f regularizer_loss %.5f original_reward %.5f permuted_reward %.5f   prep %.5f kl prep %.5f' % (epoch_vae, vae_loss[0], vae_loss[1], vae_loss[2], vae_loss[3], vae_loss[4], vae_loss[5]))   
        valid_loss = valid_loss[0]
        lr_scheduler.step(valid_loss)
        torch.save(ae.state_dict(), cmd_args.save_dir + '/epoch-%d.model' % epoch)
      
        if best_valid_loss is None or valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print('saving to best model since this is the best valid loss so far.----')
            torch.save(ae.state_dict(), cmd_args.save_dir + '/epoch-best.model')

            np.save(cmd_args.save_dir + '/kl.npy', kl) 
            np.save(cmd_args.save_dir + '/prep.npy', prep)
            np.save(cmd_args.save_dir +'/permuted_reward.npy', permuted_reward)
            np.save(cmd_args.save_dir + '/original_reward.npy', original_reward)
            
            
