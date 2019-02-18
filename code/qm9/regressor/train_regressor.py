# the regressor will be trained on previously trained and stroed supervised vae model decoder output

# 2019.1.16
# zinc data set
# regressor pretrained on the pretrained_vanilla_vae output and then fixed
# encoder take only x as input

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


#prameters to be specified 
#cmd_args.regressor_saved_dir = './regressor/regressor_pretrained_pretrained_vae_output'
#cmd_args.vanilla_vae_save_dir = './../ICML_2019/vanila_supervised_vae/zinc/vanilla_supervised_vae'
#cmd_args.training_data_dir = '/work/molecule_generation/sdvae/mol_vae_clean/zinc_data_experiments/data/data_278'
sys.path.append('%s/../' % os.path.dirname(os.path.realpath('__file__')))
from cmd_args import cmd_args
sys.path.append('%s/../util/' % os.path.dirname(os.path.realpath('__file__')))
import cfg_parser as parser
cmd_args.output_dim = 1
cmd_args.batch_size = 300
cmd_args.num_epochs= 300
cmd_args.output_dim = 1
cmd_args.kl_coeff = 1
seed = 19260817


sys.path.append('%s/../util/' % os.path.dirname(os.path.realpath('__file__')))
## define the model
from mol_util import rule_ranges, terminal_idxes, DECISION_DIM
from train_util import PerpCalculator
from train_util import raw_logit_to_smile_labels, run_job


class MolVAE(nn.Module):
    def __init__(self):
        super(MolVAE, self).__init__()
        self.latent_dim = cmd_args.latent_dim
        self.encoder = CNNEncoder(max_len=cmd_args.max_decode_steps, latent_dim=cmd_args.latent_dim)
        self.state_decoder = StateDecoder(max_len=cmd_args.max_decode_steps, latent_dim=cmd_args.latent_dim)
        self.perp_calc = PerpCalculator()

    def reparameterize(self, mu, logvar):
        if self.training:
            eps = mu.data.new(mu.size()).normal_(0, cmd_args.eps_std)            
            if cmd_args.mode == 'gpu':
                eps = eps.cuda()
            eps = Variable(eps)
            
            return mu + eps * torch.exp(logvar * 0.5)            
        else:
            return mu

    def forward(self, x_inputs, y_inputs, true_binary, rule_masks, t_y):        
        z_mean, z_log_var = self.encoder(x_inputs)
        z = self.reparameterize(z_mean, z_log_var)
        raw_logits = self.state_decoder(z, y_inputs)   
        perplexity = self.perp_calc(true_binary, rule_masks, raw_logits)
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var), -1)
        
        return perplexity, cmd_args.kl_coeff * torch.mean(kl_loss)
    
from pytorch_initializer import weights_init
#q(z|x)
class CNNEncoder(nn.Module):
    def __init__(self, max_len, latent_dim):
        super(CNNEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.max_len = max_len

        self.conv1 = nn.Conv1d(DECISION_DIM, 9, 9)
        self.conv2 = nn.Conv1d(9, 9, 9)
        self.conv3 = nn.Conv1d(9, 10, 11)

        self.last_conv_size = max_len - 9 + 1 - 9 + 1 - 11 + 1
        self.w1 = nn.Linear(self.last_conv_size * 10, 435)
        self.mean_w = nn.Linear(435, latent_dim)
        self.log_var_w = nn.Linear(435, latent_dim)
        weights_init(self)

    def forward(self, x_cpu):
        if cmd_args.mode == 'cpu':
            batch_input = Variable(torch.from_numpy(x_cpu))
        else:
            batch_input = Variable(torch.from_numpy(x_cpu).cuda())

        h1 = self.conv1(batch_input)
        h1 = F.relu(h1)        
        h2 = self.conv2(h1)
        h2 = F.relu(h2)
        h3 = self.conv3(h2)
        h3 = F.relu(h3)

        # h3 = torch.transpose(h3, 1, 2).contiguous()
        flatten = h3.view(x_cpu.shape[0], -1)
        h = self.w1(flatten)
        h = F.relu(h)

        z_mean = self.mean_w(h)
        z_log_var = self.log_var_w(h)
        
        return (z_mean, z_log_var)    
    
    
#decoder    
class StateDecoder(nn.Module):
    def __init__(self, max_len, latent_dim):
        super(StateDecoder, self).__init__()
        self.latent_dim = latent_dim + cmd_args.output_dim
        self.max_len = max_len

        self.z_to_latent = nn.Linear(self.latent_dim, self.latent_dim)
        if cmd_args.rnn_type == 'gru':
            self.gru = nn.GRU(self.latent_dim, 501, cmd_args.output_dim)
        elif cmd_args.rnn_type == 'sru':
            self.gru = SRU(self.latent_dim, 501, cmd_args.output_dim)
        else:
            raise NotImplementedError

        self.decoded_logits = nn.Linear(501, DECISION_DIM)
        weights_init(self)

    def forward(self, z, y):
       
        if cmd_args.mode == 'cpu':
            y = torch.tensor(y).float()
        else:
            y = torch.tensor(y).cuda().float()

        assert len(z.size()) == 2 # assert the input is a matrix
        h = self.z_to_latent(torch.cat((z, y.view(y.shape[0],1)), 1))
        #h = self.z_to_latent(torch.cat((z, torch.tensor(y).cuda().float()), 1))
        h = F.relu(h)
        rep_h = h.expand(self.max_len, z.size()[0], z.size()[1] + cmd_args.output_dim) # repeat along time steps
        out, _ = self.gru(rep_h) # run multi-layer gru
        logits = self.decoded_logits(out)
        return logits   

#1. Define regressor
from mol_util import DECISION_DIM
class Regressor(nn.Module):
    def __init__(self, max_len, latent_dim):
        super(Regressor, self).__init__()
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.conv1 = nn.Conv1d(DECISION_DIM, 9, 9)
        self.conv2 = nn.Conv1d(9, 9, 9)
        self.conv3 = nn.Conv1d(9, 10, 11)
        self.last_conv_size = max_len - 9 + 1 - 9 + 1 - 11 + 1
        self.w1 = nn.Linear(self.last_conv_size * 10 , 435 )
        self.mean_w = nn.Linear(435, latent_dim)
        self.label_linear_1 = nn.Linear(latent_dim, cmd_args.output_dim)
        weights_init(self)

    def forward(self, x_cpu):
        if cmd_args.mode == 'cpu':
            batch_input = Variable(x_cpu)
   
        else:
            batch_input = Variable(x_cpu.cuda())
            
        h1 = self.conv1(batch_input)
        h1 = F.relu(h1)        
        h2 = self.conv2(h1)
        h2 = F.relu(h2)
        h3 = self.conv3(h2)
        h3 = F.relu(h3)
        flatten = h3.view(x_cpu.shape[0], -1)
        h = self.w1(flatten)
        h = F.relu(h)
        ## apply a linear transformation followed by sigmoid on latent z
        z_mean = self.mean_w(h)
        z_mean = F.relu(z_mean)
        y = self.label_linear_1(z_mean)
        y = F.tanh(y)
        return y  

## train the regressor only on the training data logits
sys.path.append('%s/../util/' % os.path.dirname(os.path.realpath('__file__')))
from train_util import get_batch_input_regressor
def train_regressor(phase, epoch, optimizer, model, regressor, sample_indexs, data_binary, data_property):
    total_loss = []
    pbar = tqdm(range(0, (len(sample_indexs) + (cmd_args.batch_size - 1) * (optimizer is None)) // cmd_args.batch_size), unit='batch')
    
    if phase == 'train' and optimizer is not None:
        regressor.train()
    else:
        regressor.eval()
        
    n_samples = 0 
    for pos in pbar:
        selected_idx = sample_indexs[pos * cmd_args.batch_size : (pos + 1) * cmd_args.batch_size]
        x_inputs, y_inputs,v_x, t_y = get_batch_input_regressor(selected_idx, data_binary, data_property)  # no grad for evaluate mode.              
        z_mean, z_log_var = model.encoder(x_inputs)
        #z = model.reparameterize(z_mean,  z_log_var)
        z = z_mean
        ## p(x|z, y')
        rex_logits_ = model.state_decoder(z, y_inputs)
        
        '''
        smiles, index, y_reconstructedx = raw_logit_to_smile_labels(rex_logits_.detach())
        if smiles ==[]:
            print('smiles is empty')
            pred_y = 0
            negative_reward  = 0
        else:
            true_y = t_y[index]
            rex_logits = rex_logits_[:,index, :]
            pred_y = regressor(rex_logits.permute(1,2,0))
            negative_reward = (pred_y.float().view(-1) - true_y).norm()
            minibatch_loss = negative_reward.data.cpu().numpy()
            #pbar.set_description('At epoch: %d  regressor loss: %0.5f' % (epoch, minibatch_loss))
            pbar.set_description('At epoch: %d  regressor loss: %0.5f  reg1: %0.5f' % (epoch, minibatch_loss, negative_reward))
       '''
        pred_y = regressor(rex_logits_.permute(1,2,0))
        true_y = t_y
        negative_reward = (pred_y.float().view(-1) - true_y).norm()
        minibatch_loss = negative_reward.data.cpu().numpy()
        #pbar.set_description('At epoch: %d  regressor loss: %0.5f' % (epoch, minibatch_loss))
        pbar.set_description('At epoch: %d  regressor loss: %0.5f  reg1: %0.5f' % (epoch, minibatch_loss, negative_reward))
        if optimizer is not None:
            assert len(selected_idx) == cmd_args.batch_size
            optimizer.zero_grad()
            negative_reward.backward()
            optimizer.step()
            
        total_loss.append( np.array([minibatch_loss]) * len(selected_idx))
        n_samples += len(selected_idx) 
        
    total_loss = np.sum(np.array(total_loss), 0)
    
    return regressor, total_loss/n_samples, minibatch_loss
 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
cmd_args.num_epochs = 300
if not os.path.exists(cmd_args.regressor_saved_dir):
    os.makedirs(cmd_args.regressor_saved_dir)

sys.path.append('%s/../util/' % os.path.dirname(os.path.realpath('__file__')))    
from train_util import Prepare_data
getting_data = Prepare_data(cmd_args)

train_binary_x, train_masks, valid_binary_x, valid_masks, train_y, valid_y = getting_data.load_data()
print('num_train: %d\tnum_valid: %d' % (train_y.shape[0], valid_binary_x.shape[0]))    
    
ae = MolVAE()  
if cmd_args.mode == 'gpu':
    ae = ae.cuda()
regressor = Regressor(cmd_args.max_decode_steps, cmd_args.latent_dim)
regressor.cuda()

optimizer_regressor = optim.Adam(regressor.parameters(), lr=cmd_args.learning_rate)
lr_scheduler = ReduceLROnPlateau(optimizer_regressor, 'min', factor=0.5, patience=5, verbose=True, min_lr=0.001) 

# load the pretrained vae model
vae_pretrained_model = cmd_args.vanilla_vae_save_dir + '/epoch-best.model'
if vae_pretrained_model  is not None and vae_pretrained_model != '':
        if os.path.isfile(vae_pretrained_model):
            print('loading model from %s' % vae_pretrained_model)
            ae.load_state_dict(torch.load(vae_pretrained_model))

            

sample_idxes = list(range(train_binary_x.shape[0]))
best_valid_loss = None
for epoch in range(cmd_args.num_epochs):
    random.shuffle(sample_idxes)
    
    ## update the reggressor:
    regressor, regressor_loss, reg_1= train_regressor('train', epoch, optimizer_regressor, ae, regressor, sample_idxes,  train_binary_x, train_y)
    print('>>>>average regressor \033[92mtraining\033[0m of epoch %d: average loss %.5f, minibatch %.5f' % (epoch, regressor_loss, reg_1))  
    
    if epoch % 1 == 0:
        _, valid_loss, _= train_regressor('valid', epoch,  None, ae,  regressor, list(range(valid_binary_x.shape[0])), valid_binary_x, valid_y)
        print('>>>>average regressor \033[92mtraining\033[0m of epoch %d: average loss %.5f, minibatch %.5f' % (epoch, regressor_loss, reg_1))  
        lr_scheduler.step(valid_loss)
        torch.save(regressor.state_dict(), cmd_args.regressor_saved_dir + '/epoch-%d.model' % epoch)
    
        if best_valid_loss is None or valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print('saving to best model since this is the best valid loss so far.----')
            torch.save(regressor.state_dict(), cmd_args.regressor_saved_dir + '/epoch-best.model')
            
            