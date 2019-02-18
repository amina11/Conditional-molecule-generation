# 2019.1.16
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

sys.path.append('%s/util/' % os.path.dirname(os.path.realpath('__file__')))
from cmd_args import cmd_args
import cfg_parser as parser

# parameters to be specified
#cmd_args.save_dir = './vanilla_supervised_vae'
#cmd_args.training_data_dir = '/work/molecule_generation/sdvae/mol_vae_clean/zinc_data_experiments/data/data_278'



cmd_args.output_dim = 1
cmd_args.batch_size = 300
cmd_args.num_epochs= 500
cmd_args.output_dim = 1
cmd_args.kl_coeff = 1
seed = 19260817

class Prepare_data():
    def __init__(self, cmd_args):
        self.data_dir = cmd_args.training_data_dir + '/QM9_clean_smi_train.h5'
        self.label_dir = cmd_args.training_data_dir + '/QM9_normalized_train_y.npy'
  
    def load_data(self):
        h5f = h5py.File(self.data_dir, 'r')
        all_true_binary = h5f['x_train'][:]
        all_rule_masks = h5f['masks_train'][:]
        h5f.close()
        train_label = np.load(self.label_dir)
        num_val = min([10000, int(all_true_binary.shape[0] * 0.1)])
        return all_true_binary[num_val:], all_rule_masks[num_val:], all_true_binary[0:num_val], all_rule_masks[0:num_val] , train_label[num_val:,2], train_label[0:num_val,2]

## define the model
sys.path.append('%s/util/' % os.path.dirname(os.path.realpath('__file__')))
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
        raw_logits = self.state_decoder(z, y_inputs, n_step = None)   
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

    def forward(self, z, y, n_step):

        if n_step is None:
            n_step = self.max_len

        if cmd_args.mode == 'cpu':
            y = torch.tensor(y).float()
        else:
            y = torch.tensor(y).cuda().float()

        assert len(z.size()) == 2 # assert the input is a matrix
        h = self.z_to_latent(torch.cat((z, y.view(y.shape[0],1)), 1))
        h = F.relu(h)
        rep_h = h.expand(n_step, z.size()[0], z.size()[1] + cmd_args.output_dim) # repeat along time steps
        out, _ = self.gru(rep_h) # run multi-layer gru
        logits = self.decoded_logits(out)
        return logits   


# train vae model for one epoch
sys.path.append('%s/util/' % os.path.dirname(os.path.realpath('__file__')))
from pytorch_initializer import weights_init
from train_util import get_batch_input_vae
#q(z|x,y)
def epoch_train(phase, epoch, ae, sample_idxes, data_binary, data_masks, data_property, cmd_args, optimizer_vae=None):
    total_loss = []
    pbar = tqdm(range(0, (len(sample_idxes) + (cmd_args.batch_size - 1) * (optimizer_vae is None)) // cmd_args.batch_size), unit='batch')
    
    if phase == 'train' and optimizer_vae is not None:
        ae.train()
    else:
        ae.eval()
        
    n_samples = 0    
    for pos in pbar:
        selected_idx = sample_idxes[pos * cmd_args.batch_size : (pos + 1) * cmd_args.batch_size]
        x_inputs, y_inputs,v_tb, v_ms, t_y = get_batch_input_vae(selected_idx, data_binary, data_masks, data_property)  # no grad for evaluate mode.              
        loss_list = ae.forward(x_inputs, y_inputs,v_tb,v_ms, t_y)
        loss_vae = loss_list[0] + loss_list[1]
        
        perp = loss_list[0].data.cpu().numpy()[0] # reconstruction loss
        kl = loss_list[1].data.cpu().numpy()
        

        minibatch_vae_loss = loss_vae.data.cpu().numpy()
        pbar.set_description('At epoch: %d  %s vae loss: %0.5f perp: %0.5f kl: %0.5f' % (epoch, phase, minibatch_vae_loss, perp, kl))
        

        if optimizer_vae is not None:
            assert len(selected_idx) == cmd_args.batch_size
            optimizer_vae.zero_grad()
            loss_vae.backward(retain_graph=True)
            optimizer_vae.step()
            
            
        total_loss.append(np.array([minibatch_vae_loss, perp, kl]) * len(selected_idx))
       
        n_samples += len(selected_idx)
        
    if optimizer_vae is None:
        assert n_samples == len(sample_idxes)  
        
    total_loss = np.array(total_loss)

    avg_loss = np.sum(total_loss, 0) / n_samples   
    return ae, avg_loss

def main():
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

	if not os.path.exists(cmd_args.save_dir):
	    os.makedirs(cmd_args.save_dir)    
	    
	cmd_args.saved_model = cmd_args.save_dir + '/epoch-best.model'

	sys.path.append('%s/util/' % os.path.dirname(os.path.realpath('__file__')))
	getting_data = Prepare_data(cmd_args)
	train_binary_x, train_masks, valid_binary_x, valid_masks, train_y, valid_y = getting_data.load_data()
	print('num_train: %d\tnum_valid: %d' % (train_y.shape[0], valid_binary_x.shape[0]))    
	    
	ae = MolVAE()  





	if cmd_args.mode == 'gpu':
	    ae = ae.cuda()

	kl = []
	prep = []

	assert cmd_args.encoder_type == 'cnn'    
	optimizer_vae = optim.Adam(ae.parameters(), lr=cmd_args.learning_rate)
	lr_scheduler = ReduceLROnPlateau(optimizer_vae, 'min', factor=0.5, patience=5, verbose=True, min_lr=0.001)


	sample_idxes = list(range(train_binary_x.shape[0]))
	best_valid_loss = None

	for epoch in range(cmd_args.num_epochs):
	    random.shuffle(sample_idxes)
	    
	    ## update the vae

	    ae, vae_loss = epoch_train('train',epoch, ae, sample_idxes, train_binary_x, train_masks, train_y,cmd_args, optimizer_vae)    
	    print('>>>>average \033[92mtraining\033[0m of epoch %d: loss %.5f perp %.5f kl %.5f' % (epoch, vae_loss[0], vae_loss[1], vae_loss[2]))   
	    kl.append(vae_loss[2])
	    prep. append(vae_loss[1])
	            
	    if epoch % 1 == 0:
	        _, valid_loss = epoch_train('valid', epoch,  ae, list(range(valid_binary_x.shape[0])), valid_binary_x, valid_masks, valid_y,cmd_args)
	        print('>>>>average \033[93mvalid\033[0m of epoch %d: loss %.5f perp %.5f kl %.5f' % (epoch, valid_loss[0], valid_loss[1], valid_loss[2]))
	        valid_loss = valid_loss[0]
	        lr_scheduler.step(valid_loss)
	        torch.save(ae.state_dict(), cmd_args.save_dir + '/epoch-%d.model' % epoch)
	        
	        if best_valid_loss is None or valid_loss < best_valid_loss:
	            best_valid_loss = valid_loss
	            print('saving to best model since this is the best valid loss so far.----')
	            torch.save(ae.state_dict(), cmd_args.save_dir + '/epoch-best.model')
	            
	np.save('./kl.npy', kl) 
	np.save('./prep.npy', prep)

if __name__ == '__main__':
   main()         

