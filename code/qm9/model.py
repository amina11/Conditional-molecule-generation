## main model is defined here
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



#1. Define regressor
from pytorch_initializer import weights_init
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
  
# VAE model
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

def Deterministic_policy_regularizer(model, regressor, x_inputs, y_inputs, true_binary, rule_masks, t_y):   
        #q(z|x,y)
        z_mean, z_log_var = model.encoder(x_inputs)
        z = model.reparameterize(z_mean, z_log_var)
        ## Use new label for the regressor
        batch_size = y_inputs.shape[0]
        
        #p(x|z, y), decode with the true label
        rex_logits = model.state_decoder(z, t_y)
        pred_y = regressor(rex_logits.permute(1,2,0))
        original_reward = (pred_y.float().view(-1) - t_y).norm()  
        
        #p(x|z, y'), decode the z with the new label
        perm = torch.randperm(batch_size)
        permuted_y_ =  y_inputs[perm]
        permuted_ty_ = t_y[perm]
        rex_logits_permuted = model.state_decoder(z, permuted_ty_)
        pred_y_permuted = regressor(rex_logits_permuted.permute(1,2,0)) 
        permuted_reward = (pred_y_permuted.float().view(-1) - permuted_ty_).norm() 
        '''
        smiles, index, y_reconstrucedx = raw_logit_to_smile_labels(rex_logits_.detach())
        if smiles ==[]:
            print('smiles is empty')
            pred_y = 0
            negative_reward  = 0
        else:
            ## remove equations which has 0/0, returns nan
            #true_y = torch.gather(permuted_ty_, 0, torch.from_numpy(index).cuda().long())
            true_y = permuted_ty_[index]
            rex_logits = rex_logits_[:,index, :]
            ## reward
            pred_y = regressor(rex_logits.permute(1,2,0)) 
        '''    
        return original_reward,  permuted_reward       

# encoder and decoder
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

        # train vae model for one epoch
from pytorch_initializer import weights_init
from train_util import get_batch_input_vae
#q(z|x,y)
def epoch_train(phase, epoch, ae, regressor, sample_idxes, data_binary, data_masks, data_property, cmd_args, optimizer_vae=None, optimizer_regularizer = None):
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
        perp = loss_list[0].data.cpu().numpy()[0] # reconstruction loss
            
        if epoch > -1:
            original_reward,  permuted_reward = Deterministic_policy_regularizer(ae, regressor, x_inputs, y_inputs,v_tb,v_ms, t_y)
            print(cmd_args.RL_param)
            loss_vae = loss_list[0] + loss_list[1] + cmd_args.RL_param * original_reward
            loss_regualrizer = loss_list[0] + loss_list[1] + cmd_args.RL_param * (original_reward + permuted_reward)
            kl = loss_list[1].data.cpu().numpy()           
        else:
            original_reward = 0
            permuted_reward = 0
            loss_vae = loss_list[0] + loss_list[1]
            loss_regualrizer = loss_list[0] + loss_list[1]
            kl = loss_list[1].data.cpu().numpy()
        

        minibatch_vae_loss = loss_vae.data.cpu().numpy()
        minibatch_regularizer_loss = loss_regualrizer.data.cpu().numpy()
        pbar.set_description('At epoch: %d  %s vae loss: %0.5f regularizer loss: %0.5f original reward: %0.5f  permuted reward: %0.5f perp: %0.5f kl: %0.5f' % (epoch, phase, minibatch_vae_loss, minibatch_regularizer_loss, original_reward, permuted_reward, perp, kl))
        

        if optimizer_vae is not None:
            assert len(selected_idx) == cmd_args.batch_size
            optimizer_vae.zero_grad()
            loss_vae.backward(retain_graph=True)
            optimizer_vae.step()
            
            
        if optimizer_regularizer is not None:
            assert len(selected_idx) == cmd_args.batch_size
            optimizer_regularizer.zero_grad()
            loss_regualrizer.backward()
            optimizer_regularizer.step()    

        total_loss.append(np.array([minibatch_vae_loss, minibatch_regularizer_loss,  original_reward, permuted_reward, perp, kl]) * len(selected_idx))
       
        n_samples += len(selected_idx)
        
    if optimizer_vae is None:
        assert n_samples == len(sample_idxes)  
        
    total_loss = np.array(total_loss)

    avg_loss = np.sum(total_loss, 0) / n_samples   
    return ae, avg_loss



## train the regressor only on the training data logits
from train_util import get_batch_input_regressor
def train_regressor(epoch, optimizer, model, regressor, sample_indexs, data_binary, data_property):
    total_loss = []
    pbar = tqdm(range(0, (len(sample_indexs) + (cmd_args.batch_size - 1) * (optimizer is None)) // cmd_args.batch_size), unit='batch')
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
        
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    
    return regressor, avg_loss, negative_reward
    