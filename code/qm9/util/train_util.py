## this functions are called from main training fucntion
import numpy as np
import h5py
import sys, os
import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw
from joblib import Parallel, delayed
from rdkit.Chem import Descriptors, QED


sys.path.append('%s/./util/' % os.path.dirname(os.path.realpath('__file__')))
from cmd_args import cmd_args
import cfg_parser as parser
# prepare the data set
#load the training data and seperate train and validation set
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

# prepare mini batch input for each epoch training of regressor   
def get_batch_input_regressor(selected_idx, x_train, y_train, volatile=False):
   
    true_binary = np.transpose(x_train[selected_idx, :, :], [1, 0, 2]).astype(np.float32)
    x_inputs = np.transpose(true_binary, [1, 2, 0])
    y_inputs = (y_train[selected_idx]).astype(np.float32)
        
    v_x = torch.from_numpy(x_inputs)
    v_y = torch.from_numpy(y_inputs)
        
    if cmd_args.mode == 'gpu':
        v_x = v_x.cuda()
        v_y = v_y.cuda()

    x_variable = Variable(v_x, volatile=volatile)
    y_variable = Variable(v_y, volatile = volatile)

    return x_inputs, y_inputs, x_variable, y_variable     

# prepare mini batch input for each epoch training of vae       
def get_batch_input_vae(selected_idx, data_binary, data_masks, data_property):
    true_binary = np.transpose(data_binary[selected_idx, :, :], [1, 0, 2]).astype(np.float32)
    rule_masks = np.transpose(data_masks[selected_idx, :, :], [1, 0, 2]).astype(np.float32)
    x_inputs = np.transpose(true_binary, [1, 2, 0])
    y_inputs = (data_property[selected_idx]).astype(np.float32)

    t_b = torch.from_numpy(true_binary)
    t_ms = torch.from_numpy(rule_masks)
    t_y = torch.from_numpy(y_inputs)

    if cmd_args.mode == 'gpu':
        t_b = t_b.cuda()
        t_ms = t_ms.cuda()
        t_y = t_y.cuda()

    v_b = Variable(t_b)
    v_ms = Variable(t_ms)
    return x_inputs, y_inputs, v_b, v_ms, t_y    

## take raw logits from the decoder and output the smile string and lables y and also return the index of decoded Nan smiles
sys.path.append('%s/./util/' % os.path.dirname(os.path.realpath('__file__')))
from mol_tree import AnnotatedTree2MolTree, get_smiles_from_tree, Node
from tree_walker import ConditionalDecoder
from attribute_tree_decoder import create_tree_decoder
import sascorer
def raw_logit_to_smile_labels(raw_logits, use_random = False):
    y = []
    index = []
    result_list = []
    for i in range(raw_logits.shape[1]):
        pred_logits = raw_logits[:, i, :]
        walker = ConditionalDecoder(np.squeeze(pred_logits), use_random)
        new_t = Node('smiles')
        try:
            tree_decoder = create_tree_decoder()
            tree_decoder.decode(new_t, walker)
            sampled = get_smiles_from_tree(new_t)
            
        except Exception as ex:
            if not type(ex).__name__ == 'DecodingLimitExceeded':
                print('Warning, decoder failed with', ex)

            # failed. output None
            sampled = None
                
        if sampled is None:
            continue  
                    
        mol = Chem.MolFromSmiles(sampled)
        
        ## decoded smile is not valid molecule
        if mol is None:
            continue
            
        logP = Descriptors.MolLogP(mol)
        sa_score = sascorer.calculateScore(mol)
        qed = QED.qed(mol) 
        y.append([qed, sa_score, logP])         
        result_list.append(sampled)
        index.append(i)
    return result_list, index, y        


## from siiles string to one hot encoding and masks
sys.path.append('%s/./util/' % os.path.dirname(os.path.realpath('__file__')))
from tree_walker import OnehotBuilder, ConditionalDecoder
from batch_make_att_masks import batch_make_att_masks
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


from mol_util import rule_ranges, terminal_idxes, DECISION_DIM
def run_job(L):
    chunk_size = 5000

    list_binary = Parallel(n_jobs=cmd_args.data_gen_threads, verbose=50)(
        delayed(process_chunk)(L[start: start + chunk_size])
        for start in range(0, len(L), chunk_size)
    )
    

    all_onehot = np.zeros((len(L), cmd_args.max_decode_steps, DECISION_DIM), dtype=np.byte)
    all_masks = np.zeros((len(L), cmd_args.max_decode_steps, DECISION_DIM), dtype=np.byte)

    for start, b_pair in zip( range(0, len(L), chunk_size), list_binary ):
        all_onehot[start: start + chunk_size, :, :] = b_pair[0]
        all_masks[start: start + chunk_size, :, :] = b_pair[1]
        
    return all_onehot, all_masks




    # loss    
from custom_loss import my_perp_loss, my_binary_loss    
class PerpCalculator(nn.Module):
    def __init__(self):
        super(PerpCalculator, self).__init__()

    '''
    input:
        true_binary: one-hot, with size=time_steps x bsize x DECISION_DIM
        rule_masks: binary tensor, with size=time_steps x bsize x DECISION_DIM
        raw_logits: real tensor, with size=time_steps x bsize x DECISION_DIM
    '''
    def forward(self, true_binary, rule_masks, raw_logits):
        if cmd_args.loss_type == 'binary':
            exp_pred = torch.exp(raw_logits) * rule_masks

            norm = F.torch.sum(exp_pred, 2, keepdim=True)
            prob = F.torch.div(exp_pred, norm)

            return F.binary_cross_entropy(prob, true_binary) * cmd_args.max_decode_steps

        if cmd_args.loss_type == 'perplexity':
            return my_perp_loss(true_binary, rule_masks, raw_logits)

        if cmd_args.loss_type == 'vanilla':
            exp_pred = torch.exp(raw_logits) * rule_masks + 1e-30
            norm = torch.sum(exp_pred, 2, keepdim=True)
            prob = torch.div(exp_pred, norm)

            ll = F.torch.abs(F.torch.sum( true_binary * prob, 2))
            mask = 1 - rule_masks[:, :, -1]
            logll = mask * F.torch.log(ll)

            loss = -torch.sum(logll) / true_binary.size()[1]
            
            return loss
        print('unknown loss type %s' % cmd_args.loss_type)
        raise NotImplementedError 
        
        
'''
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
    
    
    
''' 