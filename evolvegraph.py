import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from . import model_utils
from .model_utils import RefNRIMLP, encode_onehot
import math


class EvolveGraph(nn.Module):
    def __init__(self, params):
        super(EvolveGraph, self).__init__()
        
        num_edge_types = params['num_edge_types']
        encoding_funcs = [Encoding() for i in range(num_edge_types)]
        pass
    

    def single_step_forward(self, inputs, decoder_hidden, edge_logits, hard_sample):
        passs
        
    def calculate_loss(self, inputs, is_train=False, teacher_forcing=True, return_edges=False, return_logits=False, use
        for n in N: # particles
            for t in T: # timesteps not included as input
                for k in K: # gaussian mixture components
                       # w_i^(i, k) * log p_t^(i, k)

    def predict_future(self, inputs, prediction_steps, return_edges=False, return_everything=False):
        pass

    def copy_states(self, state):
        pass
        
    def merge_hidden(self, hidden):
        pass

    def predict_future_fixedwindow(self, inputs, burn_in_steps, prediction_steps, batch_size, return_edges=False):
        pass

    def nll(self, preds, target):
        pass

    def nll_gaussian(self, preds, target, add_const=False):
        pass

    def nll_crossent(self, preds, target):
        pass

    def nll_poisson(self, preds, target):
        pass

    def kl_categorical_learned(self, preds, prior_logits):
        pass

    def kl_categorical_avg(self, preds, eps=1e-16):
        pass
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

class DNRI_Encoder(nn.Module):
    # Here, encoder also produces prior
    def __init__(self, params):
        super(DNRI_Encoder, self).__init__()
        num_vars = params['num_vars']
        self.num_edges = params['num_edge_types']
        self.sepaate_prior_encoder = params.get('separate_prior_encoder', False)
        no_bn = False
        dropout = params['encoder_dropout']
        edges = np.ones(num_vars) - np.eye(num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.edge2node_mat = nn.Parameter(torch.FloatTensor(encode_onehot(self.recv_edges).transpose()), requires_grad=False)
        self.save_eval_memory = params.get('encoder_save_eval_memory', False)


        hidden_size = params['encoder_hidden']
        rnn_hidden_size = params['encoder_rnn_hidden']
        rnn_type = params['encoder_rnn_type']
        inp_size = params['input_size']
        self.mlp1 = RefNRIMLP(inp_size, hidden_size, hidden_size, dropout, no_bn=no_bn)
        self.mlp2 = RefNRIMLP(hidden_size * 2, hidden_size, hidden_size, dropout, no_bn=no_bn)
        self.mlp3 = RefNRIMLP(hidden_size, hidden_size, hidden_size, dropout, no_bn=no_bn)

        if rnn_hidden_size is None:
            rnn_hidden_size = hidden_size
        if rnn_type == 'lstm':
            self.forward_rnn = nn.LSTM(hidden_size, rnn_hidden_size, batch_first=True)
            self.reverse_rnn = nn.LSTM(hidden_size, rnn_hidden_size, batch_first=True)
        elif rnn_type == 'gru':
            self.forward_rnn = nn.GRU(hidden_size, rnn_hidden_size, batch_first=True)
            self.reverse_rnn = nn.GRU(hidden_size, rnn_hidden_size, batch_first=True)
        out_hidden_size = 2*rnn_hidden_size
        num_layers = params['encoder_mlp_num_layers']
        if num_layers == 1:
            self.encoder_fc_out = nn.Linear(out_hidden_size, self.num_edges)
        else:
            tmp_hidden_size = params['encoder_mlp_hidden']
            layers = [nn.Linear(out_hidden_size, tmp_hidden_size), nn.ELU(inplace=True)]
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(tmp_hidden_size, tmp_hidden_size))
                layers.append(nn.ELU(inplace=True))
            layers.append(nn.Linear(tmp_hidden_size, self.num_edges))
            self.encoder_fc_out = nn.Sequential(*layers)

        num_layers = params['prior_num_layers']
        if num_layers == 1:
            self.prior_fc_out = nn.Linear(rnn_hidden_size, self.num_edges)
        else:
            tmp_hidden_size = params['prior_hidden_size']
            layers = [nn.Linear(rnn_hidden_size, tmp_hidden_size), nn.ELU(inplace=True)]
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(tmp_hidden_size, tmp_hidden_size))
                layers.append(nn.ELU(inplace=True))
            layers.append(nn.Linear(tmp_hidden_size, self.num_edges))
            self.prior_fc_out = nn.Sequential(*layers)

        self.num_vars = num_vars
        edges = np.ones(num_vars) - np.eye(num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge(self, node_embeddings):
        # Input size: [batch, num_vars, num_timesteps, embed_size]
        if len(node_embeddings.shape) == 4:
            send_embed = node_embeddings[:, self.send_edges, :, :]
            recv_embed = node_embeddings[:, self.recv_edges, :, :]
        else:
            send_embed = node_embeddings[:, self.send_edges, :]
            recv_embed = node_embeddings[:, self.recv_edges, :]
        return torch.cat([send_embed, recv_embed], dim=-1)

    def edge2node(self, edge_embeddings):
        if len(edge_embeddings.shape) == 4:
            old_shape = edge_embeddings.shape
            tmp_embeddings = edge_embeddings.view(old_shape[0], old_shape[1], -1)
            incoming = torch.matmul(self.edge2node_mat, tmp_embeddings).view(old_shape[0], -1, old_shape[2], old_shape[3])
        else:
            incoming = torch.matmul(self.edge2node_mat, edge_embeddings)
        return incoming/(self.num_vars-1) #TODO: do we want this average?


    def copy_states(self, prior_state):
        if isinstance(prior_state, tuple) or isinstance(prior_state, list):
            current_prior_state = (prior_state[0].clone(), prior_state[1].clone())
        else:
            current_prior_state = prior_state.clone()
        return current_prior_state

    def merge_hidden(self, hidden):
        if isinstance(hidden[0], tuple) or isinstance(hidden[0], list):
            result0 = torch.cat([x[0] for x in hidden], dim=0)
            result1 = torch.cat([x[1] for x in hidden], dim=0)
            result = (result0, result1)
        else:
            result = torch.cat(hidden, dim=0)
        return result



    def forward(self, inputs):
        if self.training or not self.save_eval_memory:
            # Inputs is shape [batch, num_timesteps, num_vars, input_size]
            num_timesteps = inputs.size(1)
            x = inputs.transpose(2, 1).contiguous()
            # New shape: [num_sims, num_atoms, num_timesteps, num_dims]
            # print(x.shape)
            x = self.mlp1(x)  # 2-layer ELU net per node
            x = self.node2edge(x)
            x = self.mlp2(x)
            x_skip = x
            x = self.edge2node(x)
            x = self.mlp3(x)
            x = self.node2edge(x)
            x = torch.cat((x, x_skip), dim=-1)  # Skip connection
            x = self.mlp4(x)
        
            
            # At this point, x should be [batch, num_edges, num_timesteps, hidden_size]
            # RNN aggregation
            old_shape = x.shape
            x = x.contiguous().view(-1, old_shape[2], old_shape[3])
            forward_x, prior_state = self.forward_rnn(x)
            timesteps = old_shape[2]
            reverse_x = x.flip(1)
            reverse_x, _ = self.reverse_rnn(reverse_x)
            reverse_x = reverse_x.flip(1)
            
            #x: [batch*num_edges, num_timesteps, hidden_size]
            prior_result = self.prior_fc_out(forward_x).view(old_shape[0], old_shape[1], timesteps, self.num_edges).transpose(1,2).contiguous()
            combined_x = torch.cat([forward_x, reverse_x], dim=-1)
            encoder_result = self.encoder_fc_out(combined_x).view(old_shape[0], old_shape[1], timesteps, self.num_edges).transpose(1,2).contiguous()
            return prior_result, encoder_result, prior_state
        else:
            # Inputs is shape [batch, num_timesteps, num_vars, input_size]
            num_timesteps = inputs.size(1)
            all_x = []
            all_forward_x = []
            all_prior_result = []
            prior_state = None
            for timestep in range(num_timesteps):
                x = inputs[:, timestep]
                #x = inputs.transpose(2, 1).contiguous()
                x = self.mlp1(x)  # 2-layer ELU net per node
                x = self.node2edge(x)
                x = self.mlp2(x)
                x_skip = x
                x = self.edge2node(x)
                x = self.mlp3(x)
                x = self.node2edge(x)
                x = torch.cat((x, x_skip), dim=-1)  # Skip connection
                x = self.mlp4(x)
            
                
                # At this point, x should be [batch, num_edges, num_timesteps, hidden_size]
                # RNN aggregation
                old_shape = x.shape
                x = x.contiguous().view(-1, 1, old_shape[-1])
                forward_x, prior_state = self.forward_rnn(x, prior_state)
                all_x.append(x.cpu())
                all_forward_x.append(forward_x.cpu())
                all_prior_result.append(self.prior_fc_out(forward_x).view(old_shape[0], 1, old_shape[1], self.num_edges).cpu())
            reverse_state = None
            all_encoder_result = []
            for timestep in range(num_timesteps-1, -1, -1):
                x = all_x[timestep].cuda()
                reverse_x, reverse_state = self.reverse_rnn(x, reverse_state)
                forward_x = all_forward_x[timestep].cuda()
                
                #x: [batch*num_edges, num_timesteps, hidden_size]
                combined_x = torch.cat([forward_x, reverse_x], dim=-1)
                all_encoder_result.append(self.encoder_fc_out(combined_x).view(inputs.size(0), 1, -1, self.num_edges))
            prior_result = torch.cat(all_prior_result, dim=1).cuda(non_blocking=True)
            encoder_result = torch.cat(all_encoder_result, dim=1).cuda(non_blocking=True)
            return prior_result, encoder_result, prior_state

    def single_step_forward(self, inputs, prior_state):
        # Inputs is shape [batch, num_vars, input_size]
        x = self.mlp1(inputs)  # 2-layer ELU net per node
        x = self.node2edge(x)
        x = self.mlp2(x)
        x_skip = x
        x = self.edge2node(x)
        x = self.mlp3(x)
        x = self.node2edge(x)
        x = torch.cat((x, x_skip), dim=-1)  # Skip connection
        x = self.mlp4(x)

        old_shape = x.shape
        x  = x.contiguous().view(-1, 1, old_shape[-1])
        old_prior_shape = prior_state[0].shape
        prior_state = (prior_state[0].view(1, old_prior_shape[0]*old_prior_shape[1], old_prior_shape[2]),
                       prior_state[1].view(1, old_prior_shape[0]*old_prior_shape[1], old_prior_shape[2]))

        x, prior_state = self.forward_rnn(x, prior_state)
        prior_result = self.prior_fc_out(x).view(old_shape[0], old_shape[1], self.num_edges)
        prior_state = (prior_state[0].view(old_prior_shape), prior_state[1].view(old_prior_shape))
        return prior_result, prior_state

    
class DNRI_Decoder(nn.Module):
    def __init__(self, params):
        super(DNRI_Decoder, self).__init__()
        self.num_vars = num_vars =  params['num_vars']
        input_size = params['input_size']
        self.gpu = params['gpu']
        n_hid = params['decoder_hidden']
        edge_types = params['num_edge_types']
        skip_first = params['skip_first']
        out_size = params['input_size']
        do_prob = params['decoder_dropout']

        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2*n_hid, n_hid) for _ in range(edge_types)]
        )
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(n_hid, n_hid) for _ in range(edge_types)]
        )
        self.msg_out_shape = n_hid
        self.skip_first_edge_type = skip_first

        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)

        self.input_r = nn.Linear(input_size, n_hid, bias=True)
        self.input_i = nn.Linear(input_size, n_hid, bias=True)
        self.input_n = nn.Linear(input_size, n_hid, bias=True)

        self.out_fc1 = nn.Linear(n_hid, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, out_size)

        print('Using learned recurrent interaction net decoder.')

        self.dropout_prob = do_prob

        self.num_vars = num_vars
        edges = np.ones(num_vars) - np.eye(num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.edge2node_mat = nn.Parameter(torch.FloatTensor(encode_onehot(self.recv_edges)), requires_grad=False)

    def get_initial_hidden(self, inputs):
        return torch.zeros(inputs.size(0), inputs.size(2), self.msg_out_shape, device=inputs.device)
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, inputs, hidden, edges):
        # Input Size: [batch, num_vars, input_size]
        # Hidden Size: [batch, num_vars, rnn_hidden]
        # Edges size: [batch, num_edges, num_edge_types]
        if self.training:
            dropout_prob = self.dropout_prob
        else:
            dropout_prob = 0.
        
        # node2edge
        receivers = hidden[:, self.recv_edges, :]
        senders = hidden[:, self.send_edges, :]

        # pre_msg: [batch, num_edges, 2*msg_out]
        pre_msg = torch.cat([receivers, senders], dim=-1)

        all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                        self.msg_out_shape, device=inputs.device)
        
        if self.skip_first_edge_type:
            start_idx = 1
            norm = float(len(self.msg_fc2)) - 1
        else:
            start_idx = 0
            norm = float(len(self.msg_fc2))

        # Run separate MLP for every edge type
        # NOTE: to exclude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = torch.tanh(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=dropout_prob)
            msg = torch.tanh(self.msg_fc2[i](msg))
            msg = msg * edges[:, :, i:i+1]
            all_msgs += msg/norm

        # This step sums all of the messages per node
        agg_msgs = all_msgs.transpose(-2, -1).matmul(self.edge2node_mat).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous() / (self.num_vars - 1) # Average

        # GRU-style gated aggregation
        inp_r = self.input_r(inputs).view(inputs.size(0), self.num_vars, -1)
        inp_i = self.input_i(inputs).view(inputs.size(0), self.num_vars, -1)
        inp_n = self.input_n(inputs).view(inputs.size(0), self.num_vars, -1)
        r = torch.sigmoid(inp_r + self.hidden_r(agg_msgs))
        i = torch.sigmoid(inp_i + self.hidden_i(agg_msgs))
        n = torch.tanh(inp_n + r*self.hidden_h(agg_msgs))
        hidden = (1 - i)*n + i*hidden

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(hidden)), p=dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=dropout_prob)
        pred = self.out_fc3(pred)

        pred = inputs + pred

        return pred, hidden


class DNRI_MLP_Decoder(nn.Module):
    def __init__(self, params):
        super(DNRI_MLP_Decoder, self).__init__()
        num_vars = params['num_vars']
        edge_types = params['num_edge_types']
        n_hid = params['decoder_hidden']
        msg_hid = params['decoder_hidden']
        msg_out = msg_hid #TODO: make this a param
        skip_first = params['skip_first']
        n_in_node = params['input_size']

        do_prob = params['decoder_dropout']
        in_size = n_in_node
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * in_size, msg_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])
        self.msg_out_shape = msg_out
        self.skip_first_edge_type = skip_first

        out_size = n_in_node
        self.out_fc1 = nn.Linear(in_size + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, out_size)

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob
        self.num_vars = num_vars
        edges = np.ones(num_vars) - np.eye(num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.edge2node_mat = nn.Parameter(torch.FloatTensor(encode_onehot(self.recv_edges)), requires_grad=False)

    def get_initial_hidden(self, inputs):
        return None

    def forward(self, inputs, hidden, edges):

        # single_timestep_inputs has shape
        # [batch_size, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_atoms*(num_atoms-1), num_edge_types]
        # Node2edge
        receivers = inputs[:, self.recv_edges, :]
        senders = inputs[:, self.send_edges, :]
        pre_msg = torch.cat([receivers, senders], dim=-1)

        if inputs.is_cuda:
            all_msgs = torch.cuda.FloatTensor(pre_msg.size(0), pre_msg.size(1),
                                self.msg_out_shape).fill_(0.)
        else:
            all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                self.msg_out_shape)

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0
        if self.training:
            p = self.dropout_prob
        else:
            p = 0

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=p)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * edges[:, :, i:i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(self.edge2node_mat).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=p)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=p)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        return inputs + pred, None