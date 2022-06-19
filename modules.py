import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.autograd import Variable
from utils import my_softmax, get_offdiag_indices, gumbel_softmax

_EPS = 1e-10

"""
• Agent node embedding function: for each different node type, a distinct two-layer gated
  recurrent unit (GRU) with hidden size = 128.
• Context node embedding function: four-layer convolutional blocks with kernel size = 5 and
  padding = 3. The structure is [[Conv, ReLU, Conv, ReLU, Pool], [Conv, ReLU, Conv, ReLU,
  Pool]].
• Agent node update function: a three-layer MLP with hidden size = 128.
• Edge update function: for both agent-agent edges and agent-context edges, a distinct threelayer MLP with hidden size = 128.
• Encoding function: a three-layer MLP with hidden size = 128.
• Decoding function: a two-layer gated recurrent unit (GRU) with hidden size = 128.
• Recurrent graph evolution module: a two-layer GRU with hidden size = 256
"""

class AgentNodeEmbedding(nn.Module):
    """ for each different node type, a distinct two-layer gated
	    recurrent unit (GRU) with hidden size = 128. """

    def __init__(self, n_in, n_hid, n_out):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)

#         self.init_weights()

#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_normal(m.weight.data)
#                 m.bias.data.fill_(0.1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.elu(self.fc2(x))
        return x


class AgentNodeUpdate(nn.Module):
    """ for each different node type, a distinct two-layer gated
	    recurrent unit (GRU) with hidden size = 128. """

    def __init__(self, n_in, n_hid, n_out):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, n_out)
        
    def forward(self, inputs):
        x = F.elu(self.fc1(inputs))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        return x
    

class EdgeUpdate(nn.Module):
    """ for each different node type, a distinct two-layer gated
	    recurrent unit (GRU) with hidden size = 128. """

    def __init__(self, n_in, n_hid, n_out):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, n_out)
        
    def forward(self, inputs):
        x = F.elu(self.fc1(inputs))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        return x
    

class Encoding(nn.Module):
    """ for each different node type, a distinct two-layer gated
	    recurrent unit (GRU) with hidden size = 128. """

    def __init__(self, n_in, n_hid, n_out):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, n_out)
        
    def forward(self, inputs):
        x = F.elu(self.fc1(inputs))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        return x
    
class Decoding(nn.Module):
    """ for each different node type, a distinct two-layer gated
	    recurrent unit (GRU) with hidden size = 128. """

    def __init__(self, n_in, n_hid, n_out):
        super(MLP, self).__init__()
        self.g1 = nn.GRU(n_in, n_hid)
        self.g2 = nn.GRU(n_hid, n_out)
        
    def forward(self, inputs):
        x = F.elu(self.g1(inputs))
        x = F.elu(self.g2(x))
        return x
    
class RecurrentGraphEvolution(nn.Module):
    """ for each different node type, a distinct two-layer gated
	    recurrent unit (GRU) with hidden size = 128. """

    def __init__(self, n_in, n_hid, n_out):
        super(MLP, self).__init__()
        self.g1 = nn.Linear(n_in, n_hid)
        self.g2 = nn.Linear(n_hid, n_out)
        
    def forward(self, inputs):
        x = F.elu(self.g1(inputs))
        x = F.elu(self.g2(x))
        return x
