import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import os
from class_ import pathconfig
from torch.autograd import Variable

paths = pathconfig.paths()  #paths class

class Critic(nn.Module):
    """
    Critic class. It presents three methods:
    -forward: takes state and action as parameters, builds critic network and return output
    -save actor model
    -load actor model
    """
    def __init__(self, input_size, hidden_size,output_size,name,chkpt_dir=paths.MODELS_DDPG):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self._ddpg = '_ddpg'
        self.checkpoint_file = os.path.join(chkpt_dir, name + self._ddpg)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))

class Actor(nn.Module):
    """
       Actor class. It presents three methods:
       -forward: takes state and action as parameters, builds actor network and return output
       -save actor model
       -load actor model
    """
    def __init__(self, input_size, hidden_size, output_size, name,chkpt_dir=paths.MODELS_DDPG):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self._ddpg = '_ddpg'
        self.checkpoint_file = os.path.join(chkpt_dir, name + self._ddpg)
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))