import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

TAU = 1e-3          # TAU determines how much of local network used to update target network

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class CriticNetwork(nn.Module):
    
    def __init__(self, state_size, action_size, seed):

        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.critic_layer_1 = nn.Linear(state_size*2+action_size*2, 128)
        self.critic_layer_2 = nn.Linear(128, 128)
        self.critic_out = nn.Linear(128, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.critic_layer_1.weight.data.uniform_(*hidden_init(self.critic_layer_1))
        self.critic_layer_2.weight.data.uniform_(*hidden_init(self.critic_layer_2))
        self.critic_out.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state, extra_info):
#         print(state.size(), action.size())

        state = torch.cat([state, extra_info], dim=1)
        x = F.relu(self.critic_layer_1(state))
        x = F.relu(self.critic_layer_2(x))
        return self.critic_out(x)

class ActorNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed):

        super(ActorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.actor_layer_1 = nn.Linear(state_size, 256)
        self.actor_layer_2 = nn.Linear(256, 128)
        self.actor_out = nn.Linear(128, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.actor_layer_1.weight.data.uniform_(*hidden_init(self.actor_layer_1))
        self.actor_layer_2.weight.data.uniform_(*hidden_init(self.actor_layer_2))
        self.actor_out.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state):
        x = F.leaky_relu(self.actor_layer_1(state))
        x = F.leaky_relu(self.actor_layer_2(x))
        return torch.tanh(self.actor_out(x))

class Network:
    def __init__(self, state_size, action_size, seed):
        self.seed = seed
        self.actor = ActorNetwork(state_size, action_size, seed)
        self.critic = CriticNetwork(state_size, action_size, seed)

    def copy(self, network):
        # Actor
        for target_param, local_param in zip(self.actor.parameters(), network.actor.parameters()):
            target_param.data.copy_(local_param.data)
        
        # Critic
        for target_param, local_param in zip(self.critic.parameters(), network.critic.parameters()):
            target_param.data.copy_(local_param.data)
    
    def soft_update(self, network, tau=TAU):
        
        # Actor
        for target_param, local_param in zip(self.actor.parameters(), network.actor.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        
        # Critic
        for target_param, local_param in zip(self.critic.parameters(), network.critic.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
