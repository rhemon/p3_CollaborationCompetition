import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

TAU = 1e-3          # TAU determines how much of local network used to update target network

def hidden_init(layer):
    """
    Returns lower and upper limit for the random parameterial initalisation for given
    layer.

    :param layer: torch's neural network layer
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class CriticNetwork(nn.Module):
    
    def __init__(self, state_size, action_size, seed):
        """
        Initialize critic network.

        :param state_size: Number of information provided in the state
        :param action_size: Number of actions environment can take
        :param seed: Seed for random initialization
        """
        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.critic_layer_1 = nn.Linear(state_size, 256)
        self.critic_layer_2 = nn.Linear(256+action_size, 128)
        self.critic_layer_3 = nn.Linear(128, 128)
        self.critic_out = nn.Linear(128, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset parameters initialized for the layers in the network.
        """
        self.critic_layer_1.weight.data.uniform_(*hidden_init(self.critic_layer_1))
        self.critic_layer_2.weight.data.uniform_(*hidden_init(self.critic_layer_2))
        self.critic_layer_3.weight.data.uniform_(*hidden_init(self.critic_layer_3))
        self.critic_out.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state, actions):
        """
        Feedforward the the network with provided state and action. Returns the Q value.
        
        :param state: The state of the game
        :param action: Action taken by the agent
        """
        x = F.leaky_relu(self.critic_layer_1(state))
        x = torch.cat([x, actions], dim=1)
        x = F.leaky_relu(self.critic_layer_2(x))
        x = F.leaky_relu(self.critic_layer_3(x))
        return self.critic_out(x)

class ActorNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed):
        """
        Initialize critic network.

        :param state_size: Number of information provided in the state
        :param action_size: Number of actions environment can take
        :param seed: Seed for random initialization
        """
        super(ActorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.actor_layer_1 = nn.Linear(state_size, 256)
        self.actor_layer_2 = nn.Linear(256, 128)
        # self.actor_layer_3 = nn.Linear(128, 128)
        self.actor_out = nn.Linear(128, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset parameters initialized for the layers in the network.
        """
        self.actor_layer_1.weight.data.uniform_(*hidden_init(self.actor_layer_1))
        self.actor_layer_2.weight.data.uniform_(*hidden_init(self.actor_layer_2))
        # self.actor_layer_3.weight.data.uniform_(*hidden_init(self.actor_layer_3))
        self.actor_out.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state):
        """
        Feedforward the the network with provided state. Returns the action values.
        
        :param state: The state of the game
        """
        x = F.leaky_relu(self.actor_layer_1(state))
        x = F.leaky_relu(self.actor_layer_2(x))
        # x = F.leaky_relu(self.actor_layer_3(x))
        return torch.tanh(self.actor_out(x))

class Network:
    def __init__(self, state_size, action_size, seed):
        """
        Initializes and handles both actor and critic network for the agent.

        :param state_size: Number of information provided in the state
        :param action_size: Number of actions environment can take
        :param seed: Seed for random initialization
        """
        self.seed = seed
        self.actor = ActorNetwork(state_size, action_size, seed)
        self.critic = CriticNetwork(state_size, action_size, seed)

    def copy(self, network):
        """
        Copy the parameters from the provided network into current actor critic networks.

        :param network: Network instance with actor and critic model initialized
        """
        # Actor
        for target_param, local_param in zip(self.actor.parameters(), network.actor.parameters()):
            target_param.data.copy_(local_param.data)
        
        # Critic
        for target_param, local_param in zip(self.critic.parameters(), network.critic.parameters()):
            target_param.data.copy_(local_param.data)
    
    def soft_update(self, network, tau=TAU):
        """
        Soft update the current network from the given network.

        :param network: Network values that will be used to update current network
        :param tau: Floating value to determine how much information goes into current network from provided one.
        """
        # Actor
        for target_param, local_param in zip(self.actor.parameters(), network.actor.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        
        # Critic
        for target_param, local_param in zip(self.critic.parameters(), network.critic.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    