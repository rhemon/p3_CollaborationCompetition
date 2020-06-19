from collections import deque
import numpy as np
import random
import torch
from torch import optim

from model import *
from buffer import *
from noise import *

ACTOR_LR = 1e-4         # Actor models learning rate
CRITIC_LR = 1e-3        # Critic models learning rate
WEIGHT_DECAY = 0        # Weight decay for critic model
GAMMA = 0.99            # Discount rate
UPDATE_EVERY = 1        # Learn after how many steps
TIMES_UPDATE = 1        # How many times to learn at each time to learn


class Agent:

    memory = None # Shared memory for both agents
    
    def __init__(self, state_size, action_size, random_seed, action_low=-1, action_high=1):
        """
        Initializes the agent to play in the environment.
        
        :param state_size: Number of information provided in the state
        :param action_size: Number of actions environment can take
        :param random_seed: Seed for random initialization
        :param action_low: Minimum value for action
        :param action_high: Maxmimum value for aciton
        """

        self.seed = random.seed(random_seed)
        self.state_size = state_size
        self.action_size = action_size
        self.a_low = action_low
        self.a_high = action_high
        self.network = Network(state_size, action_size, random_seed)
        
        self.actor_opt = optim.Adam(self.network.actor.parameters(), lr=ACTOR_LR)
        self.critic_opt = optim.Adam(self.network.critic.parameters(), lr=CRITIC_LR, weight_decay=WEIGHT_DECAY)

        self.target_network = Network(state_size, action_size, random_seed)
        self.ounoise = OUNoise(action_size, random_seed)
        
        if Agent.memory == None:
            Agent.memory = ReplayBuffer()
        
        self.t_step = 0
    
    def act(self, state, add_noise=True):
        """
        Returns action for given state.

        :param state: State of the environment,for which to determine an action
        :param add_noise: Used to determine whether to add nose based on Ornstein Uhlenbeck process
        """
        self.network.actor.eval()
        with torch.no_grad():
            action = self.network.actor(state)
            action = action.data.cpu().numpy()
        self.network.actor.train()
        if add_noise:
            return self.ounoise.get_action(action)
        return action
        
    def add_memory(states0, actions0, rewards0, next_states0, dones0, 
                   states1, actions1, rewards1, next_states1, dones1):
        """
        Add experience of both agent to memory.

        :param states0: State of first agent
        :param actions0: Action of first agent
        :param rewards0: Reward of first agent
        :param next_states0: Next state of first agent
        :param dones0: Terminal state for first agent or not
        :param states1: State of first agent
        :param actions1: Action of second agent
        :param rewards1: Reward of second agent
        :param next_states1: Next state of second agent
        :param dones1: Terminal state for second agent or not
        """
        Agent.memory.add(states0, actions0, rewards0, next_states0, dones0, 
                   states1, actions1, rewards1, next_states1, dones1)
    
    def step(self, agent_num):
        """
        Make a step, and if its time to update, learn.

        :param agent_num: Agent making a step
        """
        self.t_step = (self.t_step + 1) % UPDATE_EVERY

        if len(self.memory) > BATCH_SIZE and self.t_step == 0:
            for i in range(TIMES_UPDATE):
                experiences = self.memory.sample()
                self.learn(experiences, agent_num)
                self.target_network.soft_update(self.network)
    
    def learn(self, experiences, agent_num, gamma=GAMMA):
        """
        Learning algorithm for the model. Uses the target critic network to determine
        the MSE loss for predicted Q values with local network for the experieces sampled 
        from the memory. In the backpropagation of critic network, clips the gradient values to 1.
        Then updates the actor network with the goal to maximize the average value determined by the critic model.
        So the loss is -Q_local(state, action).mean(). 

        :param experiences: State, Actions, Rewards, Next states, dones randomly sampled from the memory
        :param agent_num: Agent which is learning now
        :param gamma: Discount rate that determines how much of future reward impacts total reward.
        """

        states0, actions0, rewards0, next_states0, dones0, states1, actions1, rewards1, next_states1, dones1 = experiences
        if agent_num == 0:
            states = states0
            actions = actions0
            rewards = rewards0
            next_states = next_states0
            dones = dones0
        else:
            states = states1
            actions = actions1
            rewards = rewards1
            next_states = next_states1
            dones = dones1
        
        next_actions = self.target_network.actor(next_states)
        
#         print(next_states.size(), next_actions.size())
        Q_target_next = self.target_network.critic(next_states, next_actions)
        Q_target = rewards + (gamma * Q_target_next * (1-dones))
        Q_predicted = self.network.critic(states, actions)
        # print(Q_predicted.size())
        critic_loss = F.mse_loss(Q_predicted, Q_target)
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.critic.parameters(), 1)
        self.critic_opt.step()
        
        actions = self.network.actor(states)
        actor_loss = -self.network.critic(states, actions).mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()