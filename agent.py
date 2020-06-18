from collections import deque
import numpy as np
import random
import torch
from torch import optim

from model import *
from buffer import *
from noise import *

ACTOR_LR = 5e-4         # Actor models learning rate
CRITIC_LR = 5e-4        # Critic models learning rate
WEIGHT_DECAY = 0        # Weight decay for critic model
GAMMA = 0.99            # Discount rate
UPDATE_EVERY = 5       # Learn after how many steps
TIMES_UPDATE = 10       # How many times to learn at each time to learn


class Agent:

    def __init__(self, state_size, action_size, random_seed, action_low=-1, action_high=1):
        
        self.seed = random.seed(random_seed)
        self.state_size = state_size
        self.action_size = action_size
        self.a_low = action_low
        self.a_high = action_high
        self.network = Network(state_size, action_size, random_seed)
        
        self.actor_opt = optim.Adam(self.network.actor.parameters(), lr=ACTOR_LR)
        self.critic_opt = optim.Adam(self.network.critic.parameters(), lr=CRITIC_LR, weight_decay=WEIGHT_DECAY)

        self.target_network = Network(state_size, action_size, random_seed)
        self.ounoise = OUNoise(action_size, action_low, action_high)
        self.memory = ReplayBuffer()
        self.t_step = 0
    
    def act(self, state, add_noise=True):
        state = torch.tensor(state).float()
        self.network.actor.eval()
        with torch.no_grad():
            action = self.network.actor(state)
            action = action.data.cpu().numpy()
        self.network.actor.train()
        if add_noise:
            return self.ounoise.get_action(action)
        return action
    
    def step(self, state, action, reward, next_state, done, agent_num):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY

        if len(self.memory) > BATCH_SIZE and self.t_step == 0:
            for i in range(TIMES_UPDATE):
                experiences = self.memory.sample()
                self.learn(experiences, agent_num=agent_num)
                self.target_network.soft_update(self.network)
    
    def learn(self, experiences, gamma=GAMMA, agent_num=0):
        
        states, actions, rewards, next_states, dones = experiences
#         print(states.size(), next_states.size())
        if agent_num == 0:
            curr_agent_state = states[:, :self.state_size]
            curr_agent_next = next_states[:, :self.state_size]
            other_agent_state = states[:, self.state_size:]
            other_agent_next = next_states[:, self.state_size:]
        else:
            curr_agent_state = states[:, self.state_size:]
            other_agent_state = states[:, :self.state_size]
            curr_agent_next = next_states[:, self.state_size:]
            other_agent_next = next_states[:, :self.state_size]
            
#         print(curr_agent_state.size(), curr_agent_next.size())
#         print(other_agent_state.size(), other_agent_next.size())
        
        
        cur_next_actions = self.target_network.actor(curr_agent_next)
        other_next_actions = self.target_network.actor(other_agent_next)
        # other_next_actions = torch.cat([actions[1:, self.action_size:], actions[:1, self.action_size:]], dim=0)
        critic_next_observations = torch.cat([other_agent_next, cur_next_actions, other_next_actions], dim=1)
        
        Q_target_next = self.target_network.critic(curr_agent_next, critic_next_observations)
        Q_target = rewards + (gamma * Q_target_next * (1-dones))
        
        critic_observations = torch.cat([other_agent_state, actions], dim=1)
        Q_predicted = self.network.critic(curr_agent_state, critic_observations)
        
        critic_loss = F.mse_loss(Q_predicted, Q_target)
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.critic.parameters(), 1)
        self.critic_opt.step()
        
        critic_observations = torch.cat([other_agent_state, actions], dim=1)
        actor_loss = -self.network.critic(curr_agent_state, critic_observations).mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()