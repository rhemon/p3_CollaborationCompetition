import torch
import numpy as np
import random
from collections import deque, namedtuple

BUFFER_SIZE = int(1e6)          # memory length; number of past time steps stored
BATCH_SIZE = 256                # number of experiences sampled for learning

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, seed=0):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["states0", "actions0", "rewards0", "next_states0", "dones0", 
                   "states1", "actions1", "rewards1", "next_states1", "dones1"])
        self.seed = random.seed(seed)
    
    def add(self, states0, actions0, rewards0, next_states0, dones0, states1, actions1, rewards1, next_states1, dones1):
        """Add a new experience to memory."""
        e = self.experience(states0, actions0, rewards0, next_states0, dones0, states1, actions1, rewards1, next_states1, dones1)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states0 = torch.from_numpy(np.vstack([e.states0 for e in experiences if e is not None])).float()
        actions0 = torch.from_numpy(np.vstack([e.actions0 for e in experiences if e is not None])).float()
        rewards0 = torch.from_numpy(np.vstack([e.rewards0 for e in experiences if e is not None])).float()
        next_states0 = torch.from_numpy(np.vstack([e.next_states0 for e in experiences if e is not None])).float()
        dones0 = torch.from_numpy(np.vstack([e.dones0 for e in experiences if e is not None]).astype(np.uint8)).float()
        states1 = torch.from_numpy(np.vstack([e.states1 for e in experiences if e is not None])).float()
        actions1 = torch.from_numpy(np.vstack([e.actions1 for e in experiences if e is not None])).float()
        rewards1 = torch.from_numpy(np.vstack([e.rewards1 for e in experiences if e is not None])).float()
        next_states1 = torch.from_numpy(np.vstack([e.next_states1 for e in experiences if e is not None])).float()
        dones1 = torch.from_numpy(np.vstack([e.dones1 for e in experiences if e is not None]).astype(np.uint8)).float()
  
        return (states0, actions0, rewards0, next_states0, dones0, states1, actions1, rewards1, next_states1, dones1)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
