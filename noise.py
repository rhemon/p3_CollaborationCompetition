import numpy as np
import random
import copy

"""
Code taken from Udacty's classroom
"""
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(1)
        self.state = x + dx
        return self.state
    
    def get_action(self, action): 
        """
        For given action add the noise to the action and return it clipped to its range.
        """
        return np.clip(action + self.sample(), -1, 1)