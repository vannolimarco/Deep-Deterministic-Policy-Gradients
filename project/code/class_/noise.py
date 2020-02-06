import numpy as np
import gym
from collections import deque
import random
from math import sqrt

'''
The noise class where there are all class with chich is possible add noise to the action In this case is used only
Ornstein Uhlenbeck Noise.
'''

class OrnsteinUhlenbeckNoise(object):
    '''
    The Ornstein Uhlenbeck Noise with which add noise to the action improving the behavior of agent and its exploration in the space.
    '''
    def __init__(self, action_space, mu=0.1, theta=0.25, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    # reset of state
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    # get a new action to which the temporally correlated noise is added.
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)
