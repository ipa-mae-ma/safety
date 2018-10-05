
"""
Created on October 1, 2018

@author: mae-ma
@attention: architectures for the safety DRL package
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 1.0.0

#############################################################################################

History:
- v1.0.0: first init
"""

from collections import deque

import time
import numpy as np
import pickle
import click
import gym

import logger
from architectures.replay_buffer import ReplayBuffer
from architectures.wrappers import NoopResetEnv, EpisodicLifeEnv

nprs = np.random.RandomState
rng = nprs(42)

#############################
# Optimizer
#############################
class Adam(object):
    def __init__(self, shape, stepsize, beta1=0.9, beta2=0.999, epsilon=1e-08):
        self.stepsize, self.beta1, self.beta2, self.epsilon = stepsize, beta1, beta2, epsilon
        self.t = 0
        self.v = np.zeros(shape, dtype=np.float32)
        self.m = np.zeros(shape, dtype=np.float32)

    def step(self, g):
        self.t += 1
        a = self.stepsize * \
            np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.v = self.beta2 * self.v + (1 - self.beta2) * (g * g)
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        step = - a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step
#############################


class DeepQNetwork:
    def __init__(self):
        pass

    def main(self):
        print('DQN here')
        print('–' * 30)


if __name__ == '__main__':
    print('DQN __main__')
