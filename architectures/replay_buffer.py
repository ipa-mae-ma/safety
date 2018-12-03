"""
Created on October 5, 2018

@author: mae-ma
@attention: replay buffer for DQN
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 1.1.0

#############################################################################################

History:
- v1.1.0: add maluuba experience replay
- v1.0.0: first init
"""


import numpy as np
import random
import pickle
import logging


class ReplayBuffer(object):
    def __init__(self, max_size):
        """
        Simple replay buffer for storing sampled DQN (s, a, s', r) transitions as tuples.

        :param size: Maximum size of the replay buffer.
        """
        self._buffer = []
        self._max_size = int(max_size)
        # counting variable
        self._idx = 0

    def __len__(self):
        return len(self._buffer)

    def add(self, obs_t, act, rew, obs_tp1, done):
        """
        Add a new sample to the replay buffer.
        :param obs_t: observation at time t
        :param act:  action
        :param rew: reward
        :param obs_tp1: observation at time t+1
        :param done: termination signal (whether episode has finished or not)
        """
        data = (obs_t, act, rew, obs_tp1, done)
        # if replay buffer not filled then fill it
        if self._idx >= len(self._buffer):
            self._buffer.append(data)
        # else overrite data
        else:
            self._buffer[self._idx] = data
        self._idx = (self._idx + 1) % self._max_size

    def _encode_sample(self, idxes):
        """
        encode samples as numpy array of observations,
        actions, rewards, next observations and terminal signals
        """
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._buffer[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        """
        Sample a batch of transition tuples.

        :param batch_size: Number of sampled transition tuples.
        :return: Tuple of transitions.
        """
        idxes = [random.randint(0, len(self._buffer) - 1)
                 for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def dump(self, file_path=None):
        """
        Dump the replay buffer into a file.
        """
        file = open(file_path, 'wb')
        pickle.dump(self._buffer, file, -1)
        file.close()

    def load(self, file_path=None):
        """
        Load the replay buffer from a file
        """
        file = open(file_path, 'rb')
        self._buffer = pickle.load(file)
        file.close()


class ExperienceReplay(object):
    """
    Efficient experience replay pool for DQN.
    """

    def __init__(self, max_size=100, history_len=1, state_shape=None, action_dim=1, reward_dim=1, state_dtype=np.uint8,
                 rng=None):
        if rng is None:
            self.rng = np.random.RandomState(1234)
        else:
            self.rng = rng
        self.size = 0
        self.head = 0
        self.tail = 0
        self.max_size = max_size
        self.history_len = history_len
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.state_dtype = state_dtype
        self._minibatch_size = None
        self.states = np.zeros(
            [self.max_size] + list(self.state_shape), dtype=self.state_dtype)
        self.terms = np.zeros(self.max_size, dtype='bool')
        if self.action_dim == 1:
            self.actions = np.zeros(self.max_size, dtype='int32')
        else:
            self.actions = np.zeros(
                (self.max_size, self.action_dim), dtype='int32')
        if self.reward_dim == 1:
            self.rewards = np.zeros(self.max_size, dtype='float32')
        else:
            self.rewards = np.zeros(
                (self.max_size, self.reward_dim), dtype='float32')

    def _init_batch(self, number):
        self.s = np.zeros([number] + [self.history_len] +
                          list(self.state_shape), dtype=self.states[0].dtype)
        self.s2 = np.zeros([number] + [self.history_len] +
                           list(self.state_shape), dtype=self.states[0].dtype)
        self.t = np.zeros(number, dtype='bool')
        action_indicator = self.actions[0]
        if self.actions.ndim == 1:
            self.a = np.zeros(number, dtype='int32')
        else:
            self.a = np.zeros((number, action_indicator.size),
                              dtype=action_indicator.dtype)
        if self.rewards.ndim == 1:
            self.r = np.zeros(number, dtype='float32')
        else:
            self.r = np.zeros((number, self.reward_dim), dtype='float32')

    def sample(self, num=1):
        if self.size == 0:
            logging.error('cannot sample from empty transition table')
        elif num <= self.size:
            if not self._minibatch_size or num != self._minibatch_size:
                self._init_batch(number=num)
                self._minibatch_size = num
            for i in range(num):
                self.s[i], self.a[i], self.r[i], self.s2[i], self.t[i] = self._get_transition()
            return self.s, self.a, self.r, self.s2, self.t
        elif num > self.size:
            logging.error(
                'transition table has only {0} elements; {1} requested'.format(self.size, num))

    def _get_transition(self):
        sample_success = False
        while not sample_success:
            randint = self.rng.randint(
                self.head, self.head + self.size - self.history_len)
            state_indices = np.arange(randint, randint + self.history_len)
            next_state_indices = state_indices + 1
            transition_index = randint + self.history_len - 1
            a_axis = None if self.action_dim == 1 else 0
            r_axis = None if self.reward_dim == 1 else 0
            if not np.any(self.terms.take(state_indices[:-1], mode='wrap')):
                s = self.states.take(state_indices, mode='wrap', axis=0)
                a = self.actions.take(
                    transition_index, mode='wrap', axis=a_axis)
                r = self.rewards.take(
                    transition_index, mode='wrap', axis=r_axis)
                t = self.terms.take(transition_index, mode='wrap')
                s2 = self.states.take(next_state_indices, mode='wrap', axis=0)
                sample_success = True
        return s, a, r, s2, t

    def add(self, s, a, r, t):
        self.states[self.tail] = s
        self.actions[self.tail] = a
        self.rewards[self.tail] = r
        self.terms[self.tail] = t
        self.tail = (self.tail + 1) % self.max_size
        if self.size == self.max_size:
            self.head = (self.head + 1) % self.max_size
        else:
            self.size += 1

    def reset(self):
        self.size = 0
        self.head = 0
        self.tail = 0
        self._minibatch_size = None
        self.states = np.zeros(
            [self.max_size] + list(self.state_shape), dtype=self.state_dtype)
        self.terms = np.zeros(self.max_size, dtype='bool')
        if isinstance(self.action_dim, int):
            self.actions = np.zeros(self.max_size, dtype='int32')
        else:
            self.actions = np.zeros(
                (self.max_size, self.action_dim.size), dtype=self.action_dim.dtype)
        if isinstance(self.reward_dim, int):
            self.rewards = np.zeros(self.max_size, dtype='float32')
        else:
            self.rewards = np.zeros((self.max_size, 2), dtype='float32')
