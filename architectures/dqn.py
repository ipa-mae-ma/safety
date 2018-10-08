
"""
Created on October 1, 2018

@author: mae-ma
@attention: architectures for the safety DRL package
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 1.1.0

#############################################################################################

History:
- v1.1.0: add functions
- v1.0.0: first init
"""

from collections import deque

import numpy as np
# import time
# import click
# import gym
# import logger
import tensoflow
from tensorflow import keras

from architectures.replay_buffer import ReplayBuffer
import architectures.misc as misc
from architectures.mdp import MDP
# from architectures.wrappers import NoopResetEnv, EpisodicLifeEnv



class DeepQNetwork:
    def __init__(self,
                 mdp: MDP,
                 input_size: tuple=(84, 84, 4),
                 output_size: int=4,
                 name: str='DQN') -> None:
        """
        DQN Agent can:
        1. build network
        2. predict Q_value for given state
        3. train parameters

        Input:
            input_size (int): Input dimension
            mdp (MDP): markov decision process class
            output_size (int): Output dimension
            name (str, optional): TF Graph will be built under this name
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        cfg_file = os.path.join(dir_path, 'config_dqn.yaml')
        self.params = yaml.safe_load(open(cfg_file, 'r'))

        nprs = np.random.RandomState
        self.rng = nprs(self.params['random_seed'])

        self.mdp = mdp
        self.input_size = input_size
        self.output_size = output_size
        self.l_rate = self.params['learning_rate']
        self.minibatch_size = self.params['minibatch_size']
        self.gamma = self.params['gamma']
        self.epsilon = self.params['epsilon']
        self.epsilon_min = self.params['epsilon_min']
        self.epsilon_decay = self.params['epsilon_decay']
        self.replay_buffer = ReplayBuffer(self.params['replay_memory_size'])

        self.net_name = name

        self.model = self._build_network()

    def _build_network(self, l_rate=self.l_rate) -> keras.Sequential:
        """
        build network with DQN parameters
        """
        model = keras.Sequential()
        # first hidden layer
        model.add(keras.layers.Conv2D(input_shape=self.input_size, filters=32,
                                      kernel_size=(8, 8), strides=4, activation='relu')
        # second hidden layer
        model.add(keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=2, activation='relu')
        # third hidden layer
        model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation='relu')
        # fourth hidden layer
        model.add(keras.layers.Dense(512, activation='relu')
        # output layer
        model.add(keras.layers.Dense(self.output_size, activation='relu')
        # compile model
        model.compile(optimizer=tf.train.RMSPropOptimizer(learning_rate=self.l_rate,
                                                            decay=0.9,
                                                            momentum=self.params['gradient_momentum'],
                                                            epsilon=self.params['min_squared_gradient']),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        return model


    def act(self, state: int) -> action:
        """
        return action from neural net
        Input:
            state (int): the current state
        """
        q_vals=self.model.predict(state)
        action=misc.eps_greedy(q_vals=q_vals[0], eps=self.epsilon)
        # return action
        return action


    def remember(self, state: int, action: int, reward: int, next_state: int, done: int) -> None:
        """
        Add values to the replay buffer
        Input:
            state (int): scalar value for current state
            action (int): scalar value for chosen action
            reward (int): scalar value for received reward
            next_state (int): scalar value of next state
            done (int): scalar value if episode is finished
        """
        self.replay_buffer.add(obs_t=state, act=action, rew=reward, obs_tp1=next_state, done=done)

    def replay(self, batch_size: int=self.minibatch_size) -> None:
        """
        get data from replay buffer and train session
        Input:
            batch_size (int): number of minibatches to get from replay buffer
        """
        minibatch=self.replay_buffer.sample(batch_size=batch_size)
        # all of type np.array
        for state, action, reward, next_state, done in minibatch:
            target=reward
            if not done:
                target=reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f=self.model.predict(state)
            target_f[0][action]=target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        # decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= - self.epsilon_decay


    def main(self):
        print('DQN here')
        print('–' * 30)


if __name__ == '__main__':
    print('DQN __main__')
