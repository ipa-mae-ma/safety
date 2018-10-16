
"""
Created on October 1, 2018

@author: mae-ma
@attention: architectures for the safety DRL package
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 1.2.4

#############################################################################################

History:
- v1.2.4: use target model
- v1.2.3: add some documentation and new function to update target network
- v1.2.2: update 'act' func to work with eps_greedy
- v1.2.1: rename 'replay' to '_replay' and tweak function to input
- v1.2.0: add 'flatten' layer to neural net
- v1.1.2: rename functions
- v1.1.1: add logger and target_model
- v1.1.0: add functions
- v1.0.0: first init
"""

from collections import deque

import numpy as np
import os
import yaml
# import time
# import click
# import gym
# import logger
import tensorflow as tf
from tensorflow import keras
import tqdm

from architectures.replay_buffer import ReplayBuffer
import architectures.misc as misc
# from architectures.mdp import MDP
from architectures.misc import Font
# from architectures.wrappers import NoopResetEnv, EpisodicLifeEnv

# tf.enable_eager_execution()

# TODO: add update from target network instead of only one network
# TODO: compare pyplot output of network after training with pyplot
# TODO: implement warmstart

class DeepQNetwork:
    def __init__(self,
                 env,
                 input_size: tuple=(1, 14, 21, 1),
                 output_size: int=4,
                 name: str='DQN') -> None:
        """
        DQN Agent can:
        1. build network
        2. predict Q_value for given state
        3. train parameters

        Input:
            input_size (int): Input dimension of format type (n_img, img_height, img_width, n_channels)
            env (env): environment class
            output_size (int): Output dimension
            name (str, optional): TF Graph will be built under this name
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        cfg_file = os.path.join(dir_path, 'config_dqn.yaml')
        self.params = yaml.safe_load(open(cfg_file, 'r'))

        nprs = np.random.RandomState
        self.rng = nprs(self.params['random_seed'])

        # self.mdp = mdp
        self.input_size = input_size
        self.output_size = output_size
        self.l_rate = self.params['learning_rate']
        self.minibatch_size = self.params['minibatch_size']
        self.gamma = self.params['gamma']
        self.epsilon = self.params['epsilon']
        self.epsilon_min = self.params['epsilon_min']
        self.epsilon_decay = self.params['epsilon_decay']
        self.replay_buffer = ReplayBuffer(float(self.params['replay_memory_size']))
        self.csv_logger = keras.callbacks.CSVLogger('training_log_DQN.csv', append=True)
        self.episode_max_len = self.params['episode_max_len']
        self.total_eps = self.params['total_eps']

        self.fps = 0
        self.episode_num = 0
        self.last_episode_steps = 0
        self.total_training_steps = 0
        self.score_agent = 0
        self.eval_steps = []
        self.eval_scores = []
        self.env = env

        self.name = name
        self.model, self.target_model = self._build_network()

    def _build_network(self) -> (keras.models.Sequential, keras.models.Sequential):
        """
        build network with DQN parameters
        """
        model = keras.Sequential()
        # first hidden layer
        model.add(keras.layers.Conv2D(input_shape=(88, 88, 1), batch_size=1, filters=32,
                                      kernel_size=(8, 8), strides=4, activation='relu', data_format="channels_last"))
        # second hidden layer
        model.add(keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=2, activation='relu'))
        # third hidden layer
        model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation='relu'))
        # flatten conv output so last output is of shape (batchsize, output_size)
        model.add(keras.layers.Flatten())
        # fourth hidden layer
        model.add(keras.layers.Dense(512, activation='relu'))
        # output layer
        model.add(keras.layers.Dense(self.output_size, activation='relu'))
        # compile model
        model.compile(optimizer=tf.train.RMSPropOptimizer(learning_rate=self.l_rate,
                                                          decay=0.9,
                                                          momentum=self.params['gradient_momentum'],
                                                          epsilon=self.params['min_squared_gradient']),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        target_model = keras.models.clone_model(model)
        return model, target_model

    def do_training(self, total_eps=10, eps_per_epoch=100, eps_per_test=100, is_learning=True, is_testing=True):
        """
        train DQN algorithm with replay buffer and minibatch
        """
        pbar = tqdm.tqdm(total=total_eps, unit='Episodes', ncols=100)
        while self.episode_num < total_eps:
            # print(Font.yellow + Font.bold + 'Training ... ' + str(self.episode_num) + '/' + str(total_eps) + Font.end,
            #    end='\n')
            if is_learning:
                pbar.update(1)
                self._replay(batch_size=self.minibatch_size)
                self.episode_num += 1

            if is_testing:
                # TODO: implement testing output
                pass
        pbar.close()
        self.episode_num = 0


    def update_target_model(self) -> None:
        """
        update the target model with the weights from the trained model
        """
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state: int) -> float:
        """
        return action from neural net
        Input:
            state (np.array): the current state as shape (1, img_height, img_width, 1)
        Output:
            action (float): probability to choose action
        """
        state = state[np.newaxis, ...]
        q_vals = self.model.predict(state)
        action = misc.eps_greedy(q_vals=q_vals[0], eps=self.epsilon, rng=self.rng)
        # decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        # return action
        return action

    def remember(self, state, action: int, reward: int, next_state: int, done: int) -> None:
        """
        Add values to the replay buffer
        Input:
            state (np.array): numpy array of current state
            action (int): scalar value for chosen action
            reward (int): scalar value for received reward
            next_state (np.array): numpy array of next state
            done (int): scalar value if episode is finished
        """
        self.replay_buffer.add(obs_t=state, act=action, rew=reward, obs_tp1=next_state, done=done)

    def save_buffer(self, path: str) -> None:
        """
        save the replay buffer
        Input:
            path (str): where to save the buffer
        """
        self.replay_buffer.dump(file_path=path)

    def save_weights(self, path: str) -> None:
        self.model.save_weights(filepath=path)
        self.target_model.save_weights(filepath='target_' + path)

    def _replay(self, batch_size: int=32) -> None:
        """
        get data from replay buffer and train session
        Input:
            batch_size (int): size of batch to sample from replay buffer
        """
        # get 5 arrays in minibatch for state, action, reward, next_state, done
        minibatch = self.replay_buffer.sample(batch_size=batch_size)
        # all of type np.array -> suffix "_a"
        state_a, action_a, reward_a, next_state_a, done_a = minibatch
        ACTION, REWARD, DONE = 0, 1, 2
        # make one array of size (5, batch_size)
        batch = np.concatenate((action_a, reward_a, done_a), axis=0).reshape(len(minibatch) - 2, -1)
        # transpose to array of size (batch_size, 5)
        batch = batch.transpose()
        # for state, action, reward, next_state, done in minibatch:
        i = 0
        for sample in batch:
            state = state_a[i, ...] # x[:,np.newaxis]
            # add new axis to increase the fit the expeted size for the model
            state = state[np.newaxis, ...]
            next_state = next_state_a[i, ...]
            # add new axis to increase the fit the expeted size for the model
            next_state = next_state[np.newaxis, ...]
            action = sample[ACTION]
            reward = sample[REWARD]
            done = sample[DONE]
            # set the target for DQN
            y_target = reward
            if not done:
                # self.model.predict(s') -> Q(s',a')
                y_target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            # create predictet target function
            y = self.model.predict(state)
            # self.model.predict(s) -> Q(s,a)
            # y.shape -> (1, 4)
            y[0][int(action)] = y_target
            # fit method feeds input and output pairs to the model
            # then the model will train on those data to approximate the output
            # based on the input
            # [src](https://keon.io/deep-q-learning/)
            input = state
            output = y
            self.model.fit(input, output, epochs=1, verbose=0, callbacks=[self.csv_logger])
            # self.model.fit(state, target_f, epochs=1, verbose=0)
            # TODO: return mean score and mean steps
            i += 1

    def main(self):
        print('DQN here')
        print('–' * 30)


if __name__ == '__main__':
    print('DQN __main__')
