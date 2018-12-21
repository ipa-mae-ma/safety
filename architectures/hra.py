
"""
Created on October 1, 2018

@author: mae-ma
@attention: architectures for the safety DRL package
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 1.2.3

#############################################################################################

History:
- v1.2.3: update layers
- v1.2.2: save loss
- v1.2.1: use maluuba experience replay buffer
- v1.2.0: update optimizer to use all inputs instead of only models[0] input
- v1.1.2: update layers
- v1.1.1: add flatten layer
- v1.1.0: add functions
- v1.0.1: build model
- v1.0.0: first init
"""

import os
import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import random
from copy import deepcopy
import time

from architectures.replay_buffer import ReplayBuffer, ExperienceReplay
import architectures.misc as misc
from architectures.misc import Font
from architectures.agent import Agent


class HybridRewardArchitecture(Agent):
    def __init__(self, env, params):
        self.env = env
        self.input_shape = self.env.state_shape # [110] because 'state_mode=mini'
        self.output_dim = self.env.nb_actions
        self.n_heads = 10

        self.params = params
        self.gamma = self.params['gamma']
        self.l_rate = self.params['learning_rate']
        self.rng = np.random.RandomState(self.params['random_seed'])
        self.epsilon = self.params['epsilon']
        self.epsilon_min = self.params['epsilon_min']
        self.replay_buffer = ReplayBuffer(max_size=self.params['replay_memory_size'])
        self.transitions = ExperienceReplay(max_size=self.params['replay_memory_size'], history_len=1, rng=self.rng,
                                            state_shape=self.input_shape, action_dim=1, # action_dim is 1 because we only have 1 actino
                                            reward_dim=len(self.env.possible_fruits))
        self.minibatch_size = self.params['minibatch_size']
        self.update_counter = 0
        self.update_freq = self.params['update_frequency']
        self.num_epochs = self.params['num_epochs']
        self.num_episodes = self.params['num_episodes']
        self.num_steps = self.params['num_steps']

        self.models = [self._build_network() for _ in range(self.n_heads)]
        self.model_yaml = self.models[0].to_yaml()
        self.save_model_yaml(architecture='HRA')
        print(self.model_yaml)
        self.target_models = [self._build_network() for _ in range(self.n_heads)]
        self.all_model_params = self.flatten([model.trainable_weights for model in self.models])
        self.all_target_model_params = self.flatten([target_model.trainable_weights for target_model in self.target_models])
        self.weight_transfer(from_model=self.models, to_model=self.target_models)
        self._build_optimizer()

    def _build_network(self) -> keras.models.Sequential:
        """
        build network with A3C parameters
        Output:
            model (keras.model): model
        """
        input_shape = (1,) + tuple(self.input_shape)
        layer_input = keras.Input(shape=input_shape, name='input')
        flatten = keras.layers.Flatten()(layer_input)
        # l_dense = keras.layers.Dense(250/self.n_heads, activation='relu', kernel_initializer='he_uniform', name='dense')(flatten)
        l_dense = keras.layers.Dense(25, activation='relu', kernel_initializer='he_uniform', name='dense')(flatten)
        out = keras.layers.Dense(self.output_dim, activation='linear', kernel_initializer='he_uniform', name='out')(l_dense)
        model = keras.Model(inputs=layer_input, outputs=out)
        # model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
        return model


    def _compute_loss(self, q, a, r, t, q_):
        """
        compute loss as mean squared error
        Inputs:
        Outputs:
            loss (float): mean squared error
        """
        # produce mask for actions with outputs [0, 0, 1, 0] if a = 2
        amask = tf.one_hot(a, q.get_shape()[1], 1.0, 0.0)
        # predictions have shape (len(actions), 1)
        predictions = tf.reduce_sum(q * amask, axis=1)
        # TODO: check for K.mean instead of K.max
        targets = r + (1 - t) * self.gamma * K.max(q_, axis=1)
        loss = K.sum((targets - predictions) ** 2)
        return loss

    def _build_optimizer(self):
        s = K.placeholder(shape=tuple([None] + [1] + self.input_shape))
        a = K.placeholder(ndim=1, dtype='int32')
        r = K.placeholder(ndim=2, dtype='float32')
        s_ = K.placeholder(shape=tuple([None] + [1] + self.input_shape))
        t = K.placeholder(ndim=1, dtype='float32')

        updates = []
        losses = 0.0
        qs = []
        qs_ = []
        for i in range(len(self.models)):
            local_s = s
            local_s_ = s_
            # compute Q values for head i
            qs.append(self.models[i](local_s))
            qs_.append(self.models[i](local_s_))
            # compute loss for reward of head i
            loss = self._compute_loss(qs[-1], a, r[:, i], t, qs_[-1])
            # rho and epsilon from Maluuba implementation
            # TODO: use Adam optimizer instead of RMSprop
            optimizer = keras.optimizers.RMSprop(lr=self.l_rate, rho=0.95, epsilon=1e-7)
            updates += optimizer.get_updates(params=self.models[i].trainable_weights, loss=loss)
            losses += loss

        target_updates = []
        # update target model weights to model weights
        for model, target_model in zip(self.models, self.target_models):
            for model_weight, target_model_weight in zip(model.trainable_weights, target_model.trainable_weights):
                target_updates.append(K.update(target_model_weight, model_weight))

        # train networks on batch
        self._train_on_batch = K.function(inputs=[s, a, r, s_, t], outputs=[losses], updates=updates)
        # returns all Q-values for all states -> qs
        self.predict_qs = K.function(inputs=[s], outputs=qs)
        # update target network with weights from trained network
        self.update_weights = K.function(inputs=[], outputs=[], updates=target_updates)


    def flatten(self, l):
        """
        Inputs:
            l (list): list with lists
        Ouput:
            f_l (list): flat list with all items of all lists in l
        """
        return list(np.array(l).flatten())


    def get_max_action(self, state):
        """
        aggregate the outputs of the heads
        Input:
            state (np.array): state
        Output:
            action (int): action choosen over all heads
            q (np.array): q-values
        """
        state = self._reshape(state)
        q = np.array(self.predict_qs([state]))
        # sum over corresponding action for all models -> vertical
        q = np.sum(q, axis=0)
        # get argmax from array -> horizontal
        return np.argmax(q, axis=1)
        # return np.argmax(q)

    def act(self, state):
        # if self.rng.binomial(1, self.epsilon):
        if self.rng.rand() < self.epsilon:
            return self.rng.randint(self.output_dim)
        else:
            return self.get_max_action(state=state)

    def train_on_batch(self, s, a, r, s_, t):
        """
        Train neural net on batch
        Inputs:
            s (np.array): state
            a (np.array): action
            r (np.array): reward
            s_ (np.array): next state
            t (np.array): terminal
        Outputs:
            losses (float): loss of mean squared error
        """
        # s = self._reshape(s)
        # s_ = self._reshape(s_)
        if len(r.shape) == 1:
            r = np.expand_dims(r, axis=-1)
        # inputs = [s, a, r, s_, t]
        return self._train_on_batch([s, a, r, s_, t])

    def learn(self):
        """
        Ouputs:
            losses (float): loss output of mean squared error
        """
        # if self.replay_buffer.__len__() < self.minibatch_size:
        #     return
        if self.transitions.size < self.minibatch_size:
            return
        # s, a, r, s_, t = self.replay_buffer.sample(self.minibatch_size)
        s, a, r, s_, t = self.transitions.sample(self.minibatch_size)
        losses = self.train_on_batch(s, a, r, s_, t)
        if self.update_counter == self.update_freq:
            self.update_weights([])
            self.update_counter = 0
        else:
            self.update_counter += 1
        return losses

    def training_print(self, step_counter: int, timing_list: list) -> None:
        """
        print terminal output for training
        """
        if len(timing_list) > 30:
            mean_v = np.mean(np.array(timing_list[-30:-1]) - np.array(timing_list[-31:-2]))
        else:
            mean_v = np.inf
        text = '\r' + misc.Font.yellow + \
            '>>> Training: {}/{} --- {: .1f} steps/second' + misc.Font.end
        print(text.format(step_counter, self.num_steps, 1/mean_v), end='', flush=True)


    def do_episode(self):
        reward = []
        step_counter = 0
        loss_array = np.zeros((0, 3))
        for epoch in range(self.num_epochs):
            for episode in range(self.num_episodes):
                # states = []
                state, _, _, _ = self.env.reset()
                rew = 0
                timing = [0.0]
                for step in range(self.num_steps):
                    timing.append(time.time())
                    self.training_print(step_counter=step+1, timing_list=timing)

                    action = self.act(state)
                    if False:
                        print()
                        print(Font.green + 'action' + Font.end)
                        print(action)
                    self.calc_eps_decay(step_counter=step_counter)
                    next_state, r, terminated, info = self.env.step(action)
                    reward_channels = info['head_reward']
                    state_low = next_state[2, ...]
                    # self.replay_buffer.add(obs_t=state, act=action, rew=reward_channels, 
                    #                        obs_tp1=next_state, done=terminated)
                    self.transitions.add(s=state, a=action, r=reward_channels, t=terminated)
                    # learn every 4 steps
                    if step % 4 == 0:
                        loss = self.learn()
                        if loss is None:
                            loss = 0.0
                        else:
                            loss = loss[0]
                    loss_array = np.append(loss_array, np.array([[0, 0, loss]]), axis=0)
                    if step % 100 == 0:
                        np.savetxt(os.path.join('output', 'training_log_HRA.csv'), loss_array, fmt='%.4f', delimiter=',')

                    state = next_state
                    self.env.render()
                    step_counter += 1

                    if step == self.num_steps - 1:
                        terminated = True
                    rew += r
                    if terminated is True:
                        print('\nepisode: {}/{} \nepoch: {}/{} \nscore: {} \neps: {:.3f} \nsum of steps: {}'.
                              format(episode, self.num_episodes, epoch,
                                     self.num_epochs, rew, self.epsilon, step_counter))
                        reward.append((rew, step_counter, step))
                        with open(os.path.join('output', 'reward.yml'), 'w') as f:
                            yaml.dump(reward, f)
                        break


    def calc_eps_decay(self, step_counter: int) -> None:
        """
        calculates new epsilon for the given step counter according to the annealing formula
        y = -mx + c
        eps = -(eps - eps_min) * counter + eps
        Input:
            counter (int): step counter
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon = -((self.params['epsilon'] - self.epsilon_min) /
                                self.params['eps_max_frame']) * step_counter + self.params['epsilon']


    @staticmethod
    def _reshape(states: np.array) -> np.array:
        if len(states.shape) == 2:
            states = np.expand_dims(states, axis=0)
        if len(states.shape) == 3:
            states = np.expand_dims(states, axis=1)
        return states

    @staticmethod
    def weight_transfer(from_model: keras.Model, to_model: keras.Model) -> None:
        """
        transfer network weights
        """
        for f_model, t_model in zip(from_model, to_model):
            t_model.set_weights(deepcopy(f_model.get_weights()))


    def main(self):
        print('HRA here')
        print('–' * 30)


if __name__ == '__main__':
    print('HRA __main__')
