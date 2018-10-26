
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

import numpy as np
import os
import yaml
import tensorflow as tf
from tensorflow import keras

from architectures.replay_buffer import ReplayBuffer
import architectures.misc as misc
from architectures.misc import Font
from architectures.agent import Agent


class AsynchronousAdvantageActorCritic(Agent):
    def __init__(self,
                 input_shape: tuple = (1, 14, 21, 1),
                 output_dim: int = 4,
                 warmstart: bool = False,
                 warmstart_path: str = None,
                 simple_a3c: bool = True,
                 params: dict = None) -> None:
        """
        Input:
            input_shape (int): Input shape of format type (n_img, img_height, img_width, n_channels)
            output_dim (int): Output dimension
            warmstart (bool): load network weights from disk
            warmstart_path (str): path where the weights are stored
            simple_a3c (bool): use simplified network
            params (dict): parameter dictionary with all config values
        """
        super(AsynchronousAdvantageActorCritic,
              self).__init__(parameters=params)
        self.params = params
        self.print_params(architecture_name='A3C')

        self.rng = np.random.RandomState(self.params['random_seed'])

        self.simple_a3c = simple_a3c

        if self.simple_a3c:
            # input shape = (img_height * img_width, )
            self.input_shape = input_shape
        else:
            # input shape = (img_height, img_width)
            self.input_shape = input_shape + \
                (1,)  # (height, width, channels=1)
        self.output_dim = output_dim  # number of actions
        self.l_rate = self.params['learning_rate']
        self.minibatch_size = self.params['minibatch_size']
        self.gamma = self.params['gamma']
        self.epsilon = self.params['epsilon']
        self.epsilon_min = self.params['epsilon_min']

        self.delete_training_log(architecture='A3C')

        self.csv_logger = keras.callbacks.CSVLogger(
            'training_log_A3C.csv', append=True)
        # debug flag
        self.debug = False

        self.episode_num = 0
        # build neural nets
        self.model = self._build_network()

        self.loss = [0.0 for _ in self.model.metrics_names]

        self.save_model_yaml(architecture='A3C')

        # do warmstart
        self.warmstart_flag = warmstart
        if self.warmstart_flag:
            self.warmstart(warmstart_path)

    def _build_network(self) -> keras.models.Sequential:
        """
        build network with A3C parameters
        Output:
            network (keras.model): neural net for architecture
        """
        model = keras.Sequential()

        if self.simple_a3c:
            layer_input = keras.Input(batch_shape=(None, self.input_shape))
            l_dense = keras.layers.Dense(250, activation='relu')(layer_input)

            out_actions = keras.layers.Dense(
                self.output_dim, activation='softmax')(l_dense)
            out_value = keras.layers.Dense(1, activation='linear')(l_dense)

            model = keras.Model(inputs=[layer_input], outputs=[
                                out_actions, out_value])
            model._make_predict_function()
            model.summary()
            # for evaluation purpose
            self.model_yaml = model.to_yaml()
            # compile model
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.l_rate,
                                                  decay=0.9,
                                                  momentum=self.params['gradient_momentum'])
            model.compile(optimizer=optimizer,
                          loss='mean_squared_error',
                          metrics=['accuracy'])
#                          loss=tf.losses.huber_loss,
            if self.debug:
                print(Font.yellow + '–' * 100 + Font.end)
                print(Font.yellow + 'Model: ' + Font.end)
                print(model.to_yaml())
                print(Font.yellow + '–' * 100 + Font.end)

        else:
            # first hidden layer
            # input shape = (img_height, img_width, n_channels)
            layer_input = keras.Input(batch_shape=(None, self.input_shape))
            l_hidden1 = keras.layers.Conv2D(filters=16, kernel_size=(8, 8),
                                            strides=4, activation='relu', data_format="channels_last")(layer_input)
            # second hidden layer
            l_hidden2 = keras.layers.Conv2D(filters=32, kernel_size=(4, 4),
                                            strides=2, activation='relu')(l_hidden1)
            # third hidden layer
            l_flatten = keras.layers.Flatten()(l_hidden2)
            l_full1 = keras.layers.Dense(256, activation='relu')(l_flatten)
            out_actions = keras.layers.Dense(self.output_dim, activation='softmax')(l_full1)
            out_value = keras.layers.Dense(1, activation='linear')(l_full1)
            
            model = keras.Model(inputs=[layer_input], outputs=[
                                out_actions, out_value])
            model._make_predict_function()
            model.summary()
            self.model_yaml = model.to_yaml()
            # compile model
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.l_rate,
                                                  decay=0.9,
                                                  momentum=self.params['gradient_momentum'])
            model.compile(optimizer=optimizer,
                          loss='mean_squared_error',
                          metrics=['accuracy'])
            if self.debug:
                print(Font.yellow + '–' * 100 + Font.end)
                print(Font.yellow + 'Model: ' + Font.end)
                print(model.to_yaml())
                print(Font.yellow + '–' * 100 + Font.end)

        return model

    def warmstart(self, path: str) -> None:
        """
        reading weights from disk
        Input:
            path (str): path from where to read the weights
        """
        print(Font.yellow + '–' * 100 + Font.end)
        print(Font.yellow + 'Warmstart, load weights from: ' +
              os.path.join(path, 'weights.h5') + Font.end)
        print(Font.yellow + 'Setting epsilon to eps_min: ' +
              str(self.epsilon_min) + Font.end)
        print(Font.yellow + '–' * 100 + Font.end)
        self.epsilon = self.epsilon_min
        self.model.load_weights(os.path.join(path, 'weights.h5'))
        self.target_model.load_weights(os.path.join(path, 'target_weights.h5'))

    def do_training(self, is_testing=False):
        """
        train A3C algorithm with replay buffer and minibatch
        Input:
            is_testing (bool): evaluation of the network
        """
        while self.episode_num < 1:
            if not is_testing:
                self._replay(batch_size=self.minibatch_size)
                self.episode_num += 1
            if is_testing:
                break
        self.episode_num = 0


    def reset_gradients(self):
        r"""
        set gradients $d\theta <- 0$ and $d\theta_v <- 0$
        """
        pass

    def synchronize_from_parameter_server(self):
        r"""
        synchronize thread-specific parameters $\theta' = \theta$ and $\theta_v ' = \theta_v $
        """
        pass

    def accumulate_gradients(self):
        pass

    def act(self, state) -> (int, float):
        """
        return action from neural net
        Input:
            state (np.array): the current state as shape (img_height, img_width, 1)
        Output:
            action (int): action number
            q_val (float): expected q_value
        """
        # state = state[np.newaxis, ...]
        if self.simple_a3c:
            s = state.reshape((1, self.input_shape[0]))
            if self.debug:
                print(Font.yellow + '–' * 100 + Font.end)
                print(Font.yellow + 'A3C "act" fct here:' + Font.end)
                print('state shape: ', state.shape)
                print('state input: \n', state)
                print(Font.yellow + '–' * 100 + Font.end)
        else:
            s = state
        q_vals = self.model.predict(s)
        if self.debug:
            print(Font.yellow + '–' * 100 + Font.end)
            print('q vals: ', q_vals)
            print(Font.yellow + '–' * 100 + Font.end)
        action = misc.eps_greedy(
            q_vals=q_vals[0], eps=self.epsilon, rng=self.rng)
        # return action
        return action, np.amax(q_vals[0])

    def calc_eps_decay(self, step_counter: int) -> None:
        """
        calculates new epsilon for the given step counter according to the annealing formula
        y = -mx + c
        eps = -(eps - eps_min) * counter + eps
        Input:
            counter (int): step counter
        """
        if self.warmstart_flag:
            self.epsilon = self.epsilon_min
        else:
            if self.debug:
                print(Font.yellow + '–' * 100 + Font.end)
                print(Font.yellow + 'A3C "calc_eps_decay" fct output:' + Font.end)
                print('eps:', self.params['epsilon'])
                print('counter:', step_counter)
                coord_y = (self.params['epsilon'] - self.epsilon_min)
                coord_x = self.params['eps_max_frame']
                print('mx + c:', - coord_y/coord_x *
                      step_counter + self.params['epsilon'])
                print(Font.yellow + '–' * 100 + Font.end)
            if self.epsilon > self.epsilon_min:
                self.epsilon = -((self.params['epsilon'] - self.epsilon_min) /
                                 self.params['eps_max_frame']) * step_counter + self.params['epsilon']

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
        self.replay_buffer.add(obs_t=state, act=action,
                               rew=reward, obs_tp1=next_state, done=done)

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

    def _replay(self, batch_size: int = 32) -> None:
        """
        get data from replay buffer and train session
        Input:
            batch_size (int): size of batch to sample from replay buffer
        """
        batch_size = min(batch_size, self.replay_buffer.__len__())
        # get 5 arrays in minibatch for state, action, reward, next_state, done
        minibatch = self.replay_buffer.sample(batch_size=batch_size)
        # all of type np.array -> suffix "_a"
        state_a, action_a, reward_a, next_state_a, done_a = minibatch

        if self.simple_a3c:
            state_input = np.zeros((batch_size, 100))
            next_state_input = np.zeros((batch_size, 100))
            action, reward, done = [], [], []
            # state_input = state_a[:, 0, :]
            # next_state_input = next_state_a[:, 0, :]
            state_input = state_a
            next_state_input = next_state_a
        else:
            dim = (batch_size, ) + self.input_shape
            state_input = np.zeros(dim)
            next_state_input = np.zeros(dim)
            # state_a.shape = (batch_size, 1, height, width, n_channels)
            state_input = state_a[:, 0, ...]
            # next_state_a.shape = (batch_size, 1, height, width, n_channels)
            next_state_input = next_state_a[:, 0, ...]
            action, reward, done = [], [], []

        if self.debug:
            print(Font.yellow + '–' * 100 + Font.end)
            print('input dim: ', self.input_shape)
            print('state_a shape: ', state_a.shape)
            print('input state shape: ', state_input.shape)
            print(Font.yellow + '–' * 100 + Font.end)

        for i in range(batch_size):
            action.append(action_a[i])
            reward.append(reward_a[i])
            done.append(done_a[i])

        # y = self.model.predict(state_input)
        # y_target = self.target_model.predict(next_state_input)
        y = self.model.predict_on_batch(state_input)
        y_target = self.target_model.predict_on_batch(next_state_input)

        # "fit"-method feeds input and output pairs to the model
        # then the model will train on those data to approximate the output
        # based on the input
        # [src](https://keon.io/deep-q-learning/)
        for i in range(batch_size):
            if self.debug:
                print('y[i][action[i]]:', y[i][action[i]])
                print('action[i]:', action[i])
                print('reward[i]:', reward[i])
                print('y_target[i]:', y_target[i])
            if done[i]:
                y[i][action[i]] = reward[i]
            else:
                y[i][action[i]] = reward[i] + self.gamma * np.amax(y_target[i])

        # self.loss = self.model.train_on_batch(state_input, y)
        self.model.fit(state_input, y, batch_size=batch_size,
                       epochs=1, verbose=0, callbacks=[self.csv_logger])

    def main(self):
        print('A3C here')
        print('–' * 30)


if __name__ == '__main__':
    print('A3C __main__')
