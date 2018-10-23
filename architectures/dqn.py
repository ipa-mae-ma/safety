
"""
Created on October 1, 2018

@author: mae-ma
@attention: architectures for the safety DRL package
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 1.5.2

#############################################################################################

History:
- v1.5.2: act returns also predicted q_value
- v1.5.1: use only mean-squared-error as loss
- v1.5.0: update convnet to work with batches of np.arrays
- v1.4.2: implement soft update for target weights
- v1.4.1: save model to "model.yml" file
- v1.4.0: use tf cpp backend to perform training -> faster
- v1.3.1: remove log file at beginning of training
- v1.3.0: add warmstart function and simple neural net option
- v1.2.7: add new config params, delete old ones
- v1.2.6: add func for epsilon decay
- v1.2.5: use optimized for gpu usage
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
import tensorflow as tf
from tensorflow import keras
import tqdm

from architectures.replay_buffer import ReplayBuffer
import architectures.misc as misc
from architectures.misc import Font

# tf.enable_eager_execution()

# TODO: compare output of network after training with pyplot

class DeepQNetwork:
    def __init__(self,
                 env,
                 input_dim: tuple=(1, 14, 21, 1),
                 output_dim: int=4,
                 warmstart: bool=False,
                 warmstart_path: str=None,
                 simple_dqn: bool=True,
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
            warmstart (bool): load network weights from disk
            warmstart_path (str): path where the weights are stored
            simple (bool): use simplified DQN without conv_layers
            name (str, optional): TF Graph will be built under this name
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        cfg_file = os.path.join(dir_path, 'config_dqn.yml')
        self.params = yaml.safe_load(open(cfg_file, 'r'))

        nprs = np.random.RandomState
        self.rng = nprs(self.params['random_seed'])

        # self.mdp = mdp
        self.simple_dqn = simple_dqn

        if self.simple_dqn:
            # input dim = (img_height, img_width)
            self.input_dim = input_dim[0] * input_dim[1]
        else:
            # input dim = (img_height, img_width, n_channels)
            self.input_dim = input_dim + (1,)
        self.output_dim = output_dim  # number of actions
        self.l_rate = self.params['learning_rate']
        self.minibatch_size = self.params['minibatch_size']
        self.gamma = self.params['gamma']
        self.epsilon = self.params['epsilon']
        self.epsilon_min = self.params['epsilon_min']
        self.replay_buffer = ReplayBuffer(float(self.params['replay_memory_size']))

        # delete training log file in the beginning
        # if os.path.exists(os.path.join(os.path.dirname(__file__), 'training_log_DQN.csv')):
        if os.path.exists(os.path.join('/home/mae-ma/git/safety', 'training_log_DQN.csv')):
            print(Font.yellow + '–' * 100 + Font.end)
            print('delete old training log')
            print(Font.yellow + '–' * 100 + Font.end)
            os.remove(os.path.join(
                '/home/mae-ma/git/safety', 'training_log_DQN.csv'))

        self.csv_logger = keras.callbacks.CSVLogger('training_log_DQN.csv', append=True)
        # max number of epochs
        self.num_epochs = self.params['num_epochs']
        # number of episodes in one epoch
        self.num_episodes = self.params['num_episodes']
        # number of steps in one episode
        self.num_steps = self.params['num_steps']
        # debug flag
        self.debug = False

        # self.fps = 0
        self.episode_num = 0
        self.last_episode_steps = 0
        self.total_training_steps = 0
        self.score_agent = 0
        self.eval_steps = []
        self.eval_scores = []
        self.env = env
        self.name = name
        self.model_yaml = None
        self.model = self._build_network()
        self.target_model = self._build_network()

        self.loss = [0.0 for _ in self.model.metrics_names]

        print(Font.yellow + '–' * 100 + Font.end)
        print('Save model as "model.yml"')
        with open('model.yml', 'w') as file:
            file.write(self.model_yaml)
        print(Font.yellow + '–' * 100 + Font.end)
        self.warmstart_flag = warmstart
        if self.warmstart_flag:
            self.warmstart(warmstart_path)

    def _build_network(self) -> (keras.models.Sequential, keras.models.Sequential):
        """
        build network with DQN parameters
        Output:
            target_network (keras.model): neural nets for DQN
            evaluation_network (keras.model): neural nets for DQN
        """
        model = keras.Sequential()
        
        if self.simple_dqn:
            # if len(self.input_dim) == 3:
            #     input_dim = tuple([1] + list(self.input_dim))
            # inputs = keras.Input(input_dim, dtype='float32', name='inputs')
            # flatten = keras.layers.Flatten()(inputs)
            # hid = keras.layers.Dense(250, activation='relu', name='hidden')(flatten)
            # outputs = keras.layers.Dense(self.output_dim, activation='linear', name='outputs')(hid)
            # model = keras.Model(inputs=inputs, outputs=outputs)
            
            # use input_dim instead of input_shape | dim=100 => shape=(100,) are equal
            # model.add(keras.layers.Dense(250, input_dim=self.input_dim,
            model.add(keras.layers.Dense(250, input_shape=(100,),
                                            activation='relu', kernel_initializer='he_uniform'))
            # model.add(keras.layers.Dropout(rate=0.2))
            # model.add(keras.layers.Dense(20, activation='relu', kernel_initializer='he_uniform'))
            # model.add(keras.layers.Dropout(rate=0.2))
            # # model.add(keras.layers.Flatten())
            # # hidden layer
            # model.add(keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
            # output layer
            model.add(keras.layers.Dense(self.output_dim,
                                            activation='linear', kernel_initializer='he_uniform'))
            model.summary()
            # for evaluation purpose
            self.model_yaml = model.to_yaml()
            # compile model
            model.compile(optimizer=tf.train.RMSPropOptimizer(learning_rate=self.l_rate,
                                                                decay=0.9,
                                                                momentum=self.params['gradient_momentum']),
                            loss='mean_squared_error',
                            metrics=['accuracy'])
            if self.debug:
                print(Font.yellow + '–' * 100 + Font.end)
                print(Font.yellow + 'Model: ' + Font.end)
                print(model.to_yaml())
                print(Font.yellow + '–' * 100 + Font.end)

        else:
            # first hidden layer
            # input shape = (img_height, img_width, n_channels)
            model.add(keras.layers.Conv2D(input_shape=self.input_dim, filters=32,
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
            model.add(keras.layers.Dense(self.output_dim, activation='relu'))
            model.summary()
            self.model_yaml = model.to_yaml()
            # compile model
            model.compile(optimizer=tf.train.RMSPropOptimizer(learning_rate=self.l_rate,
                                                                decay=0.9,
                                                                momentum=self.params['gradient_momentum']),
                            loss='mean_squared_error',
                            metrics=['accuracy'])

        # target_model = keras.models.clone_model(model)
        return model

    def warmstart(self, path: str) -> None:
        """
        reading weights from disk
        Input:
            path (str): path from where to read the weights
        """
        print(Font.yellow + '–' * 100 + Font.end)
        print(Font.yellow + 'Warmstart, load weights from: ' + os.path.join(path, 'weights.h5') + Font.end)
        print(Font.yellow + 'Setting epsilon to eps_min: '+ str(self.epsilon_min) + Font.end)
        print(Font.yellow + '–' * 100 + Font.end)
        self.epsilon = self.epsilon_min
        self.model.load_weights(os.path.join(path, 'weights.h5'))
        self.target_model.load_weights(os.path.join(path, 'target_weights.h5'))


    def do_training(self, is_testing=False):
        """
        train DQN algorithm with replay buffer and minibatch
        Input:
            is_testing (bool): evaluation of the network
        """
        num_episodes = self.num_episodes
        # pbar = tqdm.tqdm(total=num_episodes, unit='Episodes', ncols=100)
        while self.episode_num < 1:
            # print(Font.yellow + Font.bold + 'Training ... ' + str(self.episode_num) + '/' + str(total_eps) + Font.end,
            #    end='\n')
            if not is_testing:
                # pbar.update(1)
                self._replay(batch_size=self.minibatch_size)
                self.episode_num += 1

            if is_testing:
                break
        # pbar.close()
        self.episode_num = 0

    def update_target_model(self, soft=False, beta:float=0.8) -> None:
        """
        update the target model with the weights from the trained model
        Input:
            soft (bool): use soft update
            beta (float): factor for soft update
        """
        if soft:
            weights = beta * self.target_model.get_weights() + (1 - beta) * self.model.get_weights()
            self.target_model.set_weights(weights)
        else:
            self.target_model.set_weights(self.model.get_weights())

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
        if self.simple_dqn:
            s = state.reshape((1,100))
            if self.debug:
                print(Font.yellow + '–' * 100 + Font.end)
                print(Font.yellow + 'DQN "act" fct here:' + Font.end)
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
        action = misc.eps_greedy(q_vals=q_vals[0], eps=self.epsilon, rng=self.rng)
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
                print(Font.yellow + 'DQN "calc_eps_decay" fct output:' + Font.end)
                print('eps:',self.params['epsilon'])
                print('counter:',step_counter)
                coord_y = (self.params['epsilon'] - self.epsilon_min)
                coord_x = self.params['eps_max_frame']
                print('mx + c:', - coord_y/coord_x * step_counter + self.params['epsilon'])
                print(Font.yellow + '–' * 100 + Font.end)
            if self.epsilon > self.epsilon_min:
                self.epsilon = -((self.params['epsilon'] - self.epsilon_min) / self.params['eps_max_frame']) * step_counter + self.params['epsilon']

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
        batch_size = min(batch_size, self.replay_buffer.__len__())
        # get 5 arrays in minibatch for state, action, reward, next_state, done
        minibatch = self.replay_buffer.sample(batch_size=batch_size)
        # all of type np.array -> suffix "_a"
        state_a, action_a, reward_a, next_state_a, done_a = minibatch
        ACTION, REWARD, DONE = 0, 1, 2


        if self.simple_dqn:
            state_input = np.zeros((batch_size, 100))
            next_state_input = np.zeros((batch_size, 100))
            action, reward, done = [], [], []
            # state_input = state_a[:, 0, :]
            # next_state_input = next_state_a[:, 0, :]
            state_input = state_a
            next_state_input = next_state_a
        else:
            dim = (batch_size, ) + self.input_dim
            state_input = np.zeros(dim)
            next_state_input = np.zeros(dim)
            # state_a.shape = (batch_size, 1, height, width, n_channels)
            state_input = state_a[:, 0, ...]
            # next_state_a.shape = (batch_size, 1, height, width, n_channels)
            next_state_input = next_state_a[:, 0, ...]
            action, reward, done = [], [], []

        if self.debug:
            print(Font.yellow + '–' * 100 + Font.end)
            print('input dim: ', self.input_dim)
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

        # # make one array of size (5, batch_size)
        # batch = np.concatenate((action_a, reward_a, done_a), axis=0).reshape(len(minibatch) - 2, -1)
        # # transpose to array of size (batch_size, 5)
        # batch = batch.transpose()
        # # for state, action, reward, next_state, done in minibatch:
        # i = 0
        # for sample in batch:
        #    if self.simple_dqn:
        #        state = state_a[i, ...]  # x[:,np.newaxis]
        #        next_state = next_state_a[i, ...]
        #    else:
        #        state = state_a[i, ...]  # x[:,np.newaxis]
        #        # add new axis to increase the fit the expeted size for the model
        #        state = state[np.newaxis, ..., np.newaxis]
        #        next_state = next_state_a[i, ...]
        #        # add new axis to increase the fit the expeted size for the model
        #        next_state = next_state[np.newaxis, ..., np.newaxis]
        #    
        #    action = sample[ACTION]
        #    reward = sample[REWARD]
        #    done = sample[DONE]
        #    # set the target for DQN
        #    y_target = reward
        #    if not done:
        #        # self.model.predict(s') -> Q(s',a')
        #        y_target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
        #    # create predictet target function
        #    y = self.model.predict(state)
        #    # self.model.predict(s) -> Q(s,a)
        #    # y.shape -> (1, 4)
        #    y[0][int(action)] = y_target
        #    # fit method feeds input and output pairs to the model
        #    # then the model will train on those data to approximate the output
        #    # based on the input
        #    # [src](https://keon.io/deep-q-learning/)
        #    input = state
        #    output = y
        #    self.model.fit(input, output, epochs=1, verbose=0, callbacks=[self.csv_logger])
        #    # self.model.fit(state, target_f, epochs=1, verbose=0)
        #    i += 1

    def main(self):
        print('DQN here')
        print('–' * 30)


if __name__ == '__main__':
    print('DQN __main__')
