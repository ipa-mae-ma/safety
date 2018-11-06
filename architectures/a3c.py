
"""
Created on October 1, 2018

@author: mae-ma
@attention: architectures for the safety DRL package
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 1.1.0

#############################################################################################

History:
- v1.1.0: running version
- v1.0.0: first init
"""

import numpy as np
import os
import yaml
import tensorflow as tf
from tensorflow import keras
import threading
import time
from matplotlib import pyplot as plt
from texttable import Texttable

from architectures.replay_buffer import ReplayBuffer
import architectures.misc as misc
from architectures.misc import Font
from architectures.agent import Agent


ACTOR = 0
CRITIC = 1


class AsynchronousAdvantageActorCriticGlobal(Agent):
    def __init__(self,
                 input_shape: tuple = (10, 10),
                 output_dim: int = 4,
                 warmstart: bool = False,
                 warmstart_path: str = None,
                 simple_a3c: bool = True,
                 params: dict = None,
                 env=None) -> None:
        """
        Input:
            input_shape (int): Input shape of format type (n_img, img_height, img_width, n_channels)
            output_dim (int): Output dimension
            warmstart (bool): load network weights from disk
            warmstart_path (str): path where the weights are stored
            simple_a3c (bool): use simplified network
            params (dict): parameter dictionary with all config values
            env: fruit game environment
        """
        super(AsynchronousAdvantageActorCriticGlobal,
              self).__init__(parameters=params)
        self.session = tf.InteractiveSession()
        keras.backend.set_session(self.session)
        keras.backend.manual_variable_initialization(True)

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

        if env is None:
            raise ValueError('Please provide an environment')
        else:
            self.env = env
        self.output_dim = output_dim  # number of actions
        self.l_rate = self.params['learning_rate']
        self.minibatch_size = self.params['minibatch_size']
        self.gamma = self.params['gamma']
        self.n_step_return = self.params['n_step_return']
        self.epsilon = self.params['epsilon']
        self.epsilon_min = self.params['epsilon_min']
        self.threads = self.params['threads']
        # debug flag
        self.debug = False
        self.delete_training_log(architecture='A3C')


        self.global_train_queue = [[], [], [], [], []]
        self.lock_queue = threading.Lock()
        # build neural nets
        self.model = self._build_network()
        self.graph = self._build_graph(model=self.model)
        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()
        self.default_graph.finalize()

        self.lock_queue = threading.Lock()
        # s, a, r, s', s' terminal mask

        self.save_model_yaml(architecture='A3C')

    def _build_network(self) -> keras.models.Sequential:
        """
        build network with A3C parameters
        Output:
            network (keras.model): neural net for architecture
        """
        model = keras.Sequential()

        if self.simple_a3c:
            layer_input = keras.Input(batch_shape=(None, 100), name='input')
            l_dense = keras.layers.Dense(
                250, activation='relu', kernel_initializer='he_uniform', name='dense')(layer_input)

            out_actions = keras.layers.Dense(
                self.output_dim, activation='softmax', name='out_a')(l_dense)
            out_value = keras.layers.Dense(
                1, activation='linear', kernel_initializer='he_uniform', name='out_v')(l_dense)

            model = keras.Model(inputs=[layer_input], outputs=[
                                out_actions, out_value])
            model._make_predict_function()
            model.summary()
            self.model_yaml = model.to_yaml()

        return model

    def _build_graph(self, model):
        """
        make loss function for policy gradient
        backpropagation input: 
            [ log(action_probability) * advantages]
        with:
            advantages = discounted_reward - values
        """
        s_t = tf.placeholder(tf.float32, shape=(None, 100))
        a_t = tf.placeholder(tf.float32, shape=(None, self.output_dim))
        r_t = tf.placeholder(tf.float32, shape=(None, 1))

        p, v = model(s_t)

        log_prob = tf.log(tf.reduce_sum(p * a_t, axis=1, keepdims=True) + 1e-10)
        advantage = r_t - v

        loss_policy = - log_prob * tf.stop_gradient(advantage)
        loss_value = self.params['loss_value_coefficient'] * tf.square(advantage)
        entropy = self.params['loss_entropy_coefficient'] * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keepdims=True)

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        optimizer = keras.optimizers.RMSprop(lr=self.l_rate,
                                             rho=0.9)
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=self.l_rate, decay=0.9)
        minimize = optimizer.minimize(loss_total)
        return s_t, a_t, r_t, minimize

    def predict(self, state) -> (np.array, np.array):
        with self.default_graph.as_default():
            p, v = self.model.predict(state)
            return p, v

    def predict_p(self, state) -> np.array:
        with self.default_graph.as_default():
            p, v = self.model.predict(state)
            return p

    def predict_v(self, state) -> np.array:
        with self.default_graph.as_default():
            p, v = self.model.predict(state)
            return v

    def n_step_queue(self, s, a, r, s_):
        with self.lock_queue:
            self.global_train_queue[0].append(s)
            self.global_train_queue[1].append(a)
            self.global_train_queue[2].append(r)

            NONE_STATE = np.zeros((100, ))
            if s_ is None:
                self.global_train_queue[3].append(NONE_STATE)
                self.global_train_queue[4].append(0.0)
            else:
                self.global_train_queue[3].append(s_)
                self.global_train_queue[4].append(1.0)

    def optimize(self):
        # print(Font.yellow + 'train queue len: '+ str(len(self.global_train_queue[-1])) + Font.end)
        if len(self.global_train_queue[-1]) < 32:
            time.sleep(0)
            return

        with self.lock_queue:
            # print(Font.yellow + 'inside' + Font.end)
            # more thread could have passed without lock
            if len(self.global_train_queue[-1]) < 32:
                return 									# we can't yield inside lock
            
            # print(Font.yellow + 'inside 2' + Font.end)
            s, a, r, s_, s_mask = self.global_train_queue
            self.train_queue = [[], [], [], [], []]

        s = np.vstack(s)
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        s_mask = np.vstack(s_mask)
        p, v = self.model.predict(s_)
        # r = r + self.gamma ** self.n_step_return * v * s_mask

        # print(Font.yellow + 'Optimize' + Font.end)

        s_t, a_t, r_t, minimize = self.graph
        self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r})

    def do_training(self):
        agents = [AsynchronousAdvantageActorCriticAgent(index=i,
                                                        n_step_queue=self.n_step_queue,
                                                        optimize=self.optimize,
                                                        model=self.model,
                                                        env=self.env,
                                                        action_dim=self.output_dim,
                                                        state_shape=self.input_shape,
                                                        params=self.params) for i in range(self.threads)]

        opts = [Optimizer(optimize=self.optimize)]

        for opt in opts:
            opt.start()

        for agent in agents:
            agent.start()


        scores = []
        Q_vals = []
        names = ['Index', 'Episode', 'Epoch', 'Reward', 'Step Counter', 'Epsilon']
        while True:
            time.sleep(1)
            t = Texttable()
            for agent in agents:
                index, episode, epoch, rew, step_counter, epsilon, q_vals = agent.get_info()
                t.add_rows([names, [index, episode, epoch, rew, step_counter, epsilon]])
                print(t.draw() + '\n\n')
                # print('\nindex: {}/{} \nepisode: {}/{} \nepoch: {}/{} \nscore: {} \neps: {:.3f} \nsum of steps: {}'.
                #       format(index, self.threads-1, episode, self.num_episodes, epoch,
                #              self.num_epochs, rew, epsilon, step_counter))
                scores.append(rew)
                Q_vals.append(q_vals)
            with open('scores.yml', 'w') as f:
                yaml.dump(scores, f)
            plot = [np.mean(scores[n:n+5])
                    for n in range(0, len(scores)-5)]
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))
            ax1.plot(range(len(plot)), plot, 'r', label='Scores')
            ax1.set_xlabel('Episodes/5')
            ax1.set_ylabel('Mean scores over last 5 episodes')
            ax2.plot(range(len(Q_vals)), Q_vals, 'b', label='Q values')
            ax1.legend(fontsize=25)
            ax2.legend(fontsize=25)
            ax1.grid()
            ax2.grid()
            fig.tight_layout()
            plt.savefig('./a3c.pdf')


        for agent in agents:
            agent.stop()
        for agent in agents:
            agent.join()

    def save_weights(self, path: str) -> None:
        """
        save model weights
        Input:
            path (str): filepath
        """
        self.model.save_weights(filepath='model_' + path)

    def main(self):
        print('A3C here')
        print('–' * 30)


class AsynchronousAdvantageActorCriticAgent(threading.Thread):
    def __init__(self,
                 n_step_queue,
                 optimize,
                 index: int,
                 model,
                 env,
                 action_dim: int,
                 state_shape: tuple,
                 params: dict = None) -> None:

        super(AsynchronousAdvantageActorCriticAgent, self).__init__()

        self.n_step_queue = n_step_queue
        self.optimize = optimize
        self.index = index
        self.model = model
        self.env = env
        self.action_dim = action_dim
        self.state_shape = state_shape
        self.mc = misc
        self.overblow_factor = 8

        self.local_train_queue = []
        self.params = params
        self.gamma = self.params['gamma']
        self.num_epochs = self.params['num_epochs']
        self.num_episodes = self.params['num_episodes']
        self.num_steps = self.params['num_steps']
        self.epsilon = self.params['epsilon']
        self.epsilon_min = self.params['epsilon_min']
        self.n_step_return = self.params['n_step_return']
        self.rng = np.random.RandomState(self.params['random_seed'])
        self.debug = False
        self.info = [(0, 0, 0, 0, 0, 0, 0)]

    def run(self):
        episode = 0
        step_counter = 0
        q_vals = 0
        for epoch in range(self.num_epochs):
            for episode in range(self.num_episodes):
                obs, _, _, _ = self.env.reset()
                state = self.mc.make_frame(obs, do_overblow=False,
                                           overblow_factor=self.overblow_factor,
                                           normalization=False).reshape(self.state_shape)
                rew = 0
                for step in range(self.num_steps):
                    time.sleep(0.001)
                    action, q_vals = self.act(state)
                    obs, r, terminal, info = self.env.step(action)
                    self.calc_eps_decay(step_counter=step_counter)
                    # self.env.render()
                    next_state = self.mc.make_frame(obs, do_overblow=False,
                                                    overblow_factor=self.overblow_factor,
                                                    normalization=False).reshape(self.state_shape)
                    self.memory(state, action, r, next_state)
                    state = next_state
                    rew += r
                    if step == self.num_steps - 1:
                        terminal = True
                        next_state = None
                    # self.train_episode(terminal=terminal)
                    step_counter += 1
                    if terminal:
                        next_state = None
                        self.train_episode(terminal=terminal)
                        episode += 1
                        # self.optimize()
                        self.set_info(self.index, episode,
                                      epoch, rew, step_counter, self.epsilon, q_vals)
                        break

    def stop(self):
        self.stop_signal = True

    def get_info(self) -> tuple:
        """
        return the scores list for one thread
        Output:
            (index:int, episode:int, epoch:int, rew:float, step_counter:int, epsilon:float, q_vals:float)
        """
        index = self.info[-1][0]
        episode = self.info[-1][1]
        epoch = self.info[-1][2]
        rew = self.info[-1][3]
        step_counter = self.info[-1][4]
        epsilon = self.info[-1][5]
        q_vals = self.info[-1][6]
        return index, episode, epoch, rew, step_counter, epsilon, q_vals

    def set_info(self, index, episode, epoch, rew, step_counter, epsilon, q_vals) -> None:
        self.info.append((index, episode, epoch, rew, step_counter, epsilon, q_vals))

    def calc_eps_decay(self, step_counter: int) -> None:
        """
        calculates new epsilon for the given step counter according to the annealing formula
        y = -mx + c
        eps = -(eps - eps_min) * counter + eps
        Input:
            counter (int): step counter
        """
        # if self.warmstart_flag:
        #     self.epsilon = self.epsilon_min
        if self.epsilon > self.epsilon_min:
            self.epsilon = -((self.params['epsilon'] - self.epsilon_min) /
                             self.params['eps_max_frame']) * step_counter + self.params['epsilon']

    def memory(self, s: np.array, a: int, r: float, s_: np.array) -> None:
        """
        save <s, a, r, s'> every step
        Input:
            s (np.array): s
            a (int): a
            r (float): r
            s_ (np.array): s'
        """
        act = np.zeros(self.action_dim)
        act[a] = 1
        self.local_train_queue.append((s, act, r, s_))

    def train_episode(self, terminal: bool):
        """
        update the policy and target network every episode
        Input:
            terminal (bool): terminal flag
        """
        def get_sample(memory, n):
            r = 0.
            for i in range(n):
                r += memory[i][2] * (self.gamma ** i)
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n-1]
            return s, a, r, s_

        if terminal is True:
            while len(self.local_train_queue) > 0:
                n = len(self.local_train_queue)
                s, a, r, s_ = get_sample(self.local_train_queue, n)
                self.n_step_queue(s, a, r, s_)
                self.local_train_queue.pop(0)
            # self.optimize()

    def act(self, state) -> (int, float):
        """
        return action from neural net
        Input:
            state (np.array): the current state as shape (img_height, img_width, 1)
        Output:
            action (int): action number
            policy (float): expected q_value
        """
        s = state.reshape((1, self.state_shape[0]))
        policy, value = self.model.predict(s)
        rng = np.random.RandomState(np.random.randint(low=1,high=111))
        action = misc.eps_greedy(policy[0], self.epsilon, rng=rng)
        # return np.random.choice(self.action_dim, size=1, p=policy)
        return action, value[0]


class Optimizer(threading.Thread):
    def __init__(self, optimize):
        super(Optimizer, self).__init__()
        self.optimize = optimize
        self.stop_signal = False

    def run(self):
        self.optimize()

    def stop(self):
        self.stop_signal = True



if __name__ == '__main__':
    print('A3C __main__')
