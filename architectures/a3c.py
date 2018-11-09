
"""
Created on October 1, 2018

@author: mae-ma
@attention: architectures for the safety DRL package
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 3.0.2

#############################################################################################

History:
- v3.0.2: output entropy
- v3.0.1: add random number in [0.01, 0.3] to epsilon
- v3.0.0: use only one model for policy and value
- v2.0.2: update discounted rewards
- v2.0.1: add dropout layer
- v2.0.0: support complex conv net
- v1.2.3: first use of simple flag
- v1.2.2: update doc and plot options
- v1.2.1: break loop with signal
- v1.2.0: plot loss
- v1.1.0: first keras setup
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

import architectures.misc as misc
from architectures.misc import Font
from architectures.agent import Agent


ACTOR = 0
CRITIC = 1

class A3CGlobal(Agent):
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
        super(A3CGlobal,
              self).__init__(parameters=params)
        self.params = params
        self.print_params(architecture_name='A3C')

        self.rng = np.random.RandomState(self.params['random_seed'])
        self.simple_a3c = simple_a3c

        # input shape = (img_height * img_width, )
        self.input_shape = input_shape
        
        if env is None:
            raise ValueError('Please provide an environment')
        else:
            self.env = env
        self.output_dim = output_dim  # number of actions
        self.l_rate = self.params['learning_rate']
        self.gamma = self.params['gamma']
        self.n_step_return = self.params['n_step_return']
        self.epsilon = self.params['epsilon']
        self.epsilon_min = self.params['epsilon_min']
        self.threads = self.params['threads']
        # debug flag
        self.debug = False
        self.delete_training_log(architecture='A3C')

        self.csv_logger = keras.callbacks.CSVLogger(
            'training_log_A3C.csv', append=True)

        # build neural net
        self.model = self._build_network()
        self.single_opt = self._build_optimizer()
        
        self.save_model_yaml(architecture='A3C')
        
        self.sess = tf.InteractiveSession()
        keras.backend.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

    def _build_network(self) -> (keras.models.Sequential, keras.models.Sequential):
        """
        build network with A3C parameters
        Output:
            actor (keras.model): actor model
            critic (keras.model): critic model
        """
        
        if self.simple_a3c:
            input_shape = (None,) + self.input_shape
            # layer_input = keras.Input(batch_shape=(None, self.input_shape))
            layer_input = keras.Input(batch_shape=input_shape, name='input')
            l_dense = keras.layers.Dense(250, activation='relu', kernel_initializer='he_uniform', name='dense')(layer_input)

            out_actions = keras.layers.Dense(self.output_dim, activation='softmax', name='out_a')(l_dense)
            out_value = keras.layers.Dense(1, activation='linear', kernel_initializer='he_uniform', name='out_v')(l_dense)

            model = keras.Model(inputs=[layer_input], outputs=[out_actions, out_value])
            model._make_predict_function()
            model.summary()
            self.model_yaml = model.to_yaml()
        
            if self.debug:
                print(Font.yellow + '–' * 100 + Font.end)
                print(Font.yellow + 'Model: ' + Font.end)
                print(model.to_yaml())
                print(Font.yellow + '–' * 100 + Font.end)

            return model
        else:
            input_shape = (None,) + self.input_shape + (1,)  # (height, width, channels=1)
            # first hidden layer
            # input shape = (img_height, img_width, n_channels)
            layer_input = keras.Input(batch_shape=(None, 80, 80, 1))
            l_hidden1 = keras.layers.Conv2D(filters=16, kernel_size=(8, 8),
                                            strides=4, activation='relu',
                                            kernel_initializer='he_uniform', data_format='channels_last')(layer_input)
            # second hidden layer
            l_hidden2 = keras.layers.Conv2D(filters=32, kernel_size=(4, 4),
                                            strides=2, activation='relu', kernel_initializer='he_uniform')(l_hidden1)
            dropout = keras.layers.Dropout(.3)(l_hidden2)
            # third hidden layer
            # l_flatten = keras.layers.Flatten()(l_hidden2)
            l_flatten = keras.layers.Flatten()(dropout)
            l_full1 = keras.layers.Dense(
                256, activation='relu', kernel_initializer='he_uniform')(l_flatten)
            out_actions = keras.layers.Dense(self.output_dim, activation='softmax', kernel_initializer='he_uniform')(l_full1)
            out_value = keras.layers.Dense(1, activation='linear', kernel_initializer='he_uniform')(l_full1)

            model = keras.Model(inputs=[layer_input], outputs=[out_actions, out_value])
            model._make_predict_function()
            model.summary()
            self.model_yaml = model.to_yaml()
            return model


    def _build_optimizer(self):
        """
        make loss function for both actor and critic
        """
        K = keras.backend
        action = K.placeholder(shape=(None, self.output_dim))
        # advantages = K.placeholder(shape=(None, ))
        discounted_reward = K.placeholder(shape=(None, ))

        policy = self.model.output[0]
        values = self.model.output[1]

        alpha = self.params['loss_value_coefficient']
        beta = self.params['loss_entropy_coefficient']
        # ACTOR loss
        log_prob = K.log( K.sum(policy * action, axis=1, keepdims=True) + 1e-10)
        advantages = discounted_reward - values
        loss_policy = log_prob * K.stop_gradient(advantages)
        # loss_actor = K.sum(loss_policy, keepdims=True)
        loss_actor = K.mean(loss_policy)
        # entropy = beta * K.sum(policy * K.log(policy + 1e-10), keepdims=True)
        entropy = beta * K.mean(policy * K.log(policy + 1e-10))
        
        # CRITIC loss
        # loss_value = alpha * K.mean(K.square( discounted_reward - values))
        # loss_value = K.mean(alpha * K.sum(K.square(advantages)))
        loss_value = alpha * K.mean(K.square(advantages))
        

        loss_total = loss_actor - entropy + loss_value


        # optimizer = keras.optimizers.RMSprop(lr=self.l_rate,
        #                                     rho=0.9,
        #                                     decay=0.99)
        # updates = optimizer.get_updates(loss_total, self.model.trainable_weights)
        # train = K.function([self.model.input, action, discounted_reward], 
        #                     [loss_actor, loss_value], updates=updates)
        # return train

        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.l_rate)
        # Compute the gradients for a list of variables.
        minimize = optimizer.minimize(loss_total)
        
        grads_and_vars = optimizer.compute_gradients(loss_total)
  
        # grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
        # need to the 'gradient' part, for example cap them, etc.
        # TODO: check for lower clipping values
        capped_grads_and_vars = [(tf.clip_by_value(gv[0], -0.5, +0.5), gv[1]) for gv in grads_and_vars]
        train_op = optimizer.apply_gradients(capped_grads_and_vars)
        train = K.function([self.model.input, action, discounted_reward], 
                            [loss_actor, loss_value, entropy], updates=[train_op])
                            # [loss_actor, loss_value], updates=[minimize_actor, minimize_critic])
                            # [loss_actor, loss_value], updates=[minimize])
        return train

    def train(self):
        agents = [A3CAgent(index=i,
                simple=self.simple_a3c,
                model=self.model,
                optimizer=self.single_opt, 
                env=self.env, 
                action_dim=self.output_dim, 
                state_shape=self.input_shape, 
                params=self.params) for i in range(self.threads)]
        
        for agent in agents:
            agent.start()

        scores = []
        Q_vals = []
        losses_actor = []
        losses_critic = []
        entropies = []
        names = ['Index', 'Episode', 'Epoch',
                 'Reward', 'Step Counter', 'Epsilon']
        while True:
            time.sleep(0.1)
            t = Texttable()
            for agent in agents:
                index, episode, epoch, rew, step_counter, epsilon, q_vals, loss_actor, loss_critic, entropy = agent.get_info()
                t.add_rows([names, [index, episode, epoch, rew, step_counter, epsilon]])
                print(t.draw() + '\n\n')
                scores.append(rew)
                Q_vals.append(q_vals)
                losses_actor.append(loss_actor)
                losses_critic.append(loss_critic)
                entropies.append(entropy)
            with open('scores.yml', 'w') as f:
                yaml.dump(scores, f)
            stop_signals = []
            for agent in agents:
                stop_signals.append(agent.get_stop())
            if all(stop_signals):
                for agent in agents:
                    agent.join()
                break
                
            fig, (ax1, ax2left) = plt.subplots(2, 1, figsize=(18, 12))
            # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))
            if len(scores) <= 11:
                ax1.plot(range(len(scores)), scores, 'r', label='Scores')
            else:
                plot = misc.smooth(np.array(scores), 11)
                ax1.plot(range(len(plot)), plot, 'r', label='Scores')
            ax1.set_xlabel('Episodes')
            ax1.set_ylabel('Scores')
            ax1.grid()
            ax1.legend(fontsize=25)

            legend2left, = ax2left.plot(range(len(losses_actor)), losses_actor, 'b', label='loss actor')
            ax2right = ax2left.twinx()
            legend2right, = ax2right.plot(range(len(losses_critic)), losses_critic, 'r', label='loss critic')
            legend2right2, = ax2right.plot(range(len(entropies)), entropies, 'g', label='entropy')
            ax2left.set_xlabel('Episodes')
            ax2left.set_ylabel('Loss Actor', color='blue')
            ax2right.set_ylabel('Loss Critic', color='red')
            ax2left.grid()
            plt.legend(handles=[legend2left, legend2right, legend2right2], fontsize=25, loc='upper right')
            fig.tight_layout()
            plt.savefig('./a3c.pdf')
            plt.close(fig)


    def save_weights(self, path: str) -> None:
        """
        save model weights
        Input:
            path (str): filepath
        """
        self.model.save_weights(filepath=path)

    def main(self):
        print('A3C here')
        print('–' * 30)


class A3CAgent(threading.Thread):
    def __init__(self,
                 index: int,
                 simple: bool,
                 model,
                 optimizer,
                 env,
                 action_dim: int,
                 state_shape: tuple,
                 params: dict = None) -> None:
        
        super(A3CAgent, self).__init__()
        
        self.index = index
        self.simple_a3c = simple
        self.model = model
        self.env = env
        self.action_dim = action_dim
        self.state_shape = state_shape
        self.optimizer = optimizer
        self.mc = misc
        self.overblow_factor = 8

        self.states = []
        self.rewards = []
        self.actions = []
        self.next_states = []

        self.params = params
        self.gamma = self.params['gamma']
        self.num_epochs = self.params['num_epochs']
        self.num_episodes = self.params['num_episodes']
        self.num_steps = self.params['num_steps']
        self.epsilon = min(self.params['epsilon'] + np.random.randint(low=1, high=30) / 100, 1.0)
        self.epsilon_start = self.epsilon
        self.epsilon_min = self.params['epsilon_min']
        self.n_step_return = self.params['n_step_return']
        self.rng = np.random.RandomState(self.params['random_seed'])
        self.debug = False

        names = ['Index', 'Episode', 'Epoch',
                 'Reward', 'Step Counter', 'Epsilon', 
                 'Q Vals', 'Loss Actor', 'Loss Critic', 'Entropy']
        name_tuple = tuple([_ for _ in range(len(names))])
        self.info = [name_tuple]
        self.stop_signal = False


    def run(self):
        step_counter = 0
        q_vals = 0
        while not self.stop_signal:
            for epoch in range(self.num_epochs):
                for episode in range(self.num_episodes):
                    obs, _, _, _ = self.env.reset()
                    if self.simple_a3c:
                        norm = True
                        state = self.mc.make_frame(obs, do_overblow=False,
                                                    overblow_factor=self.overblow_factor,
                                                    normalization=norm).reshape(self.state_shape)
                    else:
                        norm = True
                        state = self.mc.make_frame(obs, do_overblow=True,
                                                    overblow_factor=self.overblow_factor,
                                                   normalization=norm)[..., np.newaxis]
                        
                    rews = 0
                    for step in range(self.num_steps):
                        # time.sleep(0.1)
                        action, q_vals = self.act(state)
                        obs, reward, terminal, info = self.env.step(action)
                        self.calc_eps_decay(step_counter=step_counter)
                        rews += reward
                        self.env.render()
                        if self.simple_a3c:
                            next_state = self.mc.make_frame(obs, do_overblow=False,
                                                         overblow_factor=self.overblow_factor,
                                                         normalization=norm).reshape(self.state_shape)
                        else:
                            next_state = self.mc.make_frame(obs, do_overblow=True,
                                                         overblow_factor=self.overblow_factor,
                                                            normalization=norm)[..., np.newaxis]

                        self.memory(state, action, reward, next_state)
                        state = next_state
                        step_counter += 1
                        if step_counter == self.num_epochs * self.num_episodes * self.num_steps - 10:
                            self.stop_signal = True
                        if terminal or step == self.num_steps - 1:
                            loss_actor, loss_critic, entropy = self.train_episode(done=terminal)
                            self.set_info(self.index, episode, epoch, rews, 
                                            step_counter, self.epsilon, q_vals, 
                                            loss_actor, loss_critic, entropy)
                            break

    def stop(self) -> None:
        """
        stops training
        """
        self.stop_signal = False

    def get_stop(self) -> bool:
        return self.stop_signal

    def get_info(self) -> tuple:
        """
        return the scores list for one thread
        Output:
            (index:int, episode:int, epoch:int, rew:float, 
            step_counter:int, epsilon:float, q_vals:float
            loss_actor:float, loss_critic:float)
        """
        index = self.info[-1][0]
        episode = self.info[-1][1]
        epoch = self.info[-1][2]
        rew = self.info[-1][3]
        step_counter = self.info[-1][4]
        epsilon = self.info[-1][5]
        q_vals = self.info[-1][6]
        loss_actor = self.info[-1][7]
        loss_critic = self.info[-1][8]
        entropy = self.info[-1][9]
        return index, episode, epoch, rew, step_counter, epsilon, q_vals, loss_actor, loss_critic, entropy

    def set_info(self, index, episode, epoch, rew, 
                step_counter, epsilon, q_vals, 
                loss_actor, loss_critic, entropy) -> None:
        self.info.append((index, episode, epoch, rew, step_counter,
                          epsilon, q_vals, loss_actor, loss_critic, entropy))


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
                             self.params['eps_max_frame']) * step_counter + self.epsilon_start

    def memory(self, state: np.array, action: int, reward: float, next_state: np.array) -> None:
        """
        save <s, a, r, s'> every step
        Input:
            state (np.array): s
            action (int): a
            reward (float): r
            next_state (np.array): s'
        """
        self.states.append(state)
        act = np.zeros(self.action_dim)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)
        self.next_states.append(next_state)

    def discount_rewards(self, rewards, done=False):
        """
        calculate discounted rewards
        Input:
            rewards (list): list of rewards
            done (bool): terminal flag
        Output:
            discounted_rewards (np.array): array with discounted rewards
        """
        discounted_rewards = np.zeros_like(rewards)
        running_rew = 0
        if not done:
            if self.simple_a3c:
                _, running_rew = self.model.predict(np.reshape(self.states[-1], (1, self.state_shape[0])))
            else:
                _, running_rew = self.model.predict(self.states[-1][np.newaxis, ...])

        for t in reversed(range(0, len(rewards))):
            running_rew = rewards[t] + self.gamma * running_rew
            discounted_rewards[t] = running_rew
        return discounted_rewards


    def train_episode(self, done) -> tuple:
        """
        update the policy and target network every episode
        Inputs:
            done (bool): if terminal state is reached
        Outputs:
            loss_actor (float): loss of optimizer
            loss_critic (float): loss of optimizer
        """
        policy, values = self.model.predict(np.array(self.states))
        values = np.reshape(values, len(values))

        discounted_rewards = self.discount_rewards(self.rewards, done)
        advantages = discounted_rewards - values

        # input: [self.actor.input, action, advantages, discounted_reward]
        # output: [loss_actor, loss_value, entropy]
        loss_actor, loss_value, entropy = self.optimizer([self.states, self.actions, discounted_rewards])

        if self.debug:
            print(Font.yellow + '-' * 100 + Font.end)
            print('loss actor:', loss_actor)
            print('loss critic:', loss_value)
            print(Font.yellow + '-' * 100 + Font.end)
        self.states, self.actions, self.rewards, self.next_states = [], [], [], []
        return loss_actor, loss_value, entropy

    def act(self, state) -> (int, float):
        """
        return action from neural net
        Input:
            state (np.array): the current state as shape (img_height, img_width, 1)
        Output:
            action (int): action number
            policy (float): expected q_value
        """
        if self.simple_a3c:
            s = state.reshape((1, self.state_shape[0]))
        else:
            s = state[np.newaxis,:,:,:]
        policy, value = self.model.predict(s)
        if self.debug:
            print(Font.yellow + 'ACT' + Font.end)
            print('action: ', np.random.choice(
                self.action_dim, size=1, p=policy[0]))
        import random
        # clip epsilon to 1.0
        eps = min(self.epsilon, 1.0)
        if random.random() <= eps:
            # random.randint(0, 4) -> x \in [0,4]
            action = random.randint(0, len(policy[0]) - 1)
        else:
            action = np.random.choice(self.action_dim, p=policy[0])
        # return on-policy action
        return action, np.max(policy[0])


if __name__ == '__main__':
    print('A3C __main__')
