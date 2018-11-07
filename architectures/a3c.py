
"""
Created on October 1, 2018

@author: mae-ma
@attention: architectures for the safety DRL package
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 1.1.0

#############################################################################################

History:
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

from architectures.replay_buffer import ReplayBuffer
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
        self.minibatch_size = self.params['minibatch_size']
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

        # build neural nets
        self.actor, self.critic = self._build_network()
        self.optimizer = [self._actor_optimizer(), self._critic_optimizer()]
        self.save_model_yaml(architecture='A3C')
        
        self.sess = tf.InteractiveSession()
        keras.backend.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

    def _build_network(self) -> keras.models.Sequential:
        """
        build network with A3C parameters
        Output:
            network (keras.model): neural net for architecture
        """
        # layer_input = keras.Input(batch_shape=(None, self.input_shape))
        layer_input = keras.Input(batch_shape=(None, 100), name='input')
        l_dense = keras.layers.Dense(250, activation='relu', kernel_initializer='glorot_uniform', name='dense')(layer_input)

        out_actions = keras.layers.Dense(self.output_dim, activation='softmax', name='out_a')(l_dense)
        out_value = keras.layers.Dense(1, activation='linear', kernel_initializer='glorot_uniform', name='out_v')(l_dense)

        model = keras.Model(inputs=[layer_input], outputs=[out_actions, out_value])
        actor = keras.Model(inputs=layer_input, outputs=out_actions)
        critic = keras.Model(inputs=layer_input, outputs=out_value)
        actor._make_predict_function()
        critic._make_predict_function()
        model.summary()
        self.model_yaml = model.to_yaml()
        
        if self.debug:
            print(Font.yellow + '–' * 100 + Font.end)
            print(Font.yellow + 'Model: ' + Font.end)
            print(model.to_yaml())
            print(Font.yellow + '–' * 100 + Font.end)

        return actor, critic


    def _actor_optimizer(self):
        """
        make loss function for policy gradient
        backpropagation input: 
            [ log(action_probability) * advantages]
        with:
            advantages = discounted_reward - values
        """
        K = keras.backend
        action = K.placeholder(shape=(None, self.output_dim))
        advantages = K.placeholder(shape=(None, ))

        policy = self.actor.output

        log_prob = K.log( K.sum(policy * action, axis=1) + 1e-10)
        loss_policy = - log_prob * K.stop_gradient(advantages)
        loss = - K.sum(loss_policy)
        
        entropy = self.params['loss_entropy_coefficient'] * K.sum(policy * K.log(policy + 1e-10), axis=1)
        
        loss_actor = K.mean(loss + entropy)

        optimizer = keras.optimizers.RMSprop(lr=self.l_rate,
                                            rho=0.9)
        updates = optimizer.get_updates(loss_actor, self.actor.trainable_weights)
        train = K.function([self.actor.input, action, advantages], [loss_actor], updates=updates)
        return train
    
    def _critic_optimizer(self):
        """
        make loss function for value approximation
        """
        K = keras.backend
        discounted_reward = K.placeholder(shape=(None, ))

        value = self.critic.output
        loss_value = K.mean( K.square( discounted_reward - value))

        optimizer = keras.optimizers.RMSprop(lr=self.l_rate,
                                             rho=0.9)
        updates = optimizer.get_updates(loss_value, self.critic.trainable_weights)
        train = K.function([self.critic.input, discounted_reward], [loss_value], updates=updates)
        return train


    def predict(self, state) -> (np.array, np.array):
        p = self.actor.predict(state)
        v = self.critic.predict(state)
        return p, v
    
    def predict_p(self, state) -> np.array:
        p = self.actor.predict(state)
        return p

    def predict_v(self, state) -> np.array:
        v = self.critic.predict(state)
        return v


    def train(self):
        agents = [A3CAgent(index=i, 
                actor=self.actor, 
                critic=self.critic, 
                optimizer=self.optimizer, 
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
        names = ['Index', 'Episode', 'Epoch',
                 'Reward', 'Step Counter', 'Epsilon']
        while True:
            time.sleep(1)
            t = Texttable()
            for agent in agents:
                index, episode, epoch, rew, step_counter, epsilon, q_vals, loss_actor, loss_critic = agent.get_info()
                t.add_rows([names, [index, episode, epoch, rew, step_counter, epsilon]])
                print(t.draw() + '\n\n')
                scores.append(rew)
                Q_vals.append(q_vals)
                losses_actor.append(loss_actor)
                losses_critic.append(loss_critic)
            with open('scores.yml', 'w') as f:
                yaml.dump(scores, f)
            plot = [np.mean(scores[n:n+5])
                    for n in range(0, len(scores)-5)]
            fig, (ax1, ax2left) = plt.subplots(2, 1, figsize=(18, 12))
            ax1.plot(range(len(plot)), plot, 'r', label='Scores')
            ax1.set_xlabel('Episodes/5')
            ax1.set_ylabel('Mean scores over last 5 episodes')
            legend2left, = ax2left.plot(range(len(losses_actor)), losses_actor, 'b', label='loss actor')
            ax2right = ax2left.twinx()
            legend2right, = ax2right.plot(range(len(losses_critic)), losses_critic, 'r', label='loss critic')
            ax2left.set_xlabel('Loss')
            ax2left.set_ylabel('Episodes')
            ax2right.set_ylabel('Episodes')
            ax1.legend(fontsize=25)
            plt.legend(handles=[legend2left, legend2right], fontsize=25, loc='center right')
            ax1.grid()
            ax2left.grid()
            fig.tight_layout()
            plt.savefig('./a3c.pdf')


    def save_weights(self, path: str) -> None:
        """
        save model weights
        Input:
            path (str): filepath
        """
        self.actor.save_weights(filepath='actor_'+ path)
        self.critic.save_weights(filepath='critic_' + path)

    def main(self):
        print('A3C here')
        print('–' * 30)


class A3CAgent(threading.Thread):
    def __init__(self,
                 index: int,
                 actor,
                 critic,
                 optimizer,
                 env,
                 action_dim: int,
                 state_shape: tuple,
                 params: dict = None) -> None:
        
        super(A3CAgent, self).__init__()
        
        self.index = index
        self.actor = actor
        self.critic = critic
        self.env = env
        self.action_dim = action_dim
        self.state_shape = state_shape
        self.optimizer = optimizer
        self.mc = misc
        self.overblow_factor = 8

        self.states = []
        self.rewards = []
        self.actions = []

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

        names = ['Index', 'Episode', 'Epoch',
                 'Reward', 'Step Counter', 'Epsilon', 'Loss Actor', 'Loss Critic']
        name_tuple = tuple([_ for _ in range(len(names))])
        self.info = [name_tuple]


    def run(self):
        step_counter = 0
        q_vals = 0
        for epoch in range(self.num_epochs):
            for episode in range(self.num_episodes):
                obs, _, _, _ = self.env.reset()
                state = self.mc.make_frame(obs, do_overblow=False,
                                                overblow_factor=self.overblow_factor,
                                                normalization=False).reshape(self.state_shape)
                rews = 0
                for step in range(self.num_steps):
                    action, q_vals = self.act(state)
                    obs, reward, terminal, info = self.env.step(action)
                    self.calc_eps_decay(step_counter=step_counter)
                    rews += reward
                    # self.env.render()
                    next_state = self.mc.make_frame(obs, do_overblow=False,
                                                     overblow_factor=self.overblow_factor,
                                                     normalization=False).reshape(self.state_shape)
                    self.memory(state, action, reward)
                    state = next_state
                    step_counter += 1
                    if step == self.num_steps - 1:
                        terminal = True
                    if terminal:
                        loss_actor, loss_critic = self.train_episode(done=terminal)
                        self.set_info(self.index, episode, epoch, rews, 
                                        step_counter, self.epsilon, q_vals, loss_actor, loss_critic)
                        break

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
        return index, episode, epoch, rew, step_counter, epsilon, q_vals, loss_actor, loss_critic

    def set_info(self, index, episode, epoch, rew, step_counter, epsilon, q_vals, loss_actor, loss_critic) -> None:
        self.info.append((index, episode, epoch, rew, step_counter,
                          epsilon, q_vals, loss_actor, loss_critic))


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

    def memory(self, state: np.array, action: int, reward: float) -> None:
        """
        save <s, a, r, s', terminal> every step
        Input:
            state (np.array): s
            action (int): a
            reward (float): r
            next_state (np.array): s'
            terminal (float): 1.0 if terminal else 0.0
        """
        self.states.append(state)
        act = np.zeros(self.action_dim)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)

    def discount_rewards(self, rewards, done=True):
        """
        calculate discounted rewards
        Input:
            rewards (list): list of rewards
            done (bool): terminal flag
        """
        discounted_rewards = np.zeros_like(rewards)
        running_rew = 0
        if not done:
            running_rew = self.critic.predict(np.reshape(self.states[-1], (1, self.state_shape[0])))[0]
        for t in reversed(range(0, len(rewards))):
            running_rew = running_rew * self.gamma + rewards[t]
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
        discounted_rewards = self.discount_rewards(self.rewards, done)

        values = self.critic.predict(np.array(self.states))
        values = np.reshape(values, len(values))

        advantages = discounted_rewards - values

        loss_actor = self.optimizer[ACTOR]([self.states, self.actions, advantages])
        loss_critic = self.optimizer[CRITIC]([self.states, discounted_rewards])
        if self.debug:
            print(Font.yellow + '-'*100 + Font.end)
            print('loss actor:', loss_actor[0])
            print('loss critic:', loss_critic[0])
            print(Font.yellow + '-'*100 + Font.end)
        # action, advantages, minimize = self.optimizer[ACTOR]
        # discounted_reward, minimize = self.optimizer[CRITIC]
        self.states, self.actions, self.rewards = [], [], []
        return loss_actor[0], loss_critic[0]

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
        policy = self.actor.predict(s)
        action = misc.eps_greedy(policy[0], self.epsilon, rng=self.rng)
        # return np.random.choice(self.action_dim, size=1, p=policy)
        return action, np.max(policy[0])


if __name__ == '__main__':
    print('A3C __main__')
