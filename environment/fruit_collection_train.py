
"""
Created on October 1, 2018

@author: mae-ma
@attention: fruit game for the safety DRL package using different architectures
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 2.4.0

#############################################################################################

History:
- v2.4.0: use new environment step output
- v2.3.2: delete framerate slowdown
- v2.3.1: terminal output for training update
- v2.3.0: add modes
- v2.2.2: use shape instead of dim
- v2.2.1: load game length from config file
- v2.2.0: change reward
- v2.1.0: rename of variables / use of different flags
- v2.0.3: add more terminal flags
- v2.0.2: add bool for rendering
- v2.0.1: implement click functionality
- v2.0.0: update main function for better structure
- v1.1.1: save reward output in textfile
- v1.1.0: dqn input fixed -> working, overblow input image
- v1.0.2: first dqn test
- v1.0.1: extend path to find other packages
- v1.0.0: first init
"""

import os
from copy import deepcopy
import pygame
import numpy as np
import time
import yaml
import click
import tqdm
# or FruitCollectionLarge or FruitCollectionMini
from fruit_collection import FruitCollection, FruitCollectionSmall, FruitCollectionLarge, FruitCollectionMini
###############################
# Necessary to import packages from different folders
###############################
import sys
import os
sys.path.extend([os.path.split(sys.path[0])[0]])
############################
# architectures
############################
from architectures.a3c import AsynchronousAdvantageActorCriticGlobal
# from architectures.hra import HybridRewardArchitecture
from architectures.dqn import DeepQNetwork
import architectures.misc as misc
############################

############################
# RGB colors
############################
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 100, 255)
WALL = (80, 80, 80)
############################
np.set_printoptions(threshold=np.nan)


class FruitCollectionTrain(FruitCollection):
    def __init__(self, warmstart=False, simple=True, render=False, testing=False, mode='mini'):
        print('–'*100)
        print('Warmstart:\t', warmstart)
        print('Simple:\t\t', simple)
        print('Mode:\t\t', mode)

        self.params = self.load_params(filename='config_a3c.yml')
        if mode == 'mini':
            params = self.params[mode]
            game_length = params['num_steps']
            self.env = FruitCollectionMini(rendering=render, lives=1, is_fruit=True, is_ghost=False, image_saving=False, game_length=game_length)
        elif mode == 'small':
            params = self.params[mode]
            game_length = params['num_steps']
            self.env = FruitCollectionSmall(rendering=render, lives=1, is_fruit=True, is_ghost=False, image_saving=False, game_length=game_length)
        else:
            raise ValueError('Incorrect mode.')

        self.render = render
        self.env.render()

        self.testing = testing
        self.simple = simple
        self.overblow_factor = 8
        self.input_shape = (self.env.scr_h, self.env.scr_w)  # (img_height, img_width)
        if self.simple:
            # input shape = (img_height * img_width, )
            self.input_shape = (self.input_shape[0] * self.input_shape[1], )
            self.input_dim = self.input_shape[0]
        else:
            self.input_shape = (self.input_shape[0] * self.overblow_factor,
                            self.input_shape[1] * self.overblow_factor)  # (img_height, img_width)
        self.mc = misc
        self.dqn = DeepQNetwork(input_shape=self.input_shape, output_dim=self.env.nb_actions,
                                warmstart=warmstart, warmstart_path='/home/mae-ma/git/safety', 
                                simple_dqn=self.simple, params=params)

        self.a3c = AsynchronousAdvantageActorCriticGlobal(input_shape=self.input_shape,
                                                          output_dim=self.env.nb_actions,
                                                          warmstart=False,
                                                          warmstart_path=None,
                                                          simple_a3c=True,
                                                          params=params,
                                                          env=self.env)

    def load_params(self, filename: str):
        """
        load parameters from the config file
        Input:
            filename (str): file to load
        """
        environment_path = os.path.dirname(os.path.realpath(__file__))
        safety_path = os.path.dirname(environment_path)
        cfg_file = os.path.join(os.path.join(
            safety_path, 'architectures'), filename)
        return yaml.safe_load(open(cfg_file, 'r'))


    def init_with_mode(self):
        self.is_ghost = False
        self.is_fruit = True
        self.reward_scheme = {'ghost': -10.0, 'fruit': +1.0, 'step': 0.0, 'wall': 0.0}
        self.nb_fruits = 4
        self.scr_w = 5
        self.scr_h = 5
        self.possible_fruits = [[0, 0], [0, 4], [4, 0], [4, 4]]
        self.rendering_scale = 50
        self.walls = [[1, 0], [2, 0], [4, 1], [0, 2], [2, 2], [3, 3], [1, 4]]
        if self.is_ghost:
            self.ghosts = [{'colour': RED, 'reward': self.reward_scheme['ghost'], 'location': [0, 1],
                            'active': True}]
        else:
            self.ghosts = []


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
        print(text.format(step_counter, self.dqn.num_steps, 1/mean_v), end='', flush=True)

    def main_dqn(self, verbose=False):
        reward = []
        step_counter = 0
        q_val = 0
        loss = np.zeros((0,3))
        Q_val = np.zeros((0,1))
        for epoch in range(self.dqn.num_epochs):
            for episode in range(self.dqn.num_episodes):
                # states = []
                state, _, _, _ = self.env.reset()
                if self.simple:
                    state = self.mc.make_frame(state, do_overblow=False,
                                           overblow_factor=None,
                                           normalization=True).reshape(self.input_shape)
                else:
                    state = self.mc.make_frame(state, do_overblow=True,
                                           overblow_factor=self.overblow_factor,
                                            normalization=True)[np.newaxis, ..., np.newaxis]
                rew = 0
                timing = [0.0]
                # for step in tqdm.trange(self.dqn.num_steps, unit='Steps', ascii=True):
                for step in range(self.dqn.num_steps):
                    timing.append(time.time())
                    self.training_print(step_counter=step+1, timing_list=timing)

                    action, q_val = self.dqn.act(state)
                    self.dqn.calc_eps_decay(step_counter=step_counter)
                    next_state, r, terminated, info = self.env.step(action)
                    state_low = next_state[2, ...]

                    if self.simple:
                        next_state = self.mc.make_frame(next_state, do_overblow=False, 
                                                     overblow_factor=None,
                                                     normalization=True).reshape(self.input_shape)
                    else:
                        # append grayscale image to state list
                        next_state = self.mc.make_frame(next_state, do_overblow=True, 
                                                        overblow_factor=self.overblow_factor,
                                                        normalization=True)[np.newaxis,..., np.newaxis]
                    self.dqn.remember(state=state, action=action, reward=r, next_state=next_state, done=terminated)
                    self.dqn.do_training(is_testing=self.testing)
                    
                    state = next_state

                    self.env.render()
                    # increase step counter
                    step_counter += 1
                    
                    loss = np.append(loss, np.array([[0,0,self.dqn.loss[0]]]), axis=0)
                    Q_val = np.append(Q_val, np.array([[q_val]]), axis=0)
                    if step_counter % 100 == 0:
                        # np.savetxt('training_log_DQN.csv', loss, fmt='%.4f', delimiter=',')
                        np.savetxt('q_val_DQN.csv', Q_val, fmt='%.4f', delimiter=',')

                    if verbose:
                        print("\033[2J\033[H\033[2J", end="")
                        print()
                        print('pos: ', self.env.player_pos_x, self.env.player_pos_y)
                        print('reward: ', r)
                        print('state:')
                        print(state_low)
                        print('─' * 30)
                        print('─' * 30)

                    # update target model
                    if step_counter % self.dqn.params['target_network_update_frequency'] == 0:
                        self.dqn.update_target_model(soft=False, beta=0.8)
                        if verbose:
                            print('\n' + '–' * 50)
                            print('update after',
                                  self.dqn.params['target_network_update_frequency'], 'steps')
                            print('–' * 50)

                    if step == self.dqn.num_steps - 1:
                        terminated = True
                    if terminated is False:
                        rew += r
                    if terminated is True:
                        rew += r
                        self.dqn.save_buffer(path='replay_buffer.pkl')
                        self.dqn.save_weights(path='weights.h5')
                        print('\nepisode: {}/{} \nepoch: {}/{} \nscore: {} \neps: {:.3f} \nsum of steps: {}'.
                              format(episode, self.dqn.num_episodes, epoch,
                                     self.dqn.num_epochs, rew, self.dqn.epsilon, step_counter))
                        reward.append((rew, step_counter, step))
                        with open('reward.yml', 'w') as f:
                            yaml.dump(reward, f)
                        break

    def main_a3c(self):
        self.a3c.do_training()


@click.command()
@click.option('--warmstart/--no-warmstart', '-w/-nw', default=False, help='load the network weights')
@click.option('--simple/--no-simple', '-s/-ns', default=True, help='uses simple DQN network')
@click.option('--render/--no-render', '-r/-nr', default=False, help='render the pygame')
@click.option('--testing/--no-testing', '-t/-nt', default=False, help='test the network')
@click.option('--mode', '-m', help='environment, possibilities: "mini", "small"')
def run(warmstart, simple, render, testing, mode):
    fct = FruitCollectionTrain(warmstart=warmstart, simple=simple, render=render, testing=testing, mode=mode)
    # fct.main_dqn(verbose=False)
    fct.main_a3c()

if __name__ == '__main__':
    run()
