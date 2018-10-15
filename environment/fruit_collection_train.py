
"""
Created on October 1, 2018

@author: mae-ma
@attention: fruit game for the safety DRL package using different architectures
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 2.0.0

#############################################################################################

History:
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
# or FruitCollectionLarge or FruitCollectionMini
from fruit_collection import FruitCollection, FruitCollectionSmall, FruitCollectionLarge

###############################
# Necessary to import packages from different folders
###############################
import sys
import os
sys.path.extend([os.path.split(sys.path[0])[0]])
############################
# architectures
############################
from architectures.a3c import AsynchronousAdvantageActorCritic
from architectures.hra import HybridRewardArchitecture
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
    def __init__(self):
        self.env = FruitCollectionSmall(rendering=False, lives=1, is_fruit=True, is_ghost=True, image_saving=False)
        # self.env.render()

        # input_dim = (14, 21, 1) # (img_height, img_width, n_channels)
        self.overblow_factor = 8
        self.input_dim = (88, 88, 1)  # (img_height, img_width, n_channels)
        self.mc = misc
        self.dqn = DeepQNetwork(env=self.env, input_size=self.input_dim, output_size=4, name='DQN')
        self.a3c = AsynchronousAdvantageActorCritic()
        self.hra = HybridRewardArchitecture()

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

    def _reset_targets(self):
        while True:
            self.player_pos_x, self.player_pos_y = np.random.randint(0, self.scr_w), np.random.randint(0, self.scr_h)
            if [self.player_pos_x, self.player_pos_y] not in self.possible_fruits + self.walls:
                break
        self.fruits = []
        self.active_fruits = []
        if self.is_fruit:
            for x in range(self.scr_w):
                for y in range(self.scr_h):
                    self.fruits.append({'colour': BLUE, 'reward': self.reward_scheme['fruit'],
                                        'location': [x, y], 'active': False})
                    self.active_fruits.append(False)
            fruits_idx = deepcopy(self.possible_fruits)
            np.random.shuffle(fruits_idx)
            fruits_idx = fruits_idx[:self.nb_fruits]
            self.mini_target = [False] * len(self.possible_fruits)
            for f in fruits_idx:
                idx = f[1] * self.scr_w + f[0]
                self.fruits[idx]['active'] = True
                self.active_fruits[idx] = True
                self.mini_target[self.possible_fruits.index(f)] = True

    def main(self, verbose=False):
        reward = []
        for e in range(self.dqn.episode_max_len):
            states = []
            ep_reward = []
            self.env.reset()

            for t in range(500):
                if t == 0:
                    action = np.random.choice(self.env.legal_actions)
                else:
                    action = self.dqn.act(states[-1])
                obs, r, terminated, info = self.env.step(action)
                state_low = obs[2, ...]
                # state_high = mc.overblow(input_array=state_low, factor=overblow_factor)
                # state = state_high.reshape(input_dim)
                # append grayscale image to state list
                states.append(self.mc.make_frame(obs, do_overblow=True, overblow_factor=self.overblow_factor))
                # states.append(state)
                if t >= 1:
                    state_t = states[-2]
                    state_t1 = states[-1]
                    self.dqn.remember(state=state_t, action=action, reward=r, next_state=state_t1, done=terminated)
                # self.env.render()

                if verbose:
                    print("\033[2J\033[H\033[2J", end="")
                    print()
                    print('pos: ', self.env.player_pos_x, self.env.player_pos_y)
                    print('reward: ', r)
                    print('state:')
                    print(state_low)
                    print('─' * 30)
                    print('─' * 30)

                if terminated == False:
                    reward.append(r)
                    # time.sleep(.1)
                if terminated == True:
                    self.dqn.do_training(is_testing=False)
                    self.dqn.save_buffer(path='replay_buffer.pkl')
                    self.dqn.save_weights(path='weights.h5')
                    print('episode: {}/{}, score: {}'.format(e, self.dqn.episode_max_len, sum(ep_reward)))
                    reward.append(sum(ep_reward))
                    ep_reward = []
                    with open('reward.yml', 'w') as f:
                        yaml.dump(reward, f)
                    break


if __name__ == '__main__':
    fct = FruitCollectionTrain()
    fct.main(verbose=False)
