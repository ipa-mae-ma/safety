
"""
Created on October 1, 2018

@author: mae-ma
@attention: fruit game for the safety DRL package using different architectures
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 1.1.0

#############################################################################################

History:
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


class FruitCollectionTrain(FruitCollection):
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

    def main(self):
        reward = []
        env = FruitCollectionSmall(rendering=True, lives=1, is_fruit=True, is_ghost=True, image_saving=False)
        env.reset()
        env.render()

        # input_dim = (14, 21, 1) # (img_height, img_width, n_channels)
        input_dim = (88, 88, 1) # (img_height, img_width, n_channels)
        mc = misc
        dqn = DeepQNetwork(env=env, input_size=input_dim, output_size=4, name='DQN')
        a3c = AsynchronousAdvantageActorCritic()
        hra = HybridRewardArchitecture()
        states = []

        for t in range(500):
            action = np.random.choice(env.legal_actions)
            obs, r, terminated, info = env.step(action)
            state_low = obs[2, ...]
            state_high = mc.overblow(input_array=state_low, factor=8)
            state = state_high.reshape(input_dim)
            states.append(state)
            if t >= 1:
                state_t = states[-2]
                state_t1 = states[-1]
                dqn.remember(state=state_t, action=action, reward=r, next_state=state_t1, done=terminated)
            env.render()
            if t == 50:
                dqn.replay()
                dqn.save_buffer(path='replay_buffer.pkl')
                dqn.save_weights(path='weights.h5')

            print("\033[2J\033[H\033[2J", end="")
            print()
            print('pos: ', env.player_pos_x, env.player_pos_y)
            print('reward: ', r)
            print('state:')
            print(state_low)
            print('─' * 30)
            frame = np.zeros(shape=obs[0, ...].shape, dtype=np.float32)
            # wall
            frame[obs[0, ...] != 0] = env.rgb2grayscale(WALL, normalization=False)
            # fruit
            frame[obs[1, ...] != 0] = env.rgb2grayscale(BLUE, normalization=False)
            # pacman
            frame[obs[2, ...] != 0] = env.rgb2grayscale(WHITE, normalization=False)
            # ghosts
            frame[obs[3, ...] != 0] = env.rgb2grayscale(RED, normalization=False)
            print('─' * 30)

            if terminated == False:
                reward.append(r)
            time.sleep(.01)
            if terminated == True:
                print(sum(reward))
                reward = []
                # self.main()


if __name__ == '__main__':
    fct = FruitCollectionTrain()
    fct.main()
