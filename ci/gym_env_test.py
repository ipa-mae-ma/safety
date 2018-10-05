
"""
Created on October 4, 2018

@author: mae-ma
@attention: tests for continuous integration
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 1.0.0

#############################################################################################

This class tests the different Q-learning functions with the help of the gym environment

History:
- v1.0.0: first init
"""
###############################
# Necessary to import packages from different folders
###############################
import sys
import os
sys.path.extend([os.path.split(sys.path[0])[0]])
from architectures import misc as mc
from architectures.mdp import MDP
###############################

import numpy as np
import numpy.random as nr
from ci.crawler_env import CrawlingRobotEnv
from ci.frozen_lake import FrozenLakeEnv
import gym
from collections import defaultdict
import random
env = CrawlingRobotEnv()
GAMMA = 0.9
ALPHA = 0.1
EPS = 0.5

# Important to start the class name with "Test"!


class TestGymEnv:
    # tweaked for testing from the "Deep Reinforcement Learning Bootcamp" by Berkeley
    # https://sites.google.com/view/deep-rl-bootcamp/lectures
    def test_tabular_q_learning_update(self):
        """
        Test for Q-learning update using gym environment "Crawler"
        """
        # testing your tabular_q_learning_update implementation
        q_vals = defaultdict(lambda: np.array([0. for _ in range(env.action_space.n)]))
        dummy_q = q_vals.copy()
        test_state = (0, 0)
        test_next_state = (0, 1)
        dummy_q[test_state][0] = 10.
        dummy_q[test_next_state][1] = 10.
        q_vals = mc.tabular_q_learning_update(0.9, 0.1, dummy_q, test_state, 0, test_next_state, 1.1)
        tgt = 10.01
        if not np.isclose(dummy_q[test_state][0], tgt,):
            print("Q(test_state, 0) is expected to be %.2f but got %.2f" % (tgt, q_vals[test_state][0]))
        assert np.isclose(dummy_q[test_state][0], tgt,) == True

    def greedy_eval(self, q_vals):
        """evaluate greedy policy w.r.t current q_vals"""
        test_env = CrawlingRobotEnv(horizon=np.inf)
        prev_state = test_env.reset()
        ret = 0.
        done = False
        H = 100
        for i in range(H):
            # print('q_vals[prev_state]:', q_vals[prev_state])
            action = np.argmax(q_vals[prev_state])
            state, reward, done, info = test_env._step(action)
            ret += reward
            prev_state = state
        return ret / H

    def test_crawler_env_tabular_q_learning(self):
        q_vals = defaultdict(lambda: np.array([0. for _ in range(env.action_space.n)]))
        cur_state = env.reset()
        n = 300000
        for itr in range(n):
            action = mc.eps_greedy(q_vals, EPS, cur_state)
            next_state, reward, done, info = env.step(action)
            mc.tabular_q_learning_update(GAMMA, ALPHA, q_vals, cur_state, action, next_state, reward)
            cur_state = next_state
            if itr % 50000 == 0:  # evaluation
                print('q_vals:', q_vals[cur_state])
                print("Itr %i # Average speed: %.6f" % (itr, self.greedy_eval(q_vals)))
            if itr == n - 1:
                # target from UC berkeley bootcamp
                TARGET = 3.368062
                print("Itr {} # Average speed: {:.6f}".format(itr, self.greedy_eval(q_vals)))
                assert np.isclose(self.greedy_eval(q_vals), TARGET) == True

    def test_frozen_lake_value_iteration(self):
        # setup environment
        env = FrozenLakeEnv()
        GAMMA_VI = 0.95
        # setup mdp
        mdp = MDP(P={s: {a: [tup[:3] for tup in tups] for (a, tups) in a2d.items()} for (s, a2d) in env.P.items()}, nS=env.nS, nA=env.nA, desc=env.desc)
        # perform value iteration
        Vs_VI, pis_VI = mc.value_iteration(mdp, gamma=GAMMA_VI, nIt=20)
        # value function target
        V_TARGET = [0.0,
                 0.0,
                 0.0,
                 0.0,
                 0.0,
                 0.0,
                 0.25355253760000007,
                 0.3450850036736001,
                 0.4416517553812481,
                 0.4782166872659994,
                 0.5059316882528422,
                 0.5170370602354525,
                 0.5243920128825474,
                 0.5274888052086196,
                 0.5293922251765532,
                 0.5302269359874546,
                 0.5307158047762821,
                 0.5309372906257446,
                 0.5310626746211281,
                 0.5311209619007249,
                 0.531153142835031,
                 ]
        for i in range(1, len(Vs_VI)):
            # print output if assert fails
            print('Vi==V_TARGET: ', np.isclose(Vs_VI[i][0], V_TARGET[i], rtol=1.e-4))
            print(i, ' VI: ', np.round(Vs_VI[i][0], 5), '\t V_TARGET: ', np.round(V_TARGET[i], 5))
            print(i, ' max: ', np.round(max(Vs_VI[i] - Vs_VI[i - 1]), 5))
            assert True == np.isclose(Vs_VI[i][0], V_TARGET[i], rtol=1.e-3)
            print('–' * 50)


if __name__ == '__main__':
    tge = TestGymEnv()
    tge.test_frozen_lake_value_iteration()
