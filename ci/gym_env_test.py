
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
import sys, os
sys.path.extend([os.path.split(sys.path[0])[0]])
from architectures import misc as mc
###############################

import numpy as np
import numpy.random as nr
from ci.crawler_env import CrawlingRobotEnv
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
    def test_q_learning_update(self):
        """
        Test for Q-learning update using gym environment "Crawler"
        """
        # testing your q_learning_update implementation
        q_vals = defaultdict(lambda: np.array([0. for _ in range(env.action_space.n)]))
        dummy_q = q_vals.copy()
        test_state = (0, 0)
        test_next_state = (0, 1)
        dummy_q[test_state][0] = 10.
        dummy_q[test_next_state][1] = 10.
        q_vals = mc.q_learning_update(0.9, 0.1, dummy_q, test_state, 0, test_next_state, 1.1)
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

    def test_crawler_env_q_learning(self):
        q_vals = defaultdict(lambda: np.array([0. for _ in range(env.action_space.n)]))
        cur_state = env.reset()
        n = 300000
        for itr in range(n):
            action = mc.eps_greedy(q_vals, EPS, cur_state)
            next_state, reward, done, info = env.step(action)
            mc.q_learning_update(GAMMA, ALPHA, q_vals, cur_state, action, next_state, reward)
            cur_state = next_state
            if itr % 50000 == 0: # evaluation
                print('q_vals:', q_vals[cur_state])
                print("Itr %i # Average speed: %.6f" % (itr, self.greedy_eval(q_vals)))
            if itr == n - 1:
                target = 3.368062
                print("Itr {} # Average speed: {:.6f}".format(itr, self.greedy_eval(q_vals)))
                assert np.isclose(self.greedy_eval(q_vals), target) == True

if __name__ == '__main__':
    tge = TestGymEnv()
    tge.test_crawler_env_q_learning()
