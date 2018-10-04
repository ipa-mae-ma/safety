
"""
Created on October 4, 2018

@author: mae-ma
@attention: tests for continuous integration
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 1.0.0

#############################################################################################

History:
- v1.0.0: first init
"""

from environment.architectures.misc import *

import numpy as np
import numpy.random as nr
import gym
from crawler_env import CrawlingRobotEnv
from collections import defaultdict
import random
env = CrawlingRobotEnv()

# Important to start the class name with "Test"!
class TestGymEnv:
    def test_q_learning_update(self):
        """
        Test for Q-learning update using gym environment "Crawler"
        """
        q_vals = defaultdict(lambda: np.array([0. for _ in range(env.action_space.n)]))
        # testing your q_learning_update implementation
        dummy_q = q_vals.copy()
        test_state = (0, 0)
        test_next_state = (0, 1)
        dummy_q[test_state][0] = 10.
        dummy_q[test_next_state][1] = 10.
        q_learning_update(0.9, 0.1, dummy_q, test_state, 0, test_next_state, 1.1)
        tgt = 10.01
        if not np.isclose(dummy_q[test_state][0], tgt,):
            print("Q(test_state, 0) is expected to be %.2f but got %.2f" % (tgt, dummy_q[test_state][0]))
        assert np.isclose(dummy_q[test_state][0], tgt,) == True
