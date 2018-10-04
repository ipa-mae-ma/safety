
"""
Created on October 1, 2018

@author: mae-ma
@attention: architectures for the safety DRL package
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 1.0.0

#############################################################################################

History:
- v1.0.0: first init
"""

import numpy as np


class MDP(object):
    def __init__(self, P, nS, nA, desc=None):
        # mdp.P[state][action] is a list of tuples (probability, nextstate, reward)
        self.P = P  # state transition and reward probabilities
        self.nS = nS  # number of states
        self.nA = nA  # number of actions
        self.desc = desc  # 2D array specifying what each grid cell means (used for plotting)

    def explanation(self):
        print("mdp.P is a two-level dict where the first key is the state and the second key is the action.")
        print("The 2D grid cells are associated with indices [0, 1, 2, ..., 15] from left to right and top to down, as in")
        print(np.arange(16).reshape(4, 4))
        print("Action indices [0, 1, 2, 3] correspond to West, South, East and North (gridworld example).")
        print("mdp.P[state][action] is a list of tuples (probability, nextstate, reward).\n")
        print("For example, state 0 is the initial state, and the transition information for s=0, a=0 is \nP[0][0] =", mdp.P[0][0], "\n")
        print("As another example, state 5 corresponds to a hole in the ice, in which all actions lead to the same state with probability 1 and reward 0.")


if __name__ == '__main__':
    pass
