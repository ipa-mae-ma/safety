
"""
Created on October 1, 2018

@author: mae-ma
@attention: miscellaneous functions for the safety DRL package
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 1.0.0

#############################################################################################

History:
- v1.0.0: first init
"""

import numpy as np


def plot_map(Vs_VI, pis_VI):
    """
    @Vs_VI: list of values for given state
    @pis_VI: list of action to chose for given policy
    This illustrates the progress of value iteration.
    Your optimal actions are shown by arrows.
    At the bottom, the value of the different states are plotted.
    """
    for (V, pi) in zip(Vs_VI[:10], pis_VI[:10]):
        plt.figure(figsize=(3, 3))
        plt.imshow(V.reshape(4, 4), cmap='gray', interpolation='none', clim=(0, 1))
        ax = plt.gca()
        ax.set_xticks(np.arange(4) - .5)
        ax.set_yticks(np.arange(4) - .5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        Y, X = np.mgrid[0:4, 0:4]
        a2uv = {0: (-1, 0), 1: (0, -1), 2: (1, 0), 3: (-1, 0)}
        Pi = pi.reshape(4, 4)
        for y in range(4):
            for x in range(4):
                a = Pi[y, x]
                u, v = a2uv[a]
                plt.arrow(x, y, u * .3, -v * .3, color='m', head_width=0.1, head_length=0.1)
                plt.text(x, y, str(env.desc[y, x].item().decode()),
                         color='g', size=12,  verticalalignment='center',
                         horizontalalignment='center', fontweight='bold')
        plt.grid(color='b', lw=2, ls='-')
    plt.figure()
    plt.plot(Vs_VI)
    plt.title("Values of different states")


def value_iteration(mdp, gamma, nIt):
    """
    Inputs:
        mdp: MDP
        gamma: discount factor
        nIt: number of iterations, corresponding to n above
    Outputs:
        (value_functions, policies)

    len(value_functions) == nIt+1 and len(policies) == nIt
    """
    Vs = [np.zeros(mdp.nS)]  # list of value functions contains the initial value function V^{(0)}, which is zero
    pis = []
    for it in range(nIt):
        oldpi = pis[-1] if len(pis) > 0 else None  # \pi^{(it)} = Greedy[V^{(it-1)}]. Just used for printout
        Vprev = Vs[-1]  # V^{(it)}

        # pi: greedy policy for Vprev (not V),
        #     corresponding to the math above: \pi^{(it)} = Greedy[V^{(it)}]
        #     ** it needs to be numpy array of ints **
        # V: bellman backup on Vprev
        #     corresponding to the math above: V^{(it+1)} = T[V^{(it)}]
        #     ** numpy array of floats **

        # set V to zero at the beginning
        V = np.zeros(mdp.nS)
        # sum over s
        for state in range(mdp.nS):
            # matrix for all actions possible in the current state
            V_action = np.zeros(mdp.nA)
            # for the current P[state] = dict{0: [(tuple with possible s' for action 0),
            # 1: [(tuple with possible s' for action 1)],
            # 2: [(tuple with possible s' for action 2)],...}
            for action in mdp.P[state]:
                # action $\in$ [0:mdp.nA]
                # action = [(prob, s', reward), (...), (...)]
                for prob, next_state, reward in mdp.P[state][action]:  # unzip mdp.P[state][action] = [(p, s', r)]
                    # print(prob)
                    # print(next_state)
                    # print(reward)
                    V_action[action] += prob * (reward + gamma * Vprev[next_state])
            V[state] = np.max(V_action)

        # set pi to zero at the beginning
        pi = np.zeros(mdp.nS)
        # sum over s
        for state in range(mdp.nS):
            # matrix for all actions possible in the current state
            V_action = np.zeros(mdp.nA)
            for action in mdp.P[state]:
                for prob, next_state, reward in mdp.P[state][action]:
                    V_action[action] += prob * (reward + gamma * Vprev[next_state])
            pi[state] = np.argmax(V_action)

        Vs.append(V)
        pis.append(pi)
    return Vs, pis


def eps_greedy(q_vals, eps, state):
    """
    Inputs:
        q_vals: q value tables
        eps: epsilon
        state: current state
    Outputs:
        random action with probability of eps; argmax Q(s, .) with probability of (1-eps)
    """
    # you might want to use random.random() to implement random exploration
    #   number of actions can be read off from len(q_vals[state])
    import random
    if random.random() <= eps:
        action = random.randint(0, len(q_vals[state]) - 1)
    else:
        action = np.argmax(q_vals[state])
    return action
