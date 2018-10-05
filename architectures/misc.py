
"""
Created on October 1, 2018

@author: mae-ma
@attention: miscellaneous functions for the safety DRL package
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 1.0.1

#############################################################################################

History:
- v1.0.1: add functions:
        - policy_iteration
        - compute_qpi
        - compute_vpi
- v1.0.0: first init
"""

import numpy as np


def plot_map(Vs_VI, pis_VI):
    """
    Inputs:
        Vs_VI: list of values for given state
        pis_VI: list of action to chose for given policy
    Outputs:
        -
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


def compute_vpi(pi, mdp, gamma):
    """
    Inputs:
        pi: Policy pi
        mdp: MDP
        gamma: discount factor
    Outputs:
        value_functions
    computes the state-value function $V^{\pi}$ for an arbitrary policy $\pi$.
    Recall that $V^{\pi}$ satisfies the following linear equation:
    $$V^{\pi}(s) = \sum_{s'} P(s,\pi(s),s')[ R(s,\pi(s),s') + \gamma V^{\pi}(s')]$$
    """
    # use pi[state] to access the action that's prescribed by this policy
    # http://aima.cs.berkeley.edu/python/mdp.html
    # https://web.engr.oregonstate.edu/~afern/classes/cs533/notes/infinite-horizon-MDP.pdf
    ###########
    # vectors
    ###########
    V = np.zeros((mdp.nS, 1))  # (s, 1)
    ###########
    # matrizes
    # Ps => mdp.P[state][action] = [(p, s1, r), (...), (...)]
    # Rs => mdp.P[state][action] = [(p, s1, r), (...), (...)]
    ###########
    Ps = np.zeros((mdp.nS, mdp.nS))  # (s, s)
    Rs = np.zeros((mdp.nS, mdp.nS))  # (s, s)
    for state in range(mdp.nS):
        action = pi[state]
        for prob, s1, reward in mdp.P[state][action]:
            Ps[state, s1] = prob
            Rs[state, s1] = reward
    alpha = np.eye(Ps.shape[0]) - Ps * gamma
    beta = Ps * Rs
    print(np.linalg.solve(alpha, beta))
    V = np.linalg.solve(alpha, beta)[:, -1]
    return V


def compute_qpi(vpi, mdp, gamma):
    """
    Inputs:
        vpi: value-function
        mdp: MDP
        gamma: discount factor
    Outputs:
        state-action-value Q for current "pi"
    """
    Qpi = np.zeros([mdp.nS, mdp.nA])
    for s in range(mdp.nS):
        for a in range(mdp.nA):
            for p, s1, r in mdp.P[s][a]:
                Qpi[s, a] += p * (r + gamma * vpi[s1])
    return Qpi


def policy_iteration(mdp, gamma, nIt):
    """
    Inputs:
        mdp: MDP
        gamma: discount factor
        nIt: number of iterations, corresponding to n above
    Outputs:
        (value_functions, policies)
    """
    Vs = []
    pis = []
    pi_prev = np.zeros(mdp.nS, dtype='int')
    pis.append(pi_prev)
    for it in range(nIt):
        # need to compute V^pi for current pi
        vpi = compute_vpi(pis[-1], mdp, gamma)
        # need to compute Q^pi which is the state-action values for current pi
        qpi = compute_qpi(vpi, mdp, gamma)
        pi = qpi.argmax(axis=1)
        Vs.append(vpi)
        pis.append(pi)
        pi_prev = pi
    return Vs, pis


def q_learning_update(gamma, alpha, q_vals, cur_state, action, next_state, reward):
    """
    Inputs:
        gamma: discount factor
        alpha: learning rate
        q_vals: q value table as defaultdict
        cur_state: current state
        action: action taken in current state
        next_state: next state results from taking `action` in `cur_state`
        reward: reward received from this transition
    Performs in-place update of q_vals table to implement one step of Q-learning
    """
    target = reward + gamma * np.max(q_vals[next_state])
    q_vals[cur_state][action] = (1 - alpha) * q_vals[cur_state][action] + alpha * target


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
