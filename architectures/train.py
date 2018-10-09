"""
Created on October 9, 2018

@author: mae-ma
@attention: train the architectures for the safety DRL package
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 1.0.0

#############################################################################################

History:
- v1.0.0: first init
"""

import os
import click
import yaml
import numpy as np

from architectures.dqn import DeepQNetwork
from architectures.mdp import MDP
from architectures.misc import Font
from environment.fruit_collection import FruitCollectionMini

np.set_printoptions(suppress=True, linewidth=200, precision=2)
floatX = 'float32'


def worker(params):
    np.random.seed(seed=params['random_seed'])
    random_state = np.random.RandomState(params['random_seed'])
    env = FruitCollectionMini(rendering=False, game_length=300, state_mode='mini')
    params['reward_dim'] = len(env.possible_fruits)
    for experiment in range(params['nb_experiments']):
        print('\n')
        print(Font.bold + Font.red + '>>>>> Experiment ', experiment, ' >>>>>' + Font.end)
        print('\n')

        dqn = DeepQNetwork(env=env, input_size: tuple=(84, 84, 4), output_size: int=4, name: str='DQN')
        env.reset()
        if not params['test']:
            dqn.do_training(total_eps=params['total_eps'], eps_per_epoch=params['eps_per_epoch'],
                             eps_per_test=params['eps_per_test'], is_learning=True, is_testing=True)
        else:
            raise NotImplementedError


@click.command()
@click.option('--mode', default='all', help='Which method to run: dqn, hra, a3c, all')
@click.option('--options', '-o', multiple=True, nargs=2, type=click.Tuple([str, str]))
def run(mode, options):
    valid_modes = ['dqn', 'hra', 'a3c']
    assert mode in valid_modes
    modes = [mode]

    dir_path = os.path.dirname(os.path.realpath(__file__))
    cfg_file = os.path.join(dir_path, 'config.yaml')
    params = yaml.safe_load(open(cfg_file, 'r'))
    # replacing params with command line options
    for opt in options:
        assert opt[0] in params
        dtype = type(params[opt[0]])
        if dtype == bool:
            new_opt = False if opt[1] != 'True' else True
        else:
            new_opt = dtype(opt[1])
        params[opt[0]] = new_opt

    for m in modes:
        worker(params)


if __name__ == '__main__':
    run()
