
"""
Created on October 26, 2018

@author: mae-ma
@attention: basic functionality for RL agents
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 1.0.1

#############################################################################################

History:
- v1.0.1: update output path
- v1.0.0: first init
"""

import os
from architectures.replay_buffer import ReplayBuffer
from architectures.misc import Font

class Agent:
    def __init__(self, parameters):
        self.params = parameters
        self.architecture_path = os.path.dirname(os.path.realpath(__file__))
        self.safety_path = os.path.dirname(self.architecture_path)
        self.output_path = os.path.join(self.safety_path, 'output')
        self.replay_buffer = ReplayBuffer(
            float(self.params['replay_memory_size']))

        # max number of epochs
        self.num_epochs = self.params['num_epochs']
        # number of episodes in one epoch
        self.num_episodes = self.params['num_episodes']
        # number of steps in one episode
        self.num_steps = self.params['num_steps']
        self.model_yaml = None

    def save_model_yaml(self, architecture: str) -> None:
        """
        save model config as yml file
        """
        print(Font.yellow + '–' * 100 + Font.end)
        model_name = 'model_' + architecture + '.yml'
        print('Save model as "{}"'.format(model_name))
        with open(os.path.join('output', model_name), 'w') as file:
            file.write(self.model_yaml)
        print(Font.yellow + '–' * 100 + Font.end)


    def delete_training_log(self, architecture: str) -> None:
        """
        delete training log file in the beginning
        """
        log_name = 'training_log_' + architecture + '.csv'
        if os.path.exists(os.path.join(self.output_path, log_name)):
            print(Font.yellow + '–' * 100 + Font.end)
            print('delete old training log')
            print(Font.yellow + '–' * 100 + Font.end)
            os.remove(os.path.join(self.output_path, log_name))


    def print_params(self, architecture_name: str) -> None:
        """
        print all hyperparamters to terminal
        """
        print(Font.yellow + '–' * 100 + Font.end)
        print(architecture_name, 'parameters:')
        for param in self.params:
            print('-', param, ':', self.params[param])
        print(Font.yellow + '–' * 100 + Font.end)

    def remember(self, state, action: int, reward: int, next_state: int, done: int) -> None:
        """
        Add values to the replay buffer
        Input:
            state (np.array): numpy array of current state
            action (int): scalar value for chosen action
            reward (int): scalar value for received reward
            next_state (np.array): numpy array of next state
            done (int): scalar value if episode is finished
        """
        self.replay_buffer.add(obs_t=state, act=action,
                               rew=reward, obs_tp1=next_state, done=done)

    def save_buffer(self, path: str) -> None:
        """
        save the replay buffer
        Input:
            path (str): where to save the buffer
        """
        self.replay_buffer.dump(file_path=path)

    def save_weights(self, model, path: str) -> None:
        """
        save model weights
        Input:
            model (keras.Sequential): keras model
            path (str): save path
        """
        model.save_weights(filepath=path)
