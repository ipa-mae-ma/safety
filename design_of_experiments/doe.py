
"""
Created on December 14, 2018

@author: mae-ma
@attention: architectures for the safety DRL package
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 1.1.0

#############################################################################################

History:
- v1.1.0: add class
- v1.0.4: use global variables for yaml names
- v1.0.3: use dict for all configurations
- v1.0.2: catch if not all possibilities are calculated
- v1.0.1: output all configurations as yaml
- v1.0.0: first init
"""

import yaml
import os.path as path
import itertools
###############################
# Necessary to import packages from different folders
###############################
import sys
import os
sys.path.extend([os.path.split(sys.path[0])[0]])
############################
from environment.fruit_collection_train import FruitCollectionTrain

DOE_YAML = 'doe.yaml'
OUTPUT_YAML = 'experiments.yaml'

class DesignOfExperiments:
    def __init__(self):
        self.file_dir = path.dirname(os.path.realpath(__file__))
        self.output_dir = path.join(self.file_dir, 'output')
        if not path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.design_dict = yaml.load(
            open(path.join(self.file_dir, DOE_YAML), 'r'))

    def create_experiments(self) -> None:
        """
        creates combinations of all possible values
        """
        values = list(self.design_dict.values())
        experiments_tuple = list(itertools.product(*values))
        experiments = {}
        for num, tup in enumerate(experiments_tuple):
            experiments[num] = dict(zip(self.design_dict.keys(), tup))
        if self.how_many_possibilities() == len(experiments):
            yaml.dump(experiments, open(path.join(self.file_dir, OUTPUT_YAML), 'w'))
            print('Succesfully saved all possibilities!')
        else:
            self.experiments = None
            raise ValueError('Not all possibilites are covered... {} Possibilities but only {} are calculated'.format(
                self.how_many_possibilities(), len(experiments)))

    def how_many_possibilities(self) -> int:
        """
        calculates all possibilities
        """
        many = 1
        for value in self.design_dict.values():
            many *= len(value)
        return many



class RunDoE:
    def __init__(self):
        self.doe = DesignOfExperiments()
        self.experiments = yaml.load(open(path.join(self.doe.file_dir, OUTPUT_YAML), 'r'))


    def info(self):
        """
        print info text in terminal
        """
        print('–' * 100)
        print('Run tests for {} possible configurations.'.format(self.doe.how_many_possibilities()))
        print('Variables:')
        print('NAME [ NUM OF CHANGES ]')
        for key in self.doe.design_dict.keys():
            print('- ', key, '[', len(self.doe.design_dict[key]), ']')
        print('–' * 100)

    def run(self):
        for num, experiment in self.experiments.items():
            print(num, experiment)
        # fct = FruitCollectionTrain(warmstart=False, 
        #                             simple=experiment['simple'], 
        #                             render=False, 
        #                             testing=False, 
        #                             mode='mini', 
        #                             architecture=experiment['architecture'], 
        #                             doe_params=experiment)

if __name__ == '__main__':
    doe = DesignOfExperiments()
    doe.create_experiments()
    run_doe = RunDoE()
    run_doe.info()
    run_doe.run()