
"""
Created on December 14, 2018

@author: mae-ma
@attention: architectures for the safety DRL package
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 1.0.3

#############################################################################################

History:
- v1.0.3: use dict for all configurations
- v1.0.2: catch if not all possibilities are calculated
- v1.0.1: output all configurations as yml
- v1.0.0: first init
"""

import yaml
import os
import os.path as path
import itertools


class DesignOfExperiments:
    def __init__(self):
        self.file_dir = path.dirname(os.path.realpath(__file__))
        self.output_dir = path.join(self.file_dir, 'output')
        if not path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.design_dict = yaml.load(
            open(path.join(self.file_dir, 'doe.yaml'), 'r'))

    def create_experiments(self):
        """
        creates combinations of all possible values
        """
        values = list(self.design_dict.values())
        experiments_tuple = list(itertools.product(*values))
        experiments = {}
        for num, tup in enumerate(experiments_tuple):
            experiments[num] = dict(zip(self.design_dict.keys(), tup))
        if self.how_many_possibilities() == len(experiments):
            yaml.dump(experiments, open(path.join(self.file_dir, 'experiments.yml'), 'w'))
            print('Succesfully saved all possibilities!')
        else:
            raise ValueError('Not all possibilites are covered... {} Possibilities but only {} are calculated'.format(
                self.how_many_possibilities(), len(experiments)))

    def how_many_possibilities(self):
        many = 1
        for value in self.design_dict.values():
            many *= len(value)
        return many


if __name__ == '__main__':
    doe = DesignOfExperiments()
    doe.create_experiments()
    doe.how_many_possibilities()
