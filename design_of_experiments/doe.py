
"""
Created on December 14, 2018

@author: mae-ma
@attention: architectures for the safety DRL package
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 1.0.0

#############################################################################################

History:
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
        self.design_dict = yaml.load(open(path.join(self.file_dir, 'doe.yaml'), 'r'))


    def create_experiments(self):
        values = list(self.design_dict.values())
        ex = list(itertools.product(*values))
        print(len(ex))
        experiments = []
        for tup in ex:
            experiments.append(dict(zip(self.design_dict.keys(), tup)))

        for ex in experiments:
            print('-', ex)
            for value in ex.values():
                
        

if __name__ == '__main__':
    doe = DesignOfExperiments()
    doe.create_experiments()