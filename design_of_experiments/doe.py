
"""
Created on December 14, 2018

@author: mae-ma
@attention: architectures for the safety DRL package
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 1.2.0

#############################################################################################

History:
- v1.2.0: save results
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
import datetime
import shutil
import time
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



class RunDesignOfExperiments:
    def __init__(self):
        self.doe = DesignOfExperiments()
        self.experiments = yaml.load(open(path.join(self.doe.file_dir, OUTPUT_YAML), 'r'))
        self.src_filepath = os.getcwd()
        self.tgt_filepath = path.join(self.src_filepath, 'results')

        if not path.exists(self.tgt_filepath):
            os.makedirs(self.tgt_filepath)


    def info(self) -> None:
        """
        print info text in terminal
        """
        print('–' * 100)
        print('Run tests for {} possible configurations.'.format(self.doe.how_many_possibilities()))
        print('Variables:')
        print('NAME [NUM OF CHANGES]')
        for key in self.doe.design_dict.keys():
            print('- ', key, '[' + str(len(self.doe.design_dict[key])) + ']')
        print('–' * 100)

    def run(self) -> None:
        """
        run the experiments
        """
        return
        global_time = time.time()
        for ex_number, experiment in self.experiments.items():
            start_time = time.time()
            architecture = experiment['architecture']
            # fct = FruitCollectionTrain(warmstart=False, 
            #                             simple=experiment['simple'], 
            #                             render=False, 
            #                             testing=False, 
            #                             mode='mini', 
            #                             architecture=architecture, 
            #                             doe_params=experiment)
            self.save_results(architecture=architecture, experiment_number=ex_number)
            time.sleep(1)
            print(('–' * 100 + '\n') * 2)
            print('>>> Time for experiment: {:.3f} min'.format((time.time() - start_time)/60))
            print(('–' * 100 + '\n') * 2)
        print(('–' * 100 + '\n') * 2)
        print('>>> Overall time for experiments: {:.3f} min'.format((time.time() - global_time)/60))
        print(('–' * 100 + '\n') * 2)

    def save_results(self, architecture: str=None, experiment_number:int=None, game_mode: str='mini') -> None:
        """
        save the results of the experiment
        """
        if architecture is None or experiment_number is None:
            raise ValueError('Please provide a correct architecture name or experiment number')
        
        filelist = ['output/reward.yml', 
                    'output/training_log_' + architecture + '.csv',
                    'architectures/config_' + architecture + '.yml',
                    'output/model_' + architecture + '.yml']
        # temp_dir?

        output_folder_name = datetime.datetime.today().strftime('%Y_%m_%d-%H_%M') + '___' + architecture + '_' + str(experiment_number)
        output_folder_path = os.path.join(self.tgt_filepath, output_folder_name)
        print('>>> Save all files to: ' + output_folder_path)
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
        for file in filelist:
            shutil.copy2(os.path.join(self.src_filepath, file), output_folder_path)



if __name__ == '__main__':
    doe = DesignOfExperiments()
    doe.create_experiments()
    run_doe = RunDesignOfExperiments()
    run_doe.info()
    run_doe.run()